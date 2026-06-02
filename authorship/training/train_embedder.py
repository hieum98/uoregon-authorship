"""Embedder training script with Hydra config and Lightning Fabric.

Usage:
    python -m authorship.training.train_embedder --config-path ../../configs/embedder
"""

import functools
import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from authorship.data.embedder_dataset import EmbedderDataModule
from authorship.models.embedder import EmbeddingModel
from authorship.training.gradcache import GradCacheTrainer
from authorship.training.schedulers import get_cosine_annealing_schedule_with_warmup


def _get_wrapping_policy(transformer_layer_cls):
    """FSDP wrapping policy for LoRA layers and transformer blocks."""
    def lambda_fn(module):
        return (
            len(list(module.named_children())) == 0
            and getattr(module, "weight", None) is not None
            and module.weight.requires_grad
        )
    return functools.partial(
        _or_policy,
        policies=[
            functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_fn),
            functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={transformer_layer_cls}),
        ],
    )


def _resolve_decoder_layer_cls(model_name: str):
    """Resolve the decoder layer class for FSDP wrapping."""
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    model_type = config.model_type

    layer_map = {}
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
        layer_map["qwen2"] = Qwen2DecoderLayer
    except ImportError:
        pass
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3DecoderLayer
        layer_map["qwen3"] = Qwen3DecoderLayer
    except ImportError:
        pass
    try:
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer
        layer_map["mistral"] = MistralDecoderLayer
    except ImportError:
        pass
    try:
        from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer
        layer_map["gemma3_text"] = Gemma3DecoderLayer
        layer_map["gemma3"] = Gemma3DecoderLayer
    except ImportError:
        pass
    try:
        from transformers.models.roberta.modeling_roberta import RobertaLayer
        layer_map["roberta"] = RobertaLayer
    except ImportError:
        pass
    try:
        from transformers.models.bert.modeling_bert import BertLayer
        layer_map["bert"] = BertLayer
    except ImportError:
        pass

    return layer_map.get(model_type)


def _get_trainable_params(model):
    trainable, total, layers = 0, 0, []
    for name, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
            layers.append(name)
    return trainable, total, 100 * trainable / max(total, 1), layers


def _trainable_filter(key, value, trainable_layers=None):
    if trainable_layers is None:
        return True
    return any(layer in key for layer in trainable_layers)


def get_dataloaders(fabric, data_module, config, epoch: int = 0):
    tokenizer = EmbeddingModel._build_tokenizer(config.model.model_name_or_path)
    data_module.connect(
        world_size=fabric.world_size,
        global_rank=fabric.global_rank,
        tokenizer=tokenizer,
        global_batch_size=config.data.global_batch_size,
        max_seq_length=config.data.max_seq_length,
        num_train_examples=config.data.get("num_train_examples", -1),
        num_hard_negatives=config.data.get("num_hard_negatives", 256),
    )
    if fabric.is_global_zero:
        data_module.prepare_data(checkpoint_dir=config.training.output_dir)
    fabric.barrier()
    with fabric.rank_zero_first():
        data_module.set_epoch(epoch)
        data_module.setup(checkpoint_dir=config.training.output_dir)
        train_loader = data_module.train_dataloader()
        train_loader = fabric.setup_dataloaders(
            train_loader, 
            use_distributed_sampler=False,
            move_to_device=True
            )
    return train_loader


def main(fabric: L.Fabric, data_module: EmbedderDataModule, config: DictConfig):
    model = EmbeddingModel(
        model_name_or_path=config.model.model_name_or_path,
        use_lora=config.model.lora.use_lora,
        dropout_prob=config.model.get("dropout", 0.1),
        lora_r=config.model.lora.get("r", 16),
        lora_alpha=config.model.lora.get("alpha", 32),
        lora_dropout=config.model.lora.get("dropout", 0.1),
        target_modules=list(config.model.lora.get("target_modules", ["all"])),
        adapter_name=config.model.lora.get("name"),
        quantization=config.model.get("quantization", False),
        attn_implementation=config.model.get("attn_implementation"),
        pooling_method=config.model.get("pooling", "mean"),
        is_bidirectional=config.model.get("is_bidirectional", False),
    )
    ckpt = config.training.get("resume_checkpoint")
    if ckpt and os.path.exists(ckpt):
        ckpt_path = Path(ckpt)
        if ckpt_path.is_dir() and (ckpt_path / "model.safetensors.index.json").exists():
            from safetensors.torch import load_file
            import json as _json
            index = _json.loads((ckpt_path / "model.safetensors.index.json").read_text())
            state_dict = {}
            for shard in sorted(set(index["weight_map"].values())):
                state_dict.update(load_file(str(ckpt_path / shard), device="cpu"))
        else:
            raw = torch.load(ckpt, map_location="cpu")
            state_dict = raw["model"] if isinstance(raw, dict) and "model" in raw else raw
        info = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint {ckpt} (missing: {info.missing_keys[:5]}...)")

    train_data = get_dataloaders(fabric, data_module, config)

    trainable, total, pct, layers = _get_trainable_params(model)
    fabric.print(f"Trainable: {trainable:,}/{total:,} ({pct:.1f}%)")


    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.get("weight_decay", 0.01),
    )

    steps_per_epoch = len(train_data)
    total_steps = steps_per_epoch * config.training.get("epochs", 1)
    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.get("warmup_steps", 100),
        num_training_steps=total_steps,
        num_cycles=config.training.get("epochs", 1),
        min_reduce_rate=config.training.get("min_lr_rate", 0.0) / config.training.lr,
    )

    model, optimizer = fabric.setup(model, optimizer)

    fabric.print(f"Model after setup: \n{model}")

    
    state = {"optimizer": optimizer, "scheduler": scheduler, "iter_num": 0, "epoch_num": 0}
    if ckpt and os.path.exists(ckpt):
        if not config.training.get("only_reload_model", True): # Reload optimizer and scheduler
            fabric.load(ckpt, state, strict=False)


    trainer = GradCacheTrainer(fabric, chunk_size=config.training.get("gc_chunk_size", 48))

    for epoch in range(state.get("epoch_num", 0), config.training.get("epochs", 1)):
        state["epoch_num"] = epoch
        ckpt_filter = functools.partial(_trainable_filter, trainable_layers=layers)
        if epoch > 0:
            train_data = get_dataloaders(fabric, data_module, config, epoch) # Reload dataloader for each epoch to get new sampler
        trainer.fit_epoch(
            model, train_data, state,
            temperature=config.loss.get("temperature", 0.05),
            use_miner=config.loss.get("use_miner", False),
            normalize=config.loss.get("normalize", True),
            lr_max_steps=total_steps,
            grad_norm_clip=config.training.get("grad_clip", 1.0),
            checkpoint_interval=config.training.get("checkpoint_interval", 1000),
            checkpoint_dir=config.training.output_dir,
            checkpoint_filter=ckpt_filter,
        )


def _choose_logger(config: DictConfig, output_dir: Path):
    logger_cfg = config.training.get("logger", {})
    logger_type = logger_cfg.get("type", "csv") if logger_cfg else "csv"
    run_name = config.get("run_name", "embedder")

    if logger_type == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        logger_dir = output_dir / "logs_wandb"
        logger_dir.mkdir(parents=True, exist_ok=True)
        return WandbLogger(
            name=run_name,
            project=logger_cfg.get("project", "authorship-embedder"),
            save_dir=str(logger_dir),
        )

    from lightning.fabric.loggers import CSVLogger
    return CSVLogger(root_dir=str(output_dir / "logs"), name=run_name)


def setup(config: DictConfig):
    data_module = EmbedderDataModule(
        dataset_names=list(config.data.dataset_names),
        languages=list(config.data.get("languages", ["en"])),
        seed=config.get("seed", 888),
        num_workers=config.get("num_workers", 4),
    )

    output_dir = Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    decoder_cls = _resolve_decoder_layer_cls(config.model.model_name_or_path)
    strategy = "auto"
    if config.training.get("devices", 1) > 1 and decoder_cls:
        from lightning.fabric.strategies import FSDPStrategy
        import datetime
        strategy = FSDPStrategy(
            auto_wrap_policy=_get_wrapping_policy(decoder_cls) if config.model.lora.get("use_lora", False) else {decoder_cls},
            state_dict_type="full",
            timeout=datetime.timedelta(days=1),
            )

    logger = _choose_logger(config, output_dir)

    fabric = L.Fabric(
        accelerator="auto",
        devices=config.training.get("devices", 1),
        precision=config.training.get("precision", "bf16-mixed"),
        strategy=strategy,
        loggers=[logger],
    )

    OmegaConf.save(config, str(output_dir / "config.yaml"))

    fabric.launch(main, data_module=data_module, config=config)


@hydra.main(version_base=None, config_path="../../configs/embedder", config_name="default")
def hydra_main(config: DictConfig):
    setup(config)


if __name__ == "__main__":
    hydra_main()
