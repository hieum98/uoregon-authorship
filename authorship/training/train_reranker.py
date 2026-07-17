"""Reranker training script with Hydra config and Lightning Fabric.

Usage:
    python -m authorship.training.train_reranker --config-path ../../configs/reranker
"""

import functools
import os
import pathlib
import time
from typing import Any

import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from authorship.data.reranker_dataset import RerankerDataModule
from authorship.models.reranker import RerankerModel
from authorship.training.losses import compute_reranker_loss, compute_reranker_metrics
from authorship.training.schedulers import get_cosine_annealing_schedule_with_warmup


def _resolve_decoder_layer_cls(model_name: str):
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

    return layer_map.get(model_type)


def _get_wrapping_policy(transformer_layer_cls):
    return functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={transformer_layer_cls},
    )


def train_step(
    fabric,
    model,
    batch,
    label_smoothing: float = 0.0,
    ranking_loss_weight: float = 0.0,
    ranking_margin: float = 0.0,
    listwise_loss_weight: float = 0.0,
    listwise_temperature: float = 1.0,
    pos_weight: float = 1.0,
):
    outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    loss = compute_reranker_loss(
        outputs["true_logits"], outputs["false_logits"], batch["labels"],
        label_smoothing=label_smoothing,
        group_ids=batch.get("group_ids"),
        ranking_loss_weight=ranking_loss_weight,
        ranking_margin=ranking_margin,
        listwise_loss_weight=listwise_loss_weight,
        listwise_temperature=listwise_temperature,
        pos_weight=pos_weight,
    )
    return loss, outputs


def fit_epoch(
    fabric, model, optimizer, scheduler, train_loader, config, epoch,
):
    grad_accum = config.training.get("gradient_accumulation_steps", 1)
    checkpoint_interval = config.training.get("checkpoint_interval", 2000)
    output_dir = config.training.output_dir

    label_smoothing = config.training.get("label_smoothing", 0.0)
    ranking_loss_weight = config.training.get("ranking_loss_weight", 0.0)
    ranking_margin = config.training.get("ranking_margin", 0.0)
    listwise_loss_weight = config.training.get("listwise_loss_weight", 0.0)
    listwise_temperature = config.training.get("listwise_temperature", 1.0)
    pos_weight = config.training.get("pos_weight", 1.0)
    model.train()
    for batch_idx, batch in enumerate(train_loader):
        t0 = time.perf_counter()
        loss, outputs = train_step(
            fabric,
            model,
            batch,
            label_smoothing=label_smoothing,
            ranking_loss_weight=ranking_loss_weight,
            ranking_margin=ranking_margin,
            listwise_loss_weight=listwise_loss_weight,
            listwise_temperature=listwise_temperature,
            pos_weight=pos_weight,
        )
        fabric.backward(loss / grad_accum)

        if (batch_idx + 1) % grad_accum == 0:
            if config.training.get("grad_clip"):
                fabric.clip_gradients(model, optimizer, max_norm=config.training.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()

        total_step = epoch * len(train_loader) + batch_idx
        if batch_idx % 10 == 0:
            metrics = compute_reranker_metrics(
                outputs["true_logits"].detach(),
                outputs["false_logits"].detach(),
                batch["labels"],
                group_ids=batch.get("group_ids"),
            )
            lr = scheduler.get_last_lr()[0] if scheduler else optimizer.param_groups[0]["lr"]
            fabric.log_dict({"loss": loss.item(), "lr": lr, **metrics}, step=total_step)
            fabric.print(
                f"Epoch {epoch} | Step {batch_idx} | Loss: {loss.item():.4f} | "
                f"Acc: {metrics['accuracy']:.3f} | F1: {metrics['f1']:.3f} | "
                f"Pgap: {metrics['p_yes_gap']:+.3f} | "
                f"PairAcc: {metrics.get('pairwise_accuracy', 0.0):.3f} | LR: {lr:.2e}"
            )

        if batch_idx > 0 and batch_idx % checkpoint_interval == 0:
            ckpt_path = pathlib.Path(output_dir) / f"checkpoint_step{total_step}.ckpt"
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            fabric.print(f"Saving checkpoint: {ckpt_path}")
            fabric.barrier()
            # Release cached/fragmented memory before FSDP state collection.
            import gc
            del loss, outputs
            gc.collect()
            torch.cuda.empty_cache()
            save_state = {"model": model, "epoch": epoch, "step": batch_idx}
            if config.training.get("checkpoint_save_optimizer", False):
                save_state["optimizer"] = optimizer
            if config.training.get("checkpoint_save_scheduler", False):
                save_state["scheduler"] = scheduler
            fabric.save(ckpt_path, save_state)
            pathlib.Path(str(ckpt_path) + ".done").touch()
            fabric.print(f"Saved checkpoint: {ckpt_path}")
            fabric.barrier()


def main(fabric, config):
    data_module = RerankerDataModule(
        dataset_names=list(config.data.dataset_names),
        languages=list(config.data.get("languages", ["en"])),
        seed=config.get("seed", 888),
        num_workers=config.get("num_workers", 4),
    )

    model = RerankerModel(
        model_name_or_path=config.model.model_name_or_path,
        use_lora=config.model.lora.use_lora,
        lora_r=config.model.lora.get("r", 16),
        lora_alpha=config.model.lora.get("alpha", 32),
        lora_dropout=config.model.lora.get("dropout", 0.1),
        target_modules=list(config.model.lora.get("target_modules", ["all"])),
        adapter_name=config.model.lora.get("name"),
        quantization=config.model.get("quantization", False),
        attn_implementation=config.model.get("attn_implementation"),
    )

    data_module.connect(
        world_size=fabric.world_size,
        global_rank=fabric.global_rank,
        tokenizer=model.tokenizer,
        global_batch_size=config.data.global_batch_size,
        max_seq_length=config.data.max_seq_length,
        num_train_examples=config.data.get("num_train_examples", 50000),
        num_pos_per_query=config.data.get("num_pos_per_query", 1),
        num_neg_per_query=config.data.get("num_neg_per_query", 1),
        num_epochs=config.training.get("epochs", 3),
        instruction=config.data.get("instruction"),
        negative_curriculum=config.data.get("negative_curriculum"),
    )
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    train_loader = fabric.setup_dataloaders(train_loader)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.lr,
        weight_decay=config.training.get("weight_decay", 0.01),
    )

    epochs = config.training.get("epochs", 5)
    grad_accum = config.training.get("gradient_accumulation_steps", 1)
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = batches_per_epoch // grad_accum
    total_steps = batches_per_epoch * epochs
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    fabric.print(
        f"\n{'='*60}\n"
        f"Data & schedule stats\n"
        f"  batches/epoch:          {batches_per_epoch:,}\n"
        f"  grad_accum_steps:       {grad_accum}\n"
        f"  optimizer steps/epoch:  {optimizer_steps_per_epoch:,}\n"
        f"  epochs:                 {epochs}\n"
        f"  total optimizer steps:  {total_optimizer_steps:,}\n"
        f"  total_steps (sched):    {total_steps:,}  "
        f"[scheduler sees {total_optimizer_steps}/{total_steps} = "
        f"{total_optimizer_steps/total_steps:.1%} of schedule]\n"
        f"  warmup steps:           {config.training.get('warmup_steps', 100)}\n"
        f"  lr:                     {config.training.lr}\n"
        f"  min_lr:                 {config.training.lr * config.training.get('min_reduce_rate', 0.0):.2e}\n"
        f"  ranking_loss_weight:    {config.training.get('ranking_loss_weight', 0.0)}\n"
        f"  ranking_margin:         {config.training.get('ranking_margin', 0.0)}\n"
        f"  listwise_loss_weight:   {config.training.get('listwise_loss_weight', 0.0)}\n"
        f"  listwise_temperature:   {config.training.get('listwise_temperature', 1.0)}\n"
        f"  pos_weight:             {config.training.get('pos_weight', 1.0)}\n"
        f"  negative curriculum:    {config.data.get('negative_curriculum')}\n"
        f"{'='*60}\n"
    )

    scheduler = get_cosine_annealing_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config.training.get("warmup_steps", 100),
        num_training_steps=total_optimizer_steps,
        num_cycles=1,# config.training.get("epochs", 5),
        min_reduce_rate=config.training.get("min_reduce_rate", 0.0),
    )

    model, optimizer = fabric.setup(model, optimizer)

    # Optional warm-start: load model weights from a prior FSDP checkpoint
    # (saved via fabric.save). Optimizer / scheduler are NOT restored so the
    # LR schedule starts fresh (warmup -> peak -> cosine).
    resume_path = config.training.get("model_resume_from")
    if resume_path:
        resume_path = pathlib.Path(resume_path)
        if not resume_path.exists():
            raise FileNotFoundError(f"model_resume_from not found: {resume_path}")
        fabric.print(f"Warm-starting model weights from: {resume_path}")
        fabric.load(resume_path, {"model": model})
        fabric.print("Warm-start load complete.")

    for epoch in range(config.training.get("epochs", 5)):
        data_module.set_epoch(epoch)
        fit_epoch(fabric, model, optimizer, scheduler, train_loader, config, epoch)
        # Diagnostic: what fraction of queries got a genuine high-similarity
        # positive this epoch (vs. falling back to a hard one). Low => mining
        # top_k_pos is too small and positives stay all-hard.
        if fabric.global_rank == 0:
            hits = sum(ds._easy_pos_hits for ds in data_module._datasets)
            total = sum(ds._easy_pos_total for ds in data_module._datasets)
            frac = hits / total if total else 0.0
            fabric.print(f"[epoch {epoch}] easy-positive fraction: {frac:.3f} ({hits}/{total})")


def _choose_logger(config: DictConfig, output_dir: pathlib.Path):
    logger_cfg = config.training.get("logger", {})
    logger_type = logger_cfg.get("type", "csv") if logger_cfg else "csv"
    run_name = config.get("run_name", "reranker")

    if logger_type == "wandb":
        from lightning.pytorch.loggers import WandbLogger
        logger_dir = output_dir / "logs_wandb"
        logger_dir.mkdir(parents=True, exist_ok=True)
        return WandbLogger(
            name=run_name,
            project=logger_cfg.get("project", "authorship-reranker"),
            save_dir=str(logger_dir),
        )

    from lightning.fabric.loggers import CSVLogger
    return CSVLogger(root_dir=str(output_dir / "logs"), name=run_name)


def setup(config: DictConfig):
    output_dir = pathlib.Path(config.training.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, str(output_dir / "config.yaml"))

    decoder_cls = _resolve_decoder_layer_cls(config.model.model_name_or_path)
    strategy = "auto"
    if config.training.get("devices", 1) > 1 and decoder_cls:
        from lightning.fabric.strategies import FSDPStrategy
        strategy = FSDPStrategy(
            auto_wrap_policy=_get_wrapping_policy(decoder_cls),
            activation_checkpointing_policy={decoder_cls},
        )

    logger = _choose_logger(config, output_dir)

    fabric = L.Fabric(
        accelerator="auto",
        devices=config.training.get("devices", 1),
        # num_nodes=config.training.get("num_nodes", 1),
        precision=config.training.get("precision", "bf16-mixed"),
        strategy=strategy,
        loggers=[logger],
    )
    fabric.launch(main, config=config)


@hydra.main(version_base=None, config_path="../../configs/reranker", config_name="default")
def hydra_main(config: DictConfig):
    setup(config)


if __name__ == "__main__":
    hydra_main()
