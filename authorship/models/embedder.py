"""Embedding model for authorship attribution retrieval.

Wraps a transformer backbone (optionally bidirectional) with pooling and
a projection head for contrastive learning. Supports LoRA and 4-bit quantization.
"""

import json
import os
from pathlib import Path
from functools import partial
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
    BatchEncoding,
    Conv1D,
)
from peft import LoraConfig, TaskType, get_peft_model

from authorship.models.bidirectional import BIDIRECTIONAL_MODEL_MAPPING


def find_all_linear_names(model: nn.Module, quantization: bool = False) -> List[str]:
    """Collect names of all linear layers suitable for LoRA, excluding head layers."""
    if quantization:
        from bitsandbytes.nn import Linear4bit
        cls = (Linear4bit, Conv1D)
    else:
        cls = (torch.nn.Linear, Conv1D)

    names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names.add(name.rsplit(".", 1)[-1])

    names.discard("lm_head")
    output_emb = model.get_output_embeddings() if isinstance(model, PreTrainedModel) else None
    if output_emb is not None:
        last_name = [n for n, m in model.named_modules() if m is output_emb]
        if last_name:
            names.discard(last_name[0])

    return list(names)


class EmbeddingModel(nn.Module):
    """Transformer encoder with pooling and projection for contrastive learning.

    Supports bidirectional attention on causal LMs (Qwen2/3, Mistral, Gemma3)
    and standard encoder models (BERT, RoBERTa, etc.).
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_lora: bool = False,
        dropout_prob: float = 0.1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Union[str, List[str]] = "all",
        adapter_name: Optional[str] = None,
        quantization: bool = False,
        attn_implementation: Optional[str] = None,
        pooling_method: str = "mean",
        is_bidirectional: bool = False,
    ):
        super().__init__()
        self.hprams = {
            "model_name_or_path": model_name_or_path,
            "use_lora": use_lora,
            "dropout_prob": dropout_prob,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "adapter_name": adapter_name,
            "quantization": quantization,
            "attn_implementation": attn_implementation,
            "pooling_method": pooling_method,
            "is_bidirectional": is_bidirectional,
        }
        self.pooling_method = pooling_method
        self.is_bidirectional = is_bidirectional

        self.transformer = self._build_transformer(
            model_name_or_path, use_lora, lora_r, lora_alpha, lora_dropout,
            target_modules, adapter_name, quantization, attn_implementation,
            is_bidirectional,
        )
        self.tokenizer = self._build_tokenizer(model_name_or_path)
        self.dropout = nn.Dropout(dropout_prob)
        hidden_size = self.transformer.config.hidden_size
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.emb_dim = hidden_size

    @staticmethod
    def _build_tokenizer(model_name_or_path: str) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="right", trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _build_transformer(
        self, model_name_or_path, use_lora, lora_r, lora_alpha, lora_dropout,
        target_modules, adapter_name, quantization, attn_implementation,
        is_bidirectional,
    ) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_cache=False,
            **({"pretraining_tp": 1} if use_lora else {}),
        )
        bnb_config = None
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            )

        if is_bidirectional:
            model_type = config.model_type
            if model_type not in BIDIRECTIONAL_MODEL_MAPPING:
                raise ValueError(
                    f"Bidirectional not supported for {model_type}. "
                    f"Supported: {list(BIDIRECTIONAL_MODEL_MAPPING)}"
                )
            model_class = BIDIRECTIONAL_MODEL_MAPPING[model_type]
            print(f"Using bidirectional model: {model_class}")
        else:
            model_class = AutoModel

        transformer = model_class.from_pretrained(
            model_name_or_path, config=config, quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
        )

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(transformer, quantization)
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type=TaskType.FEATURE_EXTRACTION,
                target_modules=target_modules,
            )
            transformer = get_peft_model(
                transformer, lora_config, adapter_name=adapter_name or "default",
            )
        return transformer

    def pooling(
        self,
        hidden_state: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        prompt_length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones(
                hidden_state.size(0), hidden_state.size(1), device=hidden_state.device,
            )
        if prompt_length is not None:
            attention_mask = attention_mask.clone()
            for i, length in enumerate(prompt_length):
                attention_mask[i, :length] = 0

        hidden_state = hidden_state.to(attention_mask.device)

        if self.pooling_method == "cls":
            return hidden_state[:, 0].contiguous().to(hidden_state.dtype)

        if self.pooling_method == "lasttoken":
            b, n, d = hidden_state.size()
            reversed_mask = torch.flip(attention_mask, dims=(1,))
            argmax_reverse = torch.argmax(reversed_mask, dim=1, keepdim=False)
            gather_indices = torch.clamp(attention_mask.size(1) - argmax_reverse - 1, min=0)
            gather_indices = gather_indices.unsqueeze(-1).repeat(1, d).unsqueeze(1)
            input_mask_expanded = attention_mask.unsqueeze(-1).expand((b, n, d)).float()
            embedding = torch.gather(hidden_state * input_mask_expanded, 1, gather_indices).squeeze(1)
            return embedding.contiguous().to(hidden_state.dtype)

        if self.pooling_method in ("mean", "weightedmean"):
            if self.pooling_method == "weightedmean":
                attention_mask = attention_mask * attention_mask.cumsum(dim=1)
            s = torch.sum(hidden_state * attention_mask.unsqueeze(-1).float(), dim=1)
            d = attention_mask.sum(dim=1, keepdim=True).float()
            return (s / d).contiguous().to(hidden_state.dtype)

        raise NotImplementedError(f"Unknown pooling method: {self.pooling_method}")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> dict:
        outputs = self.transformer(
            input_ids=input_ids, attention_mask=attention_mask,
            return_dict=True, output_hidden_states=True,
        )
        hidden_states = outputs["last_hidden_state"]
        pooled = self.pooling(hidden_states, attention_mask=attention_mask)
        projection = self.projection(self.dropout(pooled))
        return {
            "reps": projection,
            "projection": projection,
            "hidden_states": hidden_states,
        }

    def tokenize_example(self, example: str, max_length: int = 512) -> BatchEncoding:
        return self.tokenizer(
            text=example, max_length=max_length, truncation=True,
            padding=False, return_tensors=None,
        )

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        batch_size: int = 256,
        max_length: int = 512,
    ) -> np.ndarray:
        input_was_string = isinstance(sentences, str)
        if input_was_string:
            sentences = [sentences]

        all_embeddings = []
        for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding", disable=len(sentences) < 256):
            batch = sentences[start : start + batch_size]
            inputs = [self.tokenize_example(s, max_length) for s in batch]
            inputs = self.tokenizer.pad(inputs, pad_to_multiple_of=8, return_tensors="pt")
            device = next(self.transformer.parameters()).device
            emb = self(
                input_ids=inputs["input_ids"].to(device),
                attention_mask=inputs["attention_mask"].to(device),
            )["reps"]
            all_embeddings.append(emb.cpu().float().numpy())

        result = np.concatenate(all_embeddings, axis=0)
        return result[0] if input_was_string else result


class WrappedEmbeddingModel(nn.Module):
    """Inference wrapper with checkpoint loading."""

    def __init__(
        self,
        model_name_or_path: str,
        use_lora: bool = False,
        dropout_prob: float = 0.1,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Union[str, List[str]] = "all",
        adapter_name: Optional[str] = None,
        quantization: bool = False,
        attn_implementation: Optional[str] = None,
        pooling_method: str = "mean",
        is_bidirectional: bool = False,
        model_checkpoint: Optional[str] = None,
    ):
        super().__init__()
        self.model = EmbeddingModel(
            model_name_or_path=model_name_or_path, use_lora=use_lora,
            dropout_prob=dropout_prob, lora_r=lora_r, lora_alpha=lora_alpha,
            lora_dropout=lora_dropout, target_modules=target_modules,
            adapter_name=adapter_name, quantization=quantization,
            attn_implementation=attn_implementation,
            pooling_method=pooling_method, is_bidirectional=is_bidirectional,
        )
        self.tokenizer = self.model.tokenizer

        if model_checkpoint and os.path.exists(model_checkpoint):
            self._load_checkpoint(model_checkpoint)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.to(dtype=torch.bfloat16)
        self.model.eval()

    def _load_checkpoint(self, path: str):
        path = str(path)

        def _load_sharded_safetensors(index_path: str) -> dict:
            from safetensors.torch import load_file
            index_obj = json.loads(Path(index_path).read_text(encoding="utf-8"))
            weight_map = index_obj.get("weight_map", {})
            if not weight_map:
                raise ValueError(f"Invalid safetensors index without weight_map: {index_path}")

            state = {}
            base_dir = Path(index_path).parent
            for shard_name in sorted(set(weight_map.values())):
                shard_path = base_dir / shard_name
                shard_state = load_file(str(shard_path), device="cpu")
                state.update(shard_state)
            return state

        path_obj = Path(path)
        if path.endswith(".ckpt"):
            state = torch.load(path, map_location="cpu")
            info = self.model.load_state_dict(state["model"], strict=False)
        elif path.endswith(".pt"):
            state = torch.load(path, map_location="cpu")
            info = self.model.load_state_dict(state, strict=False)
        elif path.endswith(".safetensors"):
            from safetensors.torch import load_file
            state = load_file(path, device="cpu")
            info = self.model.load_state_dict(state, strict=False)
        elif path.endswith(".safetensors.index.json"):
            state = _load_sharded_safetensors(path)
            info = self.model.load_state_dict(state, strict=False)
        elif path_obj.is_dir() and (path_obj / "model.safetensors.index.json").exists():
            state = _load_sharded_safetensors(str(path_obj / "model.safetensors.index.json"))
            info = self.model.load_state_dict(state, strict=False)
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        print(f"Loaded checkpoint {path} (missing: {info.missing_keys[:5]}...)")

    @torch.no_grad()
    def encode(self, sentences: Union[List[str], str], max_length: int = 512) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]
        inputs = self.tokenizer(
            text=sentences, max_length=max_length, truncation=True,
            padding="longest", return_tensors="pt", add_special_tokens=False,
        )
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reps = self.model(
                inputs["input_ids"].to(self.device),
                inputs["attention_mask"].to(self.device),
            )["reps"].cpu().float()
        return reps

    @torch.no_grad()
    def batch_encode(self, all_sentences: List[str], max_length: int = 512, batch_size: int = 4) -> torch.Tensor:
        import datasets as ds

        dataset = ds.Dataset.from_dict({"text": all_sentences})
        collator = partial(_embedding_collator_fn, tokenizer=self.tokenizer, max_length=max_length)
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, drop_last=False, num_workers=0,
            collate_fn=collator, pin_memory=False,
        )
        all_reps = []
        for batch in tqdm(loader, desc="Encoding", disable=len(all_sentences) < 256):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reps = self.model(
                    batch["input_ids"].to(self.device),
                    batch["attention_mask"].to(self.device),
                )["reps"].cpu()
            all_reps.append(reps)
        return torch.cat(all_reps, dim=0).float()


def _embedding_collator_fn(sentences, tokenizer, max_length):
    texts = [s["text"] for s in sentences]
    return tokenizer(
        text=texts, max_length=max_length, truncation=True,
        padding="longest", return_tensors="pt", add_special_tokens=False,
    )
