"""Reranker model for authorship attribution.

Wraps a causal LM and classifies whether a (query, candidate) document pair
is written by the same author, using P(yes) from yes/no token logits.
"""

import json
import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import LoraConfig, TaskType, get_peft_model

from authorship.models.embedder import find_all_linear_names
from authorship.data.utils import tokenize_pair


class RerankerModel(nn.Module):
    """Causal LM reranker scoring (query, candidate) pairs via yes/no logits."""

    def __init__(
        self,
        model_name_or_path: str,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Union[str, List[str]] = "all",
        adapter_name: Optional[str] = None,
        quantization: bool = False,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.hprams = {
            "model_name_or_path": model_name_or_path,
            "use_lora": use_lora,
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": target_modules,
            "adapter_name": adapter_name,
            "quantization": quantization,
            "attn_implementation": attn_implementation,
        }

        self.tokenizer = self._build_tokenizer(model_name_or_path)
        self.model = self._build_model(
            model_name_or_path, use_lora, lora_r, lora_alpha, lora_dropout,
            target_modules, adapter_name, quantization, attn_implementation,
        )

        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        assert self.token_true_id != self.tokenizer.unk_token_id, "'yes' token not in vocabulary"
        assert self.token_false_id != self.tokenizer.unk_token_id, "'no' token not in vocabulary"

    @staticmethod
    def _build_tokenizer(model_name_or_path: str) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left", trust_remote_code=True,
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        return tokenizer

    def _build_model(
        self, model_name_or_path, use_lora, lora_r, lora_alpha, lora_dropout,
        target_modules, adapter_name, quantization, attn_implementation,
    ) -> PreTrainedModel:
        config = AutoConfig.from_pretrained(
            model_name_or_path, trust_remote_code=True, use_cache=False,
        )
        bnb_config = None
        if quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            )

        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, config=config, quantization_config=bnb_config,
            attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16 if attn_implementation == "flash_attention_2" else None,
        )

        if use_lora:
            if target_modules == "all":
                target_modules = find_all_linear_names(model, quantization)
            lora_config = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                bias="none", task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
            )
            model = get_peft_model(model, lora_config, adapter_name=adapter_name or "default")
        return model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        last_logits = outputs.logits[:, -1, :]
        return {
            "logits": last_logits,
            "true_logits": last_logits[:, self.token_true_id],
            "false_logits": last_logits[:, self.token_false_id],
        }

    @torch.no_grad()
    def compute_score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Return P(yes) scores in [0, 1]."""
        out = self.forward(input_ids, attention_mask)
        stacked = torch.stack([out["false_logits"], out["true_logits"]], dim=1)
        return torch.nn.functional.softmax(stacked, dim=1)[:, 1]


class WrappedRerankerModel(nn.Module):
    """Inference wrapper with checkpoint loading and batch scoring.
    """

    def __init__(
        self,
        model_name_or_path: str,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: Union[str, List[str]] = "all",
        adapter_name: Optional[str] = None,
        quantization: bool = False,
        attn_implementation: Optional[str] = None,
        model_checkpoint: Optional[str] = None,
        max_length: int = 1024,
        batch_size: int = 8,
        instruction: Optional[str] = None,
    ):
        super().__init__()
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruction = instruction

        self.model = RerankerModel(
            model_name_or_path=model_name_or_path, use_lora=use_lora,
            lora_r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
            target_modules=target_modules, adapter_name=adapter_name,
            quantization=quantization, attn_implementation=attn_implementation,
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
    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = self.model.compute_score(
                input_ids.to(self.device), attention_mask.to(self.device),
            )
        return scores.cpu().float()

    @torch.no_grad()
    def score_pairs(self, query_texts: List[str], doc_texts: List[str]) -> np.ndarray:
        """Score parallel lists of (query, doc) text pairs. Returns (n,) array."""
        assert len(query_texts) == len(doc_texts)
        all_scores = []
        for start in range(0, len(query_texts), self.batch_size):
            end = min(start + self.batch_size, len(query_texts))
            tokenized = [
                tokenize_pair(q, d, self.tokenizer, self.max_length, self.instruction)
                for q, d in zip(query_texts[start:end], doc_texts[start:end])
            ]
            padded = self.tokenizer.pad(tokenized, padding="max_length", max_length=self.max_length, return_tensors="pt")
            scores = self.score(padded["input_ids"], padded["attention_mask"])
            all_scores.append(scores.numpy())
        return np.concatenate(all_scores, axis=0)

    @torch.no_grad()
    def score_matrix(
        self,
        query_texts: List[str],
        candidate_texts: List[str],
        candidate_indices: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Build (n_q, n_c) score matrix. If candidate_indices given, only score those pairs."""
        n_q, n_c = len(query_texts), len(candidate_texts)
        score_mat = np.zeros((n_q, n_c), dtype=np.float32)

        all_q, all_d, all_pos = [], [], []
        if candidate_indices is not None:
            for i in range(n_q):
                for j_local in range(candidate_indices.shape[1]):
                    j = int(candidate_indices[i, j_local])
                    if 0 <= j < n_c:
                        all_q.append(query_texts[i])
                        all_d.append(candidate_texts[j])
                        all_pos.append((i, j))
        else:
            for i in range(n_q):
                for j in range(n_c):
                    all_q.append(query_texts[i])
                    all_d.append(candidate_texts[j])
                    all_pos.append((i, j))

        if all_q:
            scores = self.score_pairs(all_q, all_d)
            for idx, (i, j) in enumerate(all_pos):
                score_mat[i, j] = scores[idx]
        return score_mat
