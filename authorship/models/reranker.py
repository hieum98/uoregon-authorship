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
from tqdm import tqdm
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

        def _load_model_state(raw_state: dict) -> None:
            nonlocal info
            model_sd = raw_state["model"] if isinstance(raw_state, dict) and "model" in raw_state else raw_state
            model_sd = self._normalize_state_dict_keys(model_sd)
            info = self.model.load_state_dict(model_sd, strict=False)

        path_obj = Path(path)
        info = None
        if path.endswith(".ckpt"):
            _load_model_state(torch.load(path, map_location="cpu", weights_only=False))
        elif path.endswith(".pt"):
            _load_model_state(torch.load(path, map_location="cpu", weights_only=False))
        elif path.endswith(".safetensors"):
            from safetensors.torch import load_file
            _load_model_state(load_file(path, device="cpu"))
        elif path.endswith(".safetensors.index.json"):
            _load_model_state(_load_sharded_safetensors(path))
        elif path_obj.is_dir() and (path_obj / "model.safetensors.index.json").exists():
            _load_model_state(_load_sharded_safetensors(str(path_obj / "model.safetensors.index.json")))
        else:
            raise ValueError(f"Unsupported checkpoint format: {path}")
        print(f"Loaded checkpoint {path} (missing: {info.missing_keys[:5]}...)")

    @staticmethod
    def _normalize_state_dict_keys(state_dict: dict) -> dict:
        """Strip Fabric's extra ``model.`` prefix from consolidated FSDP checkpoints."""
        if not state_dict:
            return state_dict
        keys = list(state_dict.keys())
        if keys and all(k.startswith("model.model.") for k in keys):
            return {k[len("model.") :]: v for k, v in state_dict.items()}
        return state_dict

    @torch.no_grad()
    def score(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            scores = self.model.compute_score(
                input_ids.to(self.device), attention_mask.to(self.device),
            )
        return scores.cpu().float()

    @torch.no_grad()
    def score_pairs(
        self,
        query_texts: List[str],
        doc_texts: List[str],
        *,
        desc: Optional[str] = "Reranking",
        show_progress: bool = True,
    ) -> np.ndarray:
        """Score parallel lists of (query, doc) text pairs. Returns (n,) array."""
        assert len(query_texts) == len(doc_texts)
        n_pairs = len(query_texts)
        if n_pairs == 0:
            return np.zeros(0, dtype=np.float32)

        # Tokenize all pairs up front, then sort by length so each batch groups
        # similarly-sized sequences. Combined with dynamic padding this minimizes
        # padding waste when the caller hands us a large, length-varied set of
        # pairs (e.g. all doc pairs for a query author at once). Each sequence's
        # P(yes) score is independent of its batch-mates, so reordering does not
        # change results; scores are scattered back into the input order below.
        tokenized_all = [
            tokenize_pair(q, d, self.tokenizer, self.max_length, self.instruction)
            for q, d in zip(query_texts, doc_texts)
        ]
        order = sorted(range(n_pairs), key=lambda idx: len(tokenized_all[idx]["input_ids"]))

        all_scores = np.empty(n_pairs, dtype=np.float32)
        batch_starts = range(0, n_pairs, self.batch_size)
        iterator = batch_starts
        if show_progress:
            iterator = tqdm(
                batch_starts,
                desc=desc,
                total=(n_pairs + self.batch_size - 1) // self.batch_size,
                unit="batch",
            )
        for start in iterator:
            end = min(start + self.batch_size, n_pairs)
            batch_idx = order[start:end]
            tokenized = [tokenized_all[k] for k in batch_idx]
            # Dynamic padding to the batch's longest sequence (capped at max_length
            # by tokenize_pair's truncation). With left padding + attention mask the
            # P(yes) logit is read from the last real token, so this is numerically
            # identical to padding="max_length" but avoids paying for 512 tokens on
            # short pairs. pad_to_multiple_of=8 keeps tensor-core/flash-attn alignment.
            padded = self.tokenizer.pad(
                tokenized, padding="longest", pad_to_multiple_of=8, return_tensors="pt",
            )
            scores = self.score(padded["input_ids"], padded["attention_mask"])
            all_scores[batch_idx] = scores.numpy()
        return all_scores

    @torch.no_grad()
    def score_matrix(
        self,
        query_texts: List[str],
        candidate_texts: List[str],
        candidate_indices: Optional[np.ndarray] = None,
        *,
        desc: Optional[str] = "Reranking",
        show_progress: bool = True,
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
            pair_desc = desc
            if desc and len(all_q) > 0:
                pair_desc = f"{desc} ({len(all_q)} pairs)"
            scores = self.score_pairs(
                all_q, all_d, desc=pair_desc, show_progress=show_progress,
            )
            for idx, (i, j) in enumerate(all_pos):
                score_mat[i, j] = scores[idx]
        return score_mat
