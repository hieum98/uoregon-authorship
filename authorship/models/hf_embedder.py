"""HuggingFace-native embedding models for dense hard-pair mining.

Loads models like Qwen/Qwen3-Embedding-0.6B directly from HuggingFace without
the authorship training projection head. Uses last-token pooling and left
padding as recommended by the Qwen3-Embedding model card.
"""

from typing import List, Optional, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    """Pool the last non-padding token (handles left and right padding)."""
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


class HFEmbeddingModel(nn.Module):
    """Inference wrapper for HF embedding models (e.g. Qwen3-Embedding)."""

    def __init__(
        self,
        model_name_or_path: str,
        instruct: Optional[str] = None,
        attn_implementation: Optional[str] = None,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.instruct = instruct

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            padding_side="left",
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        model_kwargs = {"trust_remote_code": True}
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
            model_kwargs["torch_dtype"] = torch.bfloat16

        self.model = AutoModel.from_pretrained(model_name_or_path, **model_kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if attn_implementation != "flash_attention_2":
            self.model.to(dtype=torch.bfloat16)
        self.model.eval()

    def _format_texts(self, texts: List[str]) -> List[str]:
        if self.instruct is None:
            return texts
        prefix = f"Instruct: {self.instruct}\nQuery:"
        return [f"{prefix}{t}" for t in texts]

    @torch.no_grad()
    def encode(
        self,
        sentences: Union[List[str], str],
        max_length: int = 512,
    ) -> torch.Tensor:
        if isinstance(sentences, str):
            sentences = [sentences]

        inputs = self.tokenizer(
            self._format_texts(sentences),
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        use_autocast = self.device.type == "cuda"
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_autocast):
            outputs = self.model(**inputs)
            embeddings = last_token_pool(
                outputs.last_hidden_state, inputs["attention_mask"]
            )

        return embeddings.cpu().float()
