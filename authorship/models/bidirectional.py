"""Bidirectional attention variants of causal LMs for embedding.

Converts causal language models (Qwen2, Qwen3, Mistral, Gemma3) into
bidirectional encoders by replacing causal masks with full attention masks
that only respect padding tokens. Follows the T5Gemma approach.
"""

import copy
from typing import Callable, Optional

import torch
from torch import nn

from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils.generic import check_model_inputs
from transformers.utils import auto_docstring


def bidirectional_mask_function(attention_mask: Optional[torch.Tensor]) -> Callable:
    """Create a mask function allowing all non-padding tokens to attend to each other."""
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        if attention_mask is None:
            return torch.ones((), dtype=torch.bool)
        return attention_mask[batch_idx, kv_idx].to(torch.bool)
    return inner_mask


def sliding_window_bidirectional_mask_function(sliding_window: int) -> Callable:
    """Create a bidirectional sliding window mask function."""
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return (q_idx - sliding_window < kv_idx) & (kv_idx < q_idx + sliding_window)
    return inner_mask


def _bidirectional_window_overlay(sliding_window: int) -> Callable:
    """Bidirectional mask within sliding window for Gemma3."""
    def inner_mask(batch_idx: int, head_idx: int, q_idx: int, kv_idx: int) -> bool:
        return abs(q_idx - kv_idx) < sliding_window
    return inner_mask


# ---------------------------------------------------------------------------
# Qwen2 Bidirectional
# ---------------------------------------------------------------------------
try:
    from transformers.models.qwen2.modeling_qwen2 import (
        Qwen2PreTrainedModel,
        Qwen2DecoderLayer,
        Qwen2RMSNorm,
    )
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update

    class _Qwen2RotaryEmbedding(nn.Module):
        inv_freq: torch.Tensor

        def __init__(self, config, device=None):
            super().__init__()
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

        @torch.no_grad()
        @dynamic_rope_update
        def forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @auto_docstring
    class BidirectionalQwen2Model(Qwen2PreTrainedModel):
        """Qwen2 with full bidirectional attention."""

        def __init__(self, config: Qwen2Config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [Qwen2DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = _Qwen2RotaryEmbedding(config=config)
            self.gradient_checkpointing = False
            self.has_sliding_layers = "sliding_attention" in self.config.layer_types
            for layer in self.layers:
                layer.self_attn.is_causal = False
            self.post_init()

        @check_model_inputs()
        @auto_docstring
        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            inputs_embeds=None,
            use_cache=None,
            cache_position=None,
            **kwargs,
        ) -> BaseModelOutputWithPast:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            if not isinstance(bidirectional_mask_mapping := attention_mask, dict):
                mask_kwargs = {
                    "config": self.config, "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask, "cache_position": cache_position,
                    "past_key_values": past_key_values, "position_ids": position_ids,
                }
                bidirectional_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs, or_mask_function=bidirectional_mask_function(attention_mask)),
                }
                if self.has_sliding_layers:
                    bidirectional_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                        **mask_kwargs,
                        or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                        and_mask_function=bidirectional_mask_function(attention_mask),
                    )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=bidirectional_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    cache_position=cache_position,
                    position_embeddings=position_embeddings,
                    **kwargs,
                )
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)

    _HAS_QWEN2 = True
except ImportError:
    BidirectionalQwen2Model = None
    _HAS_QWEN2 = False


# ---------------------------------------------------------------------------
# Qwen3 Bidirectional
# ---------------------------------------------------------------------------
try:
    from transformers.models.qwen3.modeling_qwen3 import (
        Qwen3PreTrainedModel,
        Qwen3DecoderLayer,
        Qwen3RMSNorm,
    )
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    class _Qwen3RotaryEmbedding(nn.Module):
        inv_freq: torch.Tensor

        def __init__(self, config, device=None):
            super().__init__()
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

        @torch.no_grad()
        @dynamic_rope_update
        def forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @auto_docstring
    class BidirectionalQwen3Model(Qwen3PreTrainedModel):
        """Qwen3 with full bidirectional attention."""

        def __init__(self, config: Qwen3Config):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [Qwen3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = _Qwen3RotaryEmbedding(config=config)
            self.gradient_checkpointing = False
            self.has_sliding_layers = "sliding_attention" in self.config.layer_types
            for layer in self.layers:
                layer.self_attn.is_causal = False
            self.post_init()

        @check_model_inputs()
        @auto_docstring
        def forward(
            self, input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, use_cache=None,
            cache_position=None, **kwargs,
        ) -> BaseModelOutputWithPast:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            if not isinstance(bidirectional_mask_mapping := attention_mask, dict):
                mask_kwargs = {
                    "config": self.config, "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask, "cache_position": cache_position,
                    "past_key_values": past_key_values, "position_ids": position_ids,
                }
                bidirectional_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs, or_mask_function=bidirectional_mask_function(attention_mask)),
                }
                if self.has_sliding_layers:
                    bidirectional_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(
                        **mask_kwargs,
                        or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                        and_mask_function=bidirectional_mask_function(attention_mask),
                    )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=bidirectional_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs,
                )
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)

    _HAS_QWEN3 = True
except ImportError:
    BidirectionalQwen3Model = None
    _HAS_QWEN3 = False


# ---------------------------------------------------------------------------
# Mistral Bidirectional
# ---------------------------------------------------------------------------
try:
    from transformers.models.mistral.modeling_mistral import (
        MistralPreTrainedModel,
        MistralDecoderLayer,
        MistralRMSNorm,
    )
    from transformers.models.mistral.configuration_mistral import MistralConfig

    class _MistralRotaryEmbedding(nn.Module):
        inv_freq: torch.Tensor

        def __init__(self, config, device=None):
            super().__init__()
            if hasattr(config, "rope_scaling") and isinstance(config.rope_scaling, dict):
                self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
            else:
                self.rope_type = "default"
            self.max_seq_len_cached = config.max_position_embeddings
            self.original_max_seq_len = config.max_position_embeddings
            self.config = config
            self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.original_inv_freq = self.inv_freq

        @torch.no_grad()
        @dynamic_rope_update
        def forward(self, x, position_ids):
            inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
            position_ids_expanded = position_ids[:, None, :].float()
            device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
            with torch.autocast(device_type=device_type, enabled=False):
                freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos() * self.attention_scaling
                sin = emb.sin() * self.attention_scaling
            return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @auto_docstring
    class BidirectionalMistralModel(MistralPreTrainedModel):
        """Mistral with full bidirectional attention."""

        def __init__(self, config: MistralConfig):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.layers = nn.ModuleList(
                [MistralDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = _MistralRotaryEmbedding(config=config)
            self.gradient_checkpointing = False
            for layer in self.layers:
                layer.self_attn.is_causal = False
            self.post_init()

        @check_model_inputs()
        @auto_docstring
        def forward(
            self, input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, use_cache=None,
            cache_position=None, **kwargs,
        ) -> BaseModelOutputWithPast:
            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None:
                past_key_values = DynamicCache(config=self.config)
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            if not isinstance(bidirectional_mask := attention_mask, dict):
                mask_kwargs = {
                    "config": self.config, "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask, "cache_position": cache_position,
                    "past_key_values": past_key_values, "position_ids": position_ids,
                }
                if self.config.sliding_window is None:
                    bidirectional_mask = create_causal_mask(**mask_kwargs, or_mask_function=bidirectional_mask_function(attention_mask))
                else:
                    bidirectional_mask = create_sliding_window_causal_mask(
                        **mask_kwargs,
                        or_mask_function=sliding_window_bidirectional_mask_function(self.config.sliding_window),
                        and_mask_function=bidirectional_mask_function(attention_mask),
                    )

            hidden_states = inputs_embeds
            position_embeddings = self.rotary_emb(hidden_states, position_ids)
            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                hidden_states = decoder_layer(
                    hidden_states, attention_mask=bidirectional_mask,
                    position_ids=position_ids, past_key_values=past_key_values,
                    use_cache=use_cache, cache_position=cache_position,
                    position_embeddings=position_embeddings, **kwargs,
                )
            hidden_states = self.norm(hidden_states)
            return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=past_key_values if use_cache else None)

    _HAS_MISTRAL = True
except ImportError:
    BidirectionalMistralModel = None
    _HAS_MISTRAL = False


# ---------------------------------------------------------------------------
# Gemma3 Bidirectional
# ---------------------------------------------------------------------------
try:
    from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3PreTrainedModel,
        Gemma3DecoderLayer,
        Gemma3RMSNorm,
        Gemma3RotaryEmbedding,
        Gemma3TextScaledWordEmbedding,
    )

    @auto_docstring
    class BidirectionalGemma3TextModel(Gemma3PreTrainedModel):
        """Gemma3 text model with full bidirectional attention."""
        config: Gemma3TextConfig

        def __init__(self, config: Gemma3TextConfig):
            super().__init__(config)
            self.padding_idx = config.pad_token_id
            self.vocab_size = config.vocab_size
            self.embed_tokens = Gemma3TextScaledWordEmbedding(
                config.vocab_size, config.hidden_size, self.padding_idx,
                embed_scale=self.config.hidden_size ** 0.5,
            )
            self.layers = nn.ModuleList(
                [Gemma3DecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
            )
            self.norm = Gemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.rotary_emb = Gemma3RotaryEmbedding(config=config)
            self.gradient_checkpointing = False

            local_config = copy.deepcopy(config)
            local_config.rope_theta = config.rope_local_base_freq
            local_config.rope_scaling = {"rope_type": "default"}
            self.rotary_emb_local = Gemma3RotaryEmbedding(config=local_config)

            for layer in self.layers:
                layer.self_attn.is_causal = False
            self.post_init()

        @check_model_inputs()
        @auto_docstring
        def forward(
            self, input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, use_cache=None,
            output_attentions=None, output_hidden_states=None,
            cache_position=None, **kwargs,
        ) -> BaseModelOutputWithPast:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            use_cache = use_cache if use_cache is not None else self.config.use_cache

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids)
            if use_cache and past_key_values is None and not self.training:
                past_key_values = DynamicCache(config=self.config)
            if cache_position is None:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
                cache_position = torch.arange(past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device)
            if position_ids is None:
                position_ids = cache_position.unsqueeze(0)

            if not isinstance(bidirectional_mask_mapping := attention_mask, dict):
                mask_kwargs = {
                    "config": self.config, "input_embeds": inputs_embeds,
                    "attention_mask": attention_mask, "cache_position": cache_position,
                    "past_key_values": past_key_values, "position_ids": position_ids,
                }
                sliding_mask_kwargs = mask_kwargs.copy()
                mask_kwargs["or_mask_function"] = bidirectional_mask_function(attention_mask)
                sliding_mask_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)
                sliding_mask_kwargs["and_mask_function"] = bidirectional_mask_function(attention_mask)
                bidirectional_mask_mapping = {
                    "full_attention": create_causal_mask(**mask_kwargs),
                    "sliding_attention": create_sliding_window_causal_mask(**sliding_mask_kwargs),
                }

            hidden_states = inputs_embeds
            position_embeddings_global = self.rotary_emb(hidden_states, position_ids)
            position_embeddings_local = self.rotary_emb_local(hidden_states, position_ids)

            all_hidden_states = () if output_hidden_states else None
            all_self_attns = () if output_attentions else None

            for decoder_layer in self.layers[:self.config.num_hidden_layers]:
                if output_hidden_states:
                    all_hidden_states += (hidden_states,)
                layer_outputs = decoder_layer(
                    hidden_states,
                    position_embeddings_global=position_embeddings_global,
                    position_embeddings_local=position_embeddings_local,
                    attention_mask=bidirectional_mask_mapping[decoder_layer.attention_type],
                    position_ids=position_ids, past_key_values=past_key_values,
                    output_attentions=output_attentions, use_cache=use_cache,
                    cache_position=cache_position, **kwargs,
                )
                hidden_states = layer_outputs[0]
                if output_attentions:
                    all_self_attns += (layer_outputs[1],)

            hidden_states = self.norm(hidden_states)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            return BaseModelOutputWithPast(
                last_hidden_state=hidden_states,
                past_key_values=past_key_values,
                hidden_states=all_hidden_states,
                attentions=all_self_attns,
            )

    _HAS_GEMMA3 = True
except ImportError:
    BidirectionalGemma3TextModel = None
    _HAS_GEMMA3 = False


BIDIRECTIONAL_MODEL_MAPPING = {}
if _HAS_QWEN2:
    BIDIRECTIONAL_MODEL_MAPPING["qwen2"] = BidirectionalQwen2Model
if _HAS_QWEN3:
    BIDIRECTIONAL_MODEL_MAPPING["qwen3"] = BidirectionalQwen3Model
if _HAS_MISTRAL:
    BIDIRECTIONAL_MODEL_MAPPING["mistral"] = BidirectionalMistralModel
if _HAS_GEMMA3:
    BIDIRECTIONAL_MODEL_MAPPING["gemma3_text"] = BidirectionalGemma3TextModel
