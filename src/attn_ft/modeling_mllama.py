from __future__ import annotations

import torch

from transformers.models.mllama import modeling_mllama as _upstream
from transformers.models.mllama.modeling_mllama import *  # noqa: F401,F403

MllamaTextCrossAttention = _upstream.MllamaTextCrossAttention
MllamaCrossAttentionDecoderLayer = _upstream.MllamaCrossAttentionDecoderLayer


def eager_attention_forward(
    module,
    query,
    key,
    value,
    attention_mask,
    scaling,
    dropout=0.0,
    return_raw_attn_weights=False,
    **kwargs,
):
    key_states = _upstream.repeat_kv(key, module.num_key_value_groups)
    value_states = _upstream.repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    if return_raw_attn_weights:
        attn_weights_raw = attn_weights

    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = torch.nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    if return_raw_attn_weights:
        return attn_output, attn_weights, attn_weights_raw
    return attn_output, attn_weights


def _patched_cross_attention_forward(
    self,
    hidden_states,
    cross_attention_states=None,
    past_key_values=None,
    attention_mask=None,
    use_cache=None,
    cache_position=None,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()
    query_states = self.q_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    query_states = self.q_norm(query_states)

    if cross_attention_states is not None:
        key_states = self.k_proj(cross_attention_states)
        value_states = self.v_proj(cross_attention_states)
        key_states = key_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        key_states = self.k_norm(key_states)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )
    elif cache_position[0] != 0:
        key_states, value_states = (
            past_key_values.layers[self.layer_idx].keys,
            past_key_values.layers[self.layer_idx].values,
        )
    else:
        raise ValueError(
            "Cross attention layer can't find neither `cross_attn_states` nor cached values for key/values!"
        )

    attn_impl = getattr(self.config, "_attn_implementation", None) or "eager"
    if attn_impl != "eager":
        attention_interface = _upstream.ALL_ATTENTION_FUNCTIONS[attn_impl]
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            **kwargs,
        )
        attn_weights_raw = None
    else:
        attn_output, attn_weights, attn_weights_raw = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.dropout,
            scaling=self.scaling,
            return_raw_attn_weights=True,
            **kwargs,
        )

    attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()
    attn_output = self.o_proj(attn_output)

    return attn_output, attn_weights, attn_weights_raw


def _patched_cross_attention_decoder_forward(
    self,
    hidden_states,
    cross_attention_states,
    cross_attention_mask,
    attention_mask,
    full_text_row_masked_out_mask,
    position_ids=None,
    past_key_values=None,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    residual = hidden_states
    hidden_states = self.input_layernorm(hidden_states)

    hidden_states, _, _ = self.cross_attn(
        hidden_states=hidden_states,
        attention_mask=cross_attention_mask,
        cross_attention_states=cross_attention_states,
        past_key_values=past_key_values,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = residual + self.cross_attn_attn_gate.tanh() * hidden_states

    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    if full_text_row_masked_out_mask is not None:
        hidden_states = full_text_row_masked_out_mask[:, 0] * hidden_states
    hidden_states = residual + self.cross_attn_mlp_gate.tanh() * hidden_states

    return hidden_states


MllamaTextCrossAttention.forward = _patched_cross_attention_forward
MllamaCrossAttentionDecoderLayer.forward = _patched_cross_attention_decoder_forward

__all__ = getattr(_upstream, "__all__", [name for name in globals() if not name.startswith("_")])