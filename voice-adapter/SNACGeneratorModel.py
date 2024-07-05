import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

SNAC_VOCAB_SIZE = 4096
EMBED_DIM = 4096  # LLM's embedding dimensions

_HABANA_AVAILABLE = os.getenv("HABANA", "0") == "1"


class SNACEmbeddingDataset(Dataset):
    def __init__(self, file_path):
        self.data = torch.load(file_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class TELayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hidden_size, eps=eps)

    def forward(self, hidden_states):
        return self.layer_norm(hidden_states)


class TELinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        if _HABANA_AVAILABLE:
            import habana_frameworks.torch.hpex.experimental.transformer_engine as te  # type: ignore

            self.linear = te.Linear(in_features, out_features, bias=bias)
        else:
            self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight = self.linear.weight

    def forward(self, x):
        return self.linear(x)


def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # print(f"apply_rotary_pos_emb input shapes: q={q.shape}, k={k.shape}, cos={cos.shape}, sin={sin.shape}, position_ids={position_ids.shape}")
    # Select the appropriate positional encodings
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # print(f"apply_rotary_pos_emb output shapes: q_embed={q_embed.shape}, k_embed={k_embed.shape}")
    return q_embed, k_embed


class InfiniAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.query = TELinear(config.hidden_size, config.hidden_size, bias=False)
        self.key = TELinear(config.hidden_size, config.hidden_size, bias=False)
        self.value = TELinear(config.hidden_size, config.hidden_size, bias=False)
        self.o_proj = TELinear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.attention_dropout
        self.gate = nn.Parameter(torch.full((1, self.num_heads, 1, 1), 0.0))
        self.memory = None
        self.norm_term = None

        self.rotary_dim = self.hidden_size
        self.register_buffer(
            "rope_inv_freq",
            1.0
            / (
                10000 ** (torch.arange(0, self.rotary_dim, 2).float() / self.rotary_dim)
            ),
            persistent=False,
        )
        # print(f"InfiniAttention initialized with num_heads={self.num_heads}, head_dim={self.head_dim}, hidden_size={self.hidden_size}")

    def _update_memory(self, key_states, value_states):
        key_states = F.elu(key_states) + 1

        if self.memory is None:
            self.memory = torch.matmul(key_states.transpose(-2, -1), value_states)
            self.norm_term = key_states.sum(dim=2, keepdim=True)
        else:
            self.memory = self.memory + torch.matmul(
                key_states.transpose(-2, -1), value_states
            )
            self.norm_term = self.norm_term + key_states.sum(dim=2, keepdim=True)

    def _retrieve_from_memory(self, query_states):
        if self.memory is None or self.norm_term is None:
            return torch.zeros_like(query_states)

        query_states = F.elu(query_states) + 1
        memory_output = torch.matmul(query_states, self.memory)
        norm_term_broadcastable = torch.matmul(
            query_states, self.norm_term.transpose(-2, -1)
        )
        memory_output = memory_output / norm_term_broadcastable

        return memory_output

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # print(f"InfiniAttention forward input shapes: hidden_states={hidden_states.shape}, attention_mask={attention_mask.shape if attention_mask is not None else None}, position_ids={position_ids.shape if position_ids is not None else None}")
        bsz, tgt_len, _ = hidden_states.shape

        # Linear projections
        query_states = self.query(hidden_states)
        key_states = self.key(hidden_states)
        value_states = self.value(hidden_states)
        # print(f"After linear projections: query_states={query_states.shape}, key_states={key_states.shape}, value_states={value_states.shape}")

        # Apply rotary embeddings
        seq_len = query_states.shape[1]
        t = torch.arange(seq_len, device=hidden_states.device).type_as(
            self.rope_inv_freq
        )
        freqs = torch.einsum("i,j->ij", t, self.rope_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).unsqueeze(0)
        cos, sin = emb.cos(), emb.sin()
        # print(f"Rotary embedding shapes: cos={cos.shape}, sin={sin.shape}")

        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        # Reshape for multi-head attention
        query_states = query_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, tgt_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # print(f"After reshaping: query_states={query_states.shape}, key_states={key_states.shape}, value_states={value_states.shape}")

        self._update_memory(key_states, value_states)
        memory_output = self._retrieve_from_memory(query_states)

        # print(f"Before scaled_dot_product_attention: query_states={query_states.shape}, key_states={key_states.shape}, value_states={value_states.shape}")
        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.dropout if self.training else 0.0,
        )
        # print(f"After scaled_dot_product_attention: attn_output={attn_output.shape}")

        combined_output = (
            F.sigmoid(self.gate) * memory_output
            + (1 - F.sigmoid(self.gate)) * attn_output
        )
        combined_output = (
            combined_output.transpose(1, 2)
            .contiguous()
            .view(bsz, tgt_len, self.hidden_size)
        )
        attn_output = self.o_proj(combined_output)

        # print(f"InfiniAttention forward output shape: {attn_output.shape}")
        return attn_output


class InfiniDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = InfiniAttention(config)
        self.self_attn_layer_norm = TELayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation_fn = F.mish
        self.fc1 = TELinear(config.hidden_size, config.intermediate_size)
        self.fc2 = TELinear(config.intermediate_size, config.hidden_size)
        self.final_layer_norm = TELayerNorm(config.hidden_size)

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # print(f"InfiniDecoderLayer forward input shapes: hidden_states={hidden_states.shape}, attention_mask={attention_mask.shape if attention_mask is not None else None}, position_ids={position_ids.shape if position_ids is not None else None}")
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, attention_mask=attention_mask, position_ids=position_ids
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = self.fc2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states

        # print(f"InfiniDecoderLayer forward output shape: {hidden_states.shape}")
        return hidden_states


class SNACGenerator(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList(
            [InfiniDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.snac_proj = TELinear(config.hidden_size, SNAC_VOCAB_SIZE)
        self.config = config

    def forward(self, hidden_states, attention_mask=None, position_ids=None):
        # print(f"SNACGenerator forward input shapes: hidden_states={hidden_states.shape}, attention_mask={attention_mask.shape if attention_mask is not None else None}, position_ids={position_ids.shape if position_ids is not None else None}")
        for i, layer in enumerate(self.layers):
            # print(f"Processing layer {i}")
            hidden_states = layer(
                hidden_states, attention_mask=attention_mask, position_ids=position_ids
            )

        snac_logits = self.snac_proj(hidden_states)
        # print(f"SNACGenerator forward output shape: {snac_logits.shape}")
        return snac_logits


class SNACConfig:
    def __init__(self):
        self.hidden_size = EMBED_DIM
        self.num_hidden_layers = 6
        self.num_attention_heads = 32
        self.intermediate_size = 2730
        self.hidden_dropout_prob = 0.1
        self.attention_dropout = 0.1
        self.activation_dropout = 0.1
        self.max_position_embeddings = 4096
