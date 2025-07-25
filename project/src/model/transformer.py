import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from .layers import SpikingSwiGLU, SpikingGroupedSlidingAttention, PreRMSNorm

class SpikingTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None, dropout=0.0,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, beta=0.95,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.norm1 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.norm2 = PreRMSNorm(d_model, dtype=dtype, device=device)

        self.attn = SpikingGroupedSlidingAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            window_size=window_size, beta=beta,
            dtype=dtype, device=device
        )

        hidden_dim = intermediate_size or d_model * 4
        self.ffn = SpikingSwiGLU(d_model, hidden_dim, beta=beta, dtype=dtype, device=device)

    def reset_mem(self):
        self.attn.reset_mem()

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        # Attention
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(x_norm, use_cache=use_cache, past_key_value=past_key_value, layer_idx=layer_idx)
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, present_kv

class SpikingLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, num_layers,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, dropout=0.0, beta=0.95, intermediate_size=None,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            SpikingTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_theta_local=rope_theta_local,
                rope_theta_global=rope_theta_global,
                window_size=window_size,
                beta=beta, 
                dtype=dtype, 
                device=device
            )
            for _ in range(num_layers)
        ])

        self.final_norm = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype, device=device)

    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

    def reset_mem(self):
        for layer in self.layers:
            layer.reset_mem()

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        """
        input_ids: [batch, seq_len]
        past_key_values: list of (k, v) tuples per layer if use_cache=True
        """
        x = self.token_emb(input_ids)  # [batch, seq_len, d_model]
        x = self.dropout(x)

        presents = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, use_cache=use_cache, past_key_value=past_kv, layer_idx=i)
            presents.append(present_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, presents

class SpikingMoEFFN(nn.Module):
    def __init__(self, d_model, hidden_dim, num_experts=8, num_active=2, beta=0.95, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.num_experts = num_experts
        self.num_active = num_active

        self.gate_linear = nn.Linear(d_model, num_experts, dtype=dtype, device=device)
        self.gate_spike = snn.Leaky(beta=beta)
        self.gate_mem = None

        self.experts = nn.ModuleList([
            SpikingSwiGLU(d_model, hidden_dim, beta=beta, dtype=dtype, device=device)
            for _ in range(num_experts)
        ])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [batch*seq_len, d_model]

        if self.gate_mem is None:
            self.gate_mem = self.gate_spike.reset_mem()

        gate_logits = self.gate_linear(x_flat)                   
        spike_out, self.gate_mem = self.gate_spike(gate_logits, self.gate_mem)
        
        topk_logits, topk_indices = torch.topk(spike_out, self.num_active, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)  # [batch*seq_len, num_active]

        output = torch.zeros_like(x_flat)

        for i in range(self.num_active):
            # [batch*seq_len]
            expert_idx = topk_indices[:, i]
            expert_weights = topk_weights[:, i]

            for expert_id in range(self.num_experts):
                mask = (expert_idx == expert_id)
                
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[mask] += expert_weights[mask].unsqueeze(-1) * expert_output
        return output.view(batch_size, seq_len, d_model)

    def reset_mem(self):
        self.gate_mem = None
        for expert in self.experts:
            expert.reset_mem()

class SpikingMoETransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None,
                 dropout=0.0, max_seq_len=2048, rope_theta_local=1e4,
                 rope_theta_global=1e6, window_size=128, beta=0.95,
                 use_moe=False, num_experts=8, num_active=2,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.norm1 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.norm2 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.use_moe = use_moe

        self.attn = SpikingGroupedSlidingAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            window_size=window_size,
            beta=beta,
            dtype=dtype, device=device
        )

        # FFN layer (MoE or regular) 
        hidden_dim = intermediate_size or d_model * 4

        if use_moe:
            self.ffn = SpikingMoEFFN(d_model, hidden_dim, num_experts, num_active, beta, dtype=dtype, device=device)
        else:
            self.ffn = SpikingSwiGLU(d_model, hidden_dim, beta, dtype=dtype, device=device)
        
    def forward(self, x, use_cache=False, past_key_values=None, layer_idx=0):
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(x_norm, use_cache=use_cache, past_key_value=past_key_values, layer_idx=layer_idx)

        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, present_kv
    
    def reset_mem(self):
        self.attn.reset_mem()
        self.ffn.reset_mem()

class SpikingMoELLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, num_layers,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, dropout=0.0, beta=0.95, intermediate_size=None,
                 tie_embeddings=True, embedding_dropout=0.0,
                 moe_layers=None, num_experts=8, num_active=2,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        if moe_layers is None:
            moe_layers = [i for i in range(num_layers) if (i+1) % 4 == 0]
        
        self.moe_layers = set(moe_layers)

        self.token_emb = nn.Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.dropout = nn.Dropout(embedding_dropout)

        self.layers = nn.ModuleList([
            SpikingMoETransformerBlock(
                d_model=d_model, n_heads=n_heads,
                n_kv_heads=n_kv_heads, intermediate_size=intermediate_size,
                dropout=dropout, max_seq_len=max_seq_len,
                rope_theta_local=rope_theta_local, rope_theta_global=rope_theta_global,
                window_size=window_size, beta=beta,
                use_moe=(i in self.moe_layers), 
                num_experts=num_experts, num_active=num_active,
                dtype=dtype, device=device
            )
            for i in range(num_layers)
        ])

        self.final_norm = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, dtype=dtype, device=device)

        if tie_embeddings:
            self.lm_head.weight = self.token_emb.weight

    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())
    
    def count_moe_params(self):
        """Count parameters in MoE layers vs regular layers"""
        moe_params = 0
        regular_params = 0
        
        for i, layer in enumerate(self.layers):
            layer_params = sum(p.numel() for p in layer.parameters())
            if i in self.moe_layers:
                moe_params += layer_params
            else:
                regular_params += layer_params
                
        return moe_params, regular_params

    def reset_mem(self):
        for layer in self.layers:
            layer.reset_mem()

    def forward(self, input_ids, use_cache=False, past_key_values=None):
        # embeddings
        x = self.token_emb(input_ids)
        x = self.dropout(x)

        # transformer layers
        presents = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            x, present_kv = layer(x, use_cache=use_cache, past_key_values=past_kv, layer_idx=i)
            presents.append(present_kv)
        
        # output
        x = self.final_norm(x)
        logits = self.lm_head(x)

        return logits, presents


def get_moe_config(size="1b", moe_ratio=0.25):
    """
    Get configuration for MoE model
    moe_ratio: fraction of layers that use MoE
    """
    base_configs = {
        "1b": {
            "vocab_size": 128256,
            "d_model": 2048,
            "n_heads": 32,
            "n_kv_heads": 8,
            "num_layers": 24,
            "intermediate_size": 5504,
            "max_seq_len": 8192,
        },
        "3b": {  # 3B total params, ~1B active
            "vocab_size": 128256,
            "d_model": 2560,
            "n_heads": 32,
            "n_kv_heads": 8,  
            "num_layers": 28,
            "intermediate_size": 6912,
            "max_seq_len": 16384,
        }
    }
    
    config = base_configs[size]
    
    # Calculate which layers should use MoE
    num_moe_layers = max(1, int(config["num_layers"] * moe_ratio))
    # Spread MoE layers evenly (like Qwen pattern)
    step = config["num_layers"] // num_moe_layers
    moe_layers = [i * step + step - 1 for i in range(num_moe_layers)]
    
    config.update({
        "rope_theta_local": 10_000,
        "rope_theta_global": 1_000_000,
        "window_size": min(1024, config["max_seq_len"] // 8),
        "dropout": 0.0,
        "beta": 0.95,
        "tie_embeddings": True,
        "embedding_dropout": 0.0,
        
        # MoE settings
        "moe_layers": moe_layers,
        "num_experts": 8,      # Total experts per MoE layer
        "num_active": 2,       # Active experts per token
    })
    
    return config