import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from .layers import SpikingSwiGLU, GroupedQueryAttention, RoPE, SpikingGroupedAttention, PreRMSNorm, SwiGLU


class RegularTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None, dropout=0.0,
                 max_seq_len=2048, rope_theta=1e4,
                 window_size=128, dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.norm1 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.norm2 = PreRMSNorm(d_model, dtype=dtype, device=device)

        head_dim = d_model // n_heads
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, head_dim
        )

        hidden_dim = intermediate_size or d_model * 4
        self.rope = RoPE(head_dim, max_seq_len, theta=rope_theta, dtype=dtype, device=device)
        self.ffn = SwiGLU(d_model, hidden_dim, dtype=dtype, device=device)

    def forward(self, x, use_cache=False, past_key_value=None):
        # Attention
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(
            x_norm, 
            mask=None,
            use_cache=use_cache, 
            past_key_value=past_key_value, 
            rope=self.rope
        )
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out

        return x, present_kv

class RegularLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, num_layers,
                 max_seq_len=2048, rope_theta=1e4,
                 window_size=128, dropout=0.0, intermediate_size=None,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.token_emb = nn.Embedding(vocab_size, d_model, dtype=dtype, device=device)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            RegularTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                intermediate_size=intermediate_size,
                dropout=dropout,
                max_seq_len=max_seq_len,
                rope_theta=rope_theta,
                dtype=dtype, 
                device=device
            )
            for _ in range(num_layers)
        ])

        self.final_norm = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, dtype=dtype, device=device)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        
    def get_num_params(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

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
            x, present_kv = layer(x, use_cache=use_cache, past_key_value=past_kv)
            presents.append(present_kv if use_cache else None)

        x = self.final_norm(x)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, presents if use_cache else None

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=100, temperature=1.0, top_k=None, top_p=None, 
                 do_sample=True, pad_token_id=None, eos_token_id=None):
        """
        Generate text with the model
        
        Args:
            input_ids: [batch, seq_len] starting tokens
            max_new_tokens: maximum number of new tokens to generate
            temperature: sampling temperature (0.0 = greedy)
            top_k: top-k sampling
            top_p: nucleus sampling
            do_sample: whether to sample or use greedy decoding
            pad_token_id: padding token id
            eos_token_id: end of sequence token id
        
        Returns:
            generated_ids: [batch, seq_len + generated_length]
        """
        self.eval()
        device = input_ids.device
        batch_size = input_ids.shape[0]
        
        # Initialize with input tokens
        generated = input_ids.clone()
        past_key_values = None
        
        for _ in range(max_new_tokens):
            # Prepare input for this step
            if past_key_values is None:
                # First step: use all tokens
                model_input = generated
            else:
                # Subsequent steps: only use last token
                model_input = generated[:, -1:]
            
            # Forward pass
            with torch.no_grad():
                outputs = self(model_input, use_cache=True, past_key_values=past_key_values)
                logits, past_key_values = outputs
                
                # Get logits for next token prediction
                next_token_logits = logits[:, -1, :]  # [batch, vocab_size]
                
                # Apply temperature
                if temperature != 1.0 and temperature > 0:
                    next_token_logits = next_token_logits / temperature
                
                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    top_k = min(top_k, next_token_logits.size(-1))
                    # Remove tokens with probability less than top-k
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None and top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample and temperature > 0:
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    # Greedy decoding
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=-1)
                
                # Check for end of sequence
                if eos_token_id is not None and (next_tokens == eos_token_id).all():
                    break
                
                # Check for maximum sequence length
                if generated.shape[1] >= self.max_seq_len:
                    break
        
        return generated

class SpikingTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, intermediate_size=None, dropout=0.0,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, beta=0.95, num_steps=5,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()

        self.norm1 = PreRMSNorm(d_model, dtype=dtype, device=device)
        self.norm2 = PreRMSNorm(d_model, dtype=dtype, device=device)

        self.attn = SpikingGroupedAttention(
            d_model, n_heads, n_kv_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
            rope_theta_local=rope_theta_local,
            rope_theta_global=rope_theta_global,
            num_steps=num_steps,
            dtype=dtype, device=device
        )


        hidden_dim = intermediate_size or d_model * 4
        self.ffn = SpikingSwiGLU(input_dim=d_model, hidden_dim=hidden_dim, beta=beta, num_steps=num_steps, dtype=dtype, device=device)

        self.attn_gate = nn.Parameter(torch.ones(1, dtype=dtype, device=device))
        self.ffn_gate = nn.Parameter(torch.ones(1, dtype=dtype, device=device))

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        # Attention
        x_norm = self.norm1(x)
        attn_out, present_kv = self.attn(x_norm, use_cache=use_cache, past_key_value=past_key_value, layer_idx=layer_idx)
        x = x + self.attn_gate * attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.ffn_gate * ffn_out

        return x, present_kv

class SpikingLLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_kv_heads, num_layers,
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6,
                 window_size=128, dropout=0.0, beta=0.95, num_steps=5, intermediate_size=None,
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
                beta=beta, num_steps=num_steps,
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
        x = x.to(self.lm_head.weight.dtype)
        logits = self.lm_head(x)  # [batch, seq_len, vocab_size]

        return logits, presents
