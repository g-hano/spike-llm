import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

def swish(x1):
    return x1 * torch.sigmoid(x1)

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim*2, dtype=dtype, device=device) # times(2) because will chunk into 2
        self.out = nn.Linear(hidden_dim, input_dim, dtype=dtype, device=device)
    
    def forward(self, x):
        x = x.to(self.fc.weight.dtype)
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        swh = swish(x1)
        gated = swh * x2
        return self.out(gated)

class SpikingSwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.95, num_steps=5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.swiglu = SwiGLU(input_dim, hidden_dim, dtype=dtype, device=device)
        self.lif = snn.Leaky(beta=beta, learn_beta=True)
        self.num_steps = num_steps
        
        self.spike_scale = nn.Parameter(torch.tensor(1.0, dtype=dtype, device=device))
        self.register_buffer('running_mean', torch.tensor(1.0, dtype=dtype, device=device))
        self.momentum = 0.9
    
    def forward(self, x):
        x = self.swiglu(x)

        mem = self.lif.init_leaky()
        mem = mem.to(x.device)
        spike_acc = torch.zeros_like(x)

        for _ in range(self.num_steps):
            spk, mem = self.lif(x, mem)
            spike_acc += spk
        
        spike_rate = spike_acc / self.num_steps
        with torch.no_grad():
            if self.training:
                current_mean = x.abs().mean()
                self.running_mean.mul_(self.momentum).add_(current_mean, alpha=1-self.momentum)
        
        # Use running mean for stable scaling
        output = spike_rate * self.running_mean * self.spike_scale
        return output

def rotate_half(x):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(q, k, cos, sin):
    # [seq_len, head_dim//2] -> [seq_len, head_dim]
    cos = torch.repeat_interleave(cos, 2, dim=-1)
    sin = torch.repeat_interleave(sin, 2, dim=-1)
    
    cos = cos[None, :, None, :]
    sin = sin[None, :, None, :]
    
    q_rotated = q * cos + rotate_half(q) * sin
    k_rotated = k * cos + rotate_half(k) * sin

    return q_rotated, k_rotated

class RoPE(nn.Module):
    def __init__(self, head_dim, max_seq_len, theta=10_000, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.theta = theta

        # frequency matrix
        # θ_i = 1 / (theta^(2i/head_dim)) for i = 0, 1, ..., head_dim//2 - 1
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=dtype, device=device).float() / head_dim))
        self.register_buffer('inv_freq', inv_freq)

        self._precompute_cossin(max_seq_len)
    
    def _precompute_cossin(self, seq_len):
        pos = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", pos, self.inv_freq)
        self.register_buffer("cos_cached", torch.cos(freqs), persistent=False)
        self.register_buffer("sin_cached", torch.sin(freqs), persistent=False)

    def forward(self, x, seq_len=None):
        seq_len = seq_len or x.shape[-2]
        if seq_len > self.max_seq_len:
            self._precompute_cossin(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

def get_local_attn_mask(seq_len, window_size, device="cuda", kv_len=None):
    kv_len = kv_len or seq_len

    q_pos = torch.arange(seq_len, device=device).unsqueeze(1)
    k_pos = torch.arange(kv_len, device=device).unsqueeze(0)

    causal_mask = q_pos >= k_pos
    window_mask = (q_pos - k_pos) <= window_size
    combined_mask = causal_mask & window_mask
    return combined_mask

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, bias=False, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim or d_model // n_heads

        assert n_heads % n_kv_heads == 0, f"{n_heads=} must be divisible by {n_kv_heads=}"
        self.n_groups = n_heads // n_kv_heads

        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, n_heads*self.head_dim, bias=bias, dtype=dtype, device=device)
        self.k_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=bias, dtype=dtype, device=device)
        self.v_proj = nn.Linear(d_model, n_kv_heads*self.head_dim, bias=bias, dtype=dtype, device=device)
        self.o_proj = nn.Linear(n_heads*self.head_dim, d_model, bias=bias, dtype=dtype, device=device)
        
        self.dropout = dropout
        self.dtype = dtype
        self.device = device

    def forward(self, x, mask=None, past_key_value=None, use_cache=False, rope=None):
        batch_size, seq_len, _ = x.shape
        x = x.to(self.q_proj.weight.dtype)

        q = self.q_proj(x) # [batch, seq_len, n_heads*head_dim]
        k = self.k_proj(x) # [batch, seq_len, n_kv_heads*head_dim]
        v = self.v_proj(x) # [batch, seq_len, n_kv_heads*head_dim]

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        if rope is not None:
            cos, sin = rope(q, seq_len)
            q, k = apply_rope(q, k, cos, sin)
        
        if past_key_value is not None:
            past_k, past_v = past_key_value
            if past_k.dim() == 4:
                k = torch.cat([past_k, k], dim=1)
                v = torch.cat([past_v, v], dim=1)
            else:
                print(f"Warning: past_k shape {past_k.shape}, current k shape {k.shape}")
        
        present_key_value = (k.clone(), v.clone()) if use_cache else None

        if self.n_groups > 1:
            k = k.repeat_interleave(self.n_groups, dim=2)
            v = v.repeat_interleave(self.n_groups, dim=2)

        q = q.transpose(1, 2).to(dtype=self.dtype, device=self.device)
        k = k.transpose(1, 2).to(dtype=self.dtype, device=self.device)
        v = v.transpose(1, 2).to(dtype=self.dtype, device=self.device)

        if mask is None:
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
                scale=None
            )
        else:
            if mask.dtype == torch.bool:
                attn_mask = torch.zeros_like(mask, dtype=q.dtype)
                attn_mask.masked_fill_(mask.logical_not(), float('-inf'))
            else:
                attn_mask = mask
            
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)  # [B, 1, q_len, kv_len]

            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False,
                scale=None
            )

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads*self.head_dim)
        out = self.o_proj(out)
        return out, present_key_value

class RegularGroupedSlidingAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, 
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6, window_size=128,
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        
        self.local_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_local, dtype=dtype, device=device)
        self.global_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_global, dtype=dtype, device=device)

        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, dropout, dtype=dtype, device=device)
        self.window_size = window_size

    def get_causal_mask(self, seq_len, device, is_global, kv_len=None):
        kv_len = kv_len or seq_len

        if is_global:
            return torch.tril(torch.ones(seq_len, kv_len, dtype=torch.bool, device=device))
        else:
            idxs_q = torch.arange(seq_len, device=device)
            idxs_k = torch.arange(kv_len, device=device)
            mask = idxs_q.view(-1, 1) >= (idxs_k.view(1, -1) - self.window_size)
            return mask

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        batch_size, seq_len, _ = x.shape

        kv_len = seq_len
        if past_key_value is not None:
            kv_len += past_key_value[0].size(2)  # shape: [B, H, past_len, D]

        is_global = (layer_idx % 6 == 5)
        mask = self.get_causal_mask(seq_len, x.device, is_global, kv_len=kv_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

        rope = self.global_rope if is_global else self.local_rope
        out, present_kv = self.attn(x, mask=mask, past_key_value=past_key_value, use_cache=use_cache, rope=rope)
        
        return out, present_kv

class SpikingGroupedAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, 
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6, 
                 beta=0.95, num_steps=10,  # Increased num_steps
                 dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        
        # Set head dimension if not provided
        self.head_dim = head_dim or d_model // n_heads
        
        # RoPE configurations (keeping both for potential layer alternation)
        self.local_rope = RoPE(self.head_dim, max_seq_len, theta=rope_theta_local, dtype=dtype, device=device)
        self.global_rope = RoPE(self.head_dim, max_seq_len, theta=rope_theta_global, dtype=dtype, device=device)
        
        # Standard attention mechanism
        self.attn = GroupedQueryAttention(
            d_model, n_heads, n_kv_heads, head_dim, 
            dropout, dtype=dtype, device=device
        )
        
        # Spiking components
        self.spike = snn.Leaky(beta=beta, learn_beta=True)
        self.num_steps = num_steps  # Increased from 5 to 10 for better temporal integration
        self.beta = beta

    def get_causal_mask(self, seq_len, device, kv_len=None):
        """Always return full causal mask (no sliding window)"""
        kv_len = kv_len or seq_len
        return torch.tril(torch.ones(seq_len, kv_len, dtype=torch.bool, device=device))

    def forward(self, x, use_cache=False, past_key_value=None, layer_idx=0):
        batch_size, seq_len, _ = x.shape
        kv_len = seq_len
        if past_key_value is not None:
            # past_key_value[0] shape: [batch_size, n_heads, past_seq_len, head_dim]
            kv_len += past_key_value[0].size(2)

        # Always use causal mask (full attention)
        mask = self.get_causal_mask(seq_len, x.device, kv_len=kv_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]
        
        # FIX: Convert input to spikes FIRST, before attention computation
        mem = self.spike.init_leaky()
        spike_acc = torch.zeros_like(x)
        
        # Generate spike trains over time
        for _ in range(self.num_steps):
            spk, mem = self.spike(x, mem)
            spike_acc += spk
        
        # Rate-coded representation of input
        x_rate = spike_acc / self.num_steps
        
        # Choose RoPE based on layer index (optional pattern)
        # You can customize this pattern as needed
        is_global = (layer_idx % 6 == 5)
        rope = self.global_rope if is_global else self.local_rope
        
        # Process the rate-coded input through attention
        out, present_kv = self.attn(
            x_rate, 
            mask=mask, 
            past_key_value=past_key_value, 
            use_cache=use_cache, 
            rope=rope
        )
        
        return out, present_kv

class PreRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, dtype=torch.bfloat16, device="cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps
        self.d_model = d_model
    
    def forward(self, x):
        input_dtype = x.dtype
        input_device = x.device
        
        if x.device != self.weight.device:
            x = x.to(self.weight.device)
        
        x_fp32 = x.float()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_normed = x_fp32 * torch.rsqrt(variance + self.eps)
        
        result = self.weight * x_normed
        return result.to(dtype=input_dtype, device=input_device)
