import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn

def swish(x1):
    return x1 * torch.sigmoid(x1)

class SwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.fc = nn.Linear(input_dim, hidden_dim*2, dtype=dtype, device=device) # times(2) because will chunk into 2
        self.out = nn.Linear(hidden_dim, input_dim, dtype=dtype, device=device)
    
    def forward(self, x):
        x = self.fc(x)
        x1, x2 = x.chunk(2, dim=-1)
        swh = swish(x1)
        gated = swh * x2
        return self.out(gated)

class SpikingSwiGLU(nn.Module):
    def __init__(self, input_dim, hidden_dim, beta=0.95, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.swiglu = SwiGLU(input_dim, hidden_dim, dtype=dtype, device=device)
        self.lif = snn.Leaky(beta=beta, init_hidden=True)
    
    def forward(self, x):
        self.lif.reset_hidden()

        mem = self.lif.init_leaky()
        x = self.swiglu(x)
        #spk, mem = self.lif(x, mem)
        spk = self.lif(x)
        return spk

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
    def __init__(self, head_dim, max_seq_len, theta=10_000, dtype=torch.float32, device="cuda"):
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

def get_local_attn_mask(seq_len, window_size, device="cuda"):
    mask = torch.zeros(seq_len, seq_len, device=device, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size)
        mask[i, start:i+1] = 1  # causal: include i only up to itself
    return mask  # [seq_len, seq_len]

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, bias=False, dtype=torch.float32, device="cuda"):
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
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, past_key_value=None, use_cache=False, rope=None):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x) # [batch, seq_len, n_heads*head_dim]
        k = self.k_proj(x) # [batch, seq_len, n_kv_heads*head_dim]
        v = self.v_proj(x) # [batch, seq_len, n_kv_heads*head_dim]

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if rope is not None:
            # RoPE expects [batch, seq_len, n_heads, head_dim], so transpose back temporarily
            cos, sin = rope(q.transpose(1, 2), seq_len=seq_len)
            q_rope = q.transpose(1, 2)
            k_rope = k.transpose(1, 2)
            q_rope, k_rope = apply_rope(q_rope, k_rope, cos, sin)
            q = q_rope.transpose(1, 2)
            k = k_rope.transpose(1, 2)
            
        if past_key_value is not None:
            past_k, past_v = past_key_value
            k = torch.cat([past_k, k], dim=-2)
            v = torch.cat([past_v, v], dim=-2)
    
        present_key_value = (k, v) if use_cache else None

        # Expand K,V to match Q heads by repeating each KV head n_groups times
        # [batch, n_kv_heads, seq_len, head_dim] -> [batch, n_heads, seq_len, head_dim]
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask==0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, v) # [b, h, s, d]

        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_heads*self.head_dim)
        
        out = self.o_proj(out)
        return out, present_key_value

class SpikingGroupedSlidingAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, head_dim=None, dropout=0.0, 
                 max_seq_len=2048, rope_theta_local=1e4, rope_theta_global=1e6, window_size=128, beta=0.95,
                 dtype=torch.float32, device="cuda"):
        super().__init__()
        
        self.local_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_local, dtype=dtype, device=device)
        self.global_rope = RoPE(head_dim or d_model//n_heads, max_seq_len, theta=rope_theta_global, dtype=dtype, device=device)

        self.attn = GroupedQueryAttention(d_model, n_heads, n_kv_heads, head_dim, dropout, dtype=dtype, device=device)
        self.spike = snn.Leaky(beta=beta, init_hidden=True)
        
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
        self.spike.reset_hidden()

        batch_size, seq_len, _ = x.shape

        kv_len = seq_len
        if past_key_value is not None:
            kv_len += past_key_value[0].size(2)  # shape: [B, H, past_len, D]

        is_global = (layer_idx % 6 == 5)
        mask = self.get_causal_mask(seq_len, x.device, is_global, kv_len=kv_len)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

        rope = self.global_rope if is_global else self.local_rope
        out, present_kv = self.attn(x, mask=mask, past_key_value=past_key_value, use_cache=use_cache, rope=rope)

        mem = self.spike.init_leaky()
        #spk_out, mem = self.spike(out, mem)
        spk_out = self.spike(out)
        return spk_out, present_kv

class PreRMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, dtype=torch.float32, device="cuda"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model, dtype=dtype, device=device))
        self.eps = eps
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True)
        return x * self.weight / (norm + self.eps)
  
if __name__ == "__main__":
    print("Testing SwiGLU...")
    layer = SwiGLU(512, 2048)
    x = torch.randn(1, 128, 512)
    output = layer(x)
    assert x.shape == output.shape, f"Error - SwiGLU: {x.shape=} != {output.shape=}"
    print("✓ SwiGLU test passed")

    print("\nTesting SpikingSwiGLU...")
    spiking_layer = SpikingSwiGLU(512, 2048)
    x = torch.randn(1, 128, 512)
    output = spiking_layer(x)
    assert x.shape == output.shape, f"Error - SpikingSwiGLU: {x.shape=} != {output.shape=}"
    # Check that output is binary (spiking)
    assert torch.all((output == 0) | (output == 1)), "Error - SpikingSwiGLU: Output should be binary spikes"
    print("✓ SpikingSwiGLU test passed")

    print("\nTesting RoPE...")
    head_dim = 64
    max_seq_len = 512
    rope = RoPE(head_dim, max_seq_len)
    x = torch.randn(1, 128, 8, head_dim)  # [batch, seq_len, n_heads, head_dim]
    cos, sin = rope(x, seq_len=128)
    expected_cos_shape = (128, head_dim // 2)
    expected_sin_shape = (128, head_dim // 2)
    assert cos.shape == expected_cos_shape, f"Error - RoPE cos: {cos.shape=} != {expected_cos_shape=}"
    assert sin.shape == expected_sin_shape, f"Error - RoPE sin: {sin.shape=} != {expected_sin_shape=}"
    print("✓ RoPE test passed")

    print("\nTesting GroupedQueryAttention...")
    d_model = 512
    n_heads = 8
    n_kv_heads = 2
    gqa = GroupedQueryAttention(d_model, n_heads, n_kv_heads)
    x = torch.randn(1, 128, d_model)
    output, present_kv = gqa(x, use_cache=True)
    assert output.shape == x.shape, f"Error - GQA output: {output.shape=} != {x.shape=}"
    assert present_kv is not None, "Error - GQA: present_kv should not be None when use_cache=True"
    past_k, past_v = present_kv
    # After transpose, format is [batch, n_kv_heads, seq_len, head_dim]
    expected_kv_shape = (1, n_kv_heads, 128, d_model // n_heads)  
    assert past_k.shape == expected_kv_shape, f"Error - GQA past_k: {past_k.shape=} != {expected_kv_shape=}"
    assert past_v.shape == expected_kv_shape, f"Error - GQA past_v: {past_v.shape=} != {expected_kv_shape=}"
    print("✓ GroupedQueryAttention test passed")

    print("\nTesting GroupedQueryAttention with RoPE...")
    rope_gqa = RoPE(d_model // n_heads, 512)
    output_rope, _ = gqa(x, rope=rope_gqa)
    assert output_rope.shape == x.shape, f"Error - GQA with RoPE: {output_rope.shape=} != {x.shape=}"
    print("✓ GroupedQueryAttention with RoPE test passed")

    print("\nTesting SpikingGroupedSlidingAttention...")
    spiking_attn = SpikingGroupedSlidingAttention(d_model, n_heads, n_kv_heads, window_size=64)
    x = torch.randn(1, 128, d_model)
    
    # Test local attention (layer_idx % 6 != 5)
    output_local, present_kv_local = spiking_attn(x, layer_idx=0, use_cache=True)
    assert output_local.shape == x.shape, f"Error - Spiking Attn local: {output_local.shape=} != {x.shape=}"
    assert present_kv_local is not None, "Error - Spiking Attn: present_kv should not be None when use_cache=True"
    # Check that output is binary (spiking)
    assert torch.all((output_local == 0) | (output_local == 1)), "Error - Spiking Attn: Output should be binary spikes"
    
    # Test global attention (layer_idx % 6 == 5)
    output_global, present_kv_global = spiking_attn(x, layer_idx=5, use_cache=True)
    assert output_global.shape == x.shape, f"Error - Spiking Attn global: {output_global.shape=} != {x.shape=}"
    assert present_kv_global is not None, "Error - Spiking Attn: present_kv should not be None when use_cache=True"
    assert torch.all((output_global == 0) | (output_global == 1)), "Error - Spiking Attn: Output should be binary spikes"
    print("✓ SpikingGroupedSlidingAttention test passed")

    print("\nTesting helper functions...")
    
    # Test rotate_half
    test_tensor = torch.randn(2, 4)
    rotated = rotate_half(test_tensor)
    assert rotated.shape == test_tensor.shape, f"Error - rotate_half: {rotated.shape=} != {test_tensor.shape=}"
    
    # Test apply_rope
    q = torch.randn(1, 10, 4, 32)
    k = torch.randn(1, 10, 4, 32)
    cos = torch.randn(10, 16)
    sin = torch.randn(10, 16)
    q_rot, k_rot = apply_rope(q, k, cos, sin)
    assert q_rot.shape == q.shape, f"Error - apply_rope q: {q_rot.shape=} != {q.shape=}"
    assert k_rot.shape == k.shape, f"Error - apply_rope k: {k_rot.shape=} != {k.shape=}"
    
    # Test get_local_attn_mask
    mask = get_local_attn_mask(10, 3, device="cpu")
    expected_mask_shape = (10, 10)
    assert mask.shape == expected_mask_shape, f"Error - local mask: {mask.shape=} != {expected_mask_shape=}"
    assert mask.dtype == torch.bool, f"Error - local mask dtype: {mask.dtype=} != torch.bool"
    
    print("✓ Helper functions test passed")

    print("\n" + "="*50)
    print("🎉 ALL TESTS PASSED! 🎉")
    print("="*50)