import torch.nn.functional as F
import torch
import torch.nn as nn
import math
class RMSNorm(nn.Module):
    "Root mean square layer normalization"

    def __init__(self,dim : int, eps:float = 1e-6):
        super().__init__()
        
        self.eps = eps # this calculates the "e" stands for epsilon
        self.weight = nn.Parameter(torch.ones(dim)) # this calculates "y"

    def forward(self,x:torch.Tensor) ->torch.Tensor:
            #now we gonna calculate RMS
            dtype = x.dtype
            x = x.float()
            
            rms = torch.rsqrt(x.pow(2).mean(-1,keepdim = True) + self.eps)
            x = x.to(dtype)
            return (x*rms).type_as(x)*self.weight


class SwiGlU(nn.Module):
    "This is an activation function mostly use in Llama and mistral models"

    def __init__(self,dim:int, hidden_dim:int):
         super().__init__()
         self.gate_proj = nn.Linear(dim, hidden_dim, bias = False) #W_gate
         self.up_proj = nn.Linear(dim, hidden_dim, bias = False)#W_up
         self.down_proj = nn.Linear(hidden_dim, dim,bias = False)#W_down

    def forward(self,x : torch.Tensor )->torch.Tensor:
        gate  = F.silu(self.gate_proj(x))

        up = self.up_proj(x)

        gated_linear_unit = gate * up
    
        return self.down_proj(gated_linear_unit)


def rotate_half(x):
    x1 = x[...,:x.shape[-1]//2]
    x2  = x[...,x.shape[-1]//2:]

    return torch.cat((-x2,x1), dim = -1)

def apply_rotatry(q,k,cos,sin):
    q_emb = (q * cos) + (rotate_half(q) * sin )
    k_emb = (k * cos) + (rotate_half(k) * sin)

    return q_emb , k_emb

class ROPE(nn.Module):
    def __init__(self, dim = 64, base = 1000000):
        super().__init__()
        self.dim = dim
        self.base = base
        inv_freq = 1.0/(self.base **(torch.arange(0,self.dim,2, dtype = torch.int64).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent= False)

    def forward(self, x, seq_len):
        
        device = x.device
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) 
        
       
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos.to(x.dtype), sin.to(x.dtype)

class Attention(nn.Module):

    def __init__(self, hidden_size=896, num_heads=14, num_kv_heads=2, head_dim=64, rope_theta=1000000):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

        self.rotary_emb = ROPE(dim = self.head_dim, base  = rope_theta)

    def expand_gqa_vector(self,kv_tensor):
        return torch.repeat_interleave(kv_tensor,repeats=self.num_groups,dim = 1)

    def expand_gqa_loop(self,kv_tensor):
        outputs = []

        for i in range(self.num_heads):
            kv_head_idx = i // self.num_groups
            outputs.append(kv_tensor[:,kv_head_idx:kv_head_idx+1,:,:])

        return torch.cat(outputs, dim=1)

    def forward(self, hidden_states):
        # hidden_states shape: (batch, seq_len, hidden_size)
        batch_size, seq_len, _ = hidden_states.shape
        
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        
        cos, sin = self.rotary_emb(q, seq_len)
        q, k = apply_rotatry(q, k, cos, sin)
        
        
        k = self.expand_gqa_vector(k)
        v = self.expand_gqa_vector(v)
        
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=q.device), diagonal=1)
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        context_layer = torch.matmul(attn_weights, v)
        
        
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, seq_len, self.num_heads * self.head_dim)
        
        return self.o_proj(context_layer)





            
# ---- Phase 0 wiring: decoder layer, model, generation ----

class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hidden = cfg.hidden_size
        head_dim = hidden // cfg.num_attention_heads
        theta = cfg.rope_parameters["rope_theta"]   # nested in transformers 5.x

        # attribute names must match Qwen's checkpoint keys
        self.input_layernorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.self_attn = Attention(hidden, cfg.num_attention_heads,
                                   cfg.num_key_value_heads, head_dim, theta)
        self.post_attention_layernorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.mlp = SwiGlU(hidden, cfg.intermediate_size)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))        # norm -> attention -> add
        x = x + self.mlp(self.post_attention_layernorm(x))     # norm -> MLP -> add
        return x


class QwenBody(nn.Module):          # saved as "model" in the checkpoint
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, ids):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class QwenForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.model = QwenBody(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:                  # true for Qwen2.5-0.5B
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, ids):
        return self.lm_head(self.model(ids))         # logits: (batch, tokens, vocab)


@torch.no_grad()
def greedy_generate(model, ids, n_new):
    for _ in range(n_new):
        logits = model(ids)                                   # naive: reprocess everything
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids