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

    def forward(self, x, seq_len,start_pos  = 0):
        
        device = x.device
        t = torch.arange(start_pos, start_pos+seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) 
        
       
        cos = emb.cos().unsqueeze(0).unsqueeze(0)
        sin = emb.sin().unsqueeze(0).unsqueeze(0)
        return cos.to(x.dtype), sin.to(x.dtype)
def make_mask(T, start_pos ,device):
        S  = start_pos + T
        q_pos = torch.arange(start_pos, S,device = device).unsqueeze(1)
        k_pos = torch.arange(0, S, device =  device).unsqueeze(0)

        return k_pos > q_pos



class Attention(nn.Module):

    def __init__(self, hidden_size=896, num_heads=14, num_kv_heads=2, head_dim=64, rope_theta=1000000, layer_idx = 0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_groups = num_heads // num_kv_heads
        self.layer_idx = layer_idx

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

    def forward(self, hidden_states,cache = None,start_pos = 0):
        # hidden_states shape: (batch, T, hidden_size)
        batch_size, T, _ = hidden_states.shape
        
        
        q = self.q_proj(hidden_states).view(batch_size, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch_size, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch_size, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        
        cos, sin = self.rotary_emb(q, T, start_pos)
        q, k = apply_rotatry(q, k, cos, sin)
        
        if cache is not None:
            assert cache.length == start_pos        
            k, v = cache.update(self.layer_idx, k, v) 
        
        k = self.expand_gqa_vector(k)
        v = self.expand_gqa_vector(v)
        
        
        attn_weights = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        mask = make_mask(T, start_pos, q.device)
        attn_weights = attn_weights.masked_fill(mask, float("-inf"))
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        
        context_layer = torch.matmul(attn_weights, v)
        
        
        context_layer = context_layer.transpose(1, 2).contiguous()
        context_layer = context_layer.view(batch_size, T, self.num_heads * self.head_dim)
        
        return self.o_proj(context_layer)





            


class DecoderLayer(nn.Module):
    def __init__(self, cfg,layer_idx:int):
        super().__init__()
        hidden = cfg.hidden_size
        head_dim = hidden // cfg.num_attention_heads
        theta = cfg.rope_parameters["rope_theta"]   

        self.input_layernorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.self_attn = Attention(hidden, cfg.num_attention_heads,
                                   cfg.num_key_value_heads, head_dim, theta,layer_idx=layer_idx)
        self.post_attention_layernorm = RMSNorm(hidden, cfg.rms_norm_eps)
        self.mlp = SwiGlU(hidden, cfg.intermediate_size)

    def forward(self, x,cache = None, start_pos = 0):
        x = x + self.self_attn(self.input_layernorm(x),cache = cache, start_pos = start_pos)        # norm -> attention -> add
        x = x + self.mlp(self.post_attention_layernorm(x))     # norm -> MLP -> add
        return x


class QwenBody(nn.Module):         
    def __init__(self, cfg):
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(cfg,i) for i in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, ids,cache = None, start_pos = 0):
        x = self.embed_tokens(ids)
        for layer in self.layers:
            x = layer(x,cache = cache, start_pos = start_pos)
        return self.norm(x)


class QwenForCausalLM(nn.Module):
    def __init__(self, cfg,):
        super().__init__()
        self.model = QwenBody(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:                  # true for Qwen2.5-0.5B
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, ids,cache = None , start_pos = 0,logits_last_only=False):
        T = ids.shape[1]
        x = self.model(ids,cache = cache, start_pos = start_pos)
        if cache is not None:
            cache.advance(T)
        if logits_last_only:
            x = x[:, -1:, :]
        return self.lm_head(x)         # logits: (batch, tokens, vocab)


@torch.no_grad()
def greedy_generate(model, ids, n_new):
    for _ in range(n_new):
        logits = model(ids)                                   # naive: reprocess everything
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
    return ids


class KVCache:
    def __init__(self,num_layers,batch,num_kv_heads,max_len,head_dim,dtype,device):
        self.k_cache = [torch.zeros(batch, num_kv_heads, max_len, head_dim,dtype=dtype,device=device)
            for _ in range(num_layers)
        ]

        self.v_cache = [torch.zeros(batch, num_kv_heads, max_len, head_dim,dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.length = 0
        self.max_len = max_len

    def update(self, layer_idx, k_new, v_new):
        assert k_new.shape[0] == self.k_cache[layer_idx].shape[0]
        

    
        assert k_new.shape[1] == self.num_kv_heads

    
        assert k_new.shape[3] == self.head_dim

    
        assert k_new.shape == v_new.shape

    
       

    
        T = k_new.shape[2]
        end = self.length + T
        

        if end > self.max_len:
            raise ValueError("KV Cache capacity exceeded!")

        self.k_cache[layer_idx][:, :, self.length:end, :] = k_new
        self.v_cache[layer_idx][:, :, self.length:end, :] = v_new

        return (
            self.k_cache[layer_idx][:, :, :end, :],
            self.v_cache[layer_idx][:, :, :end, :]
        )

    def advance(self, T):
        if self.length + T > self.max_len:
            raise ValueError("KV Cache capacity exceeded!")

        self.length += T
    def reset(self):
        self.length = 0    

@torch.no_grad()
def generate_cached(model, ids, n_new, eos_id=None):
    assert n_new >= 1

    p = next(model.parameters())                 
    attn = model.model.layers[0].self_attn       
    cache = KVCache(
        num_layers=len(model.model.layers),
        batch=ids.shape[0],
        num_kv_heads=attn.num_kv_heads,
        max_len=ids.shape[1] + n_new,           
        head_dim=attn.head_dim,
        dtype=p.dtype,                          
        device=p.device,
    )

    logits = model(ids, cache=cache, start_pos=0, logits_last_only=True)
    nxt = logits[:, -1].argmax(-1, keepdim=True)         # first new token
    ids = torch.cat([ids, nxt], dim=1)

    for _ in range(n_new - 1):                           # first token already came from prefill
        if eos_id is not None and nxt.item() == eos_id:
            break
        logits = model(nxt, cache=cache, start_pos=cache.length, logits_last_only=True)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)

    return ids