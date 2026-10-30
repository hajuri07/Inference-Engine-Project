import torch.nn.functional as F
import torch
import torch.nn as nn


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


            
