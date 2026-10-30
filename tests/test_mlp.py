import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import torch
from transformers import AutoConfig
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from model import SwiGlU

cfg = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B")
hf = Qwen2MLP(cfg)
mine = SwiGlU(cfg.hidden_size, cfg.intermediate_size)
mine.load_state_dict(hf.state_dict())
x = torch.randn(2, 8, cfg.hidden_size)
print(torch.allclose(hf(x), mine(x), atol=1e-5))