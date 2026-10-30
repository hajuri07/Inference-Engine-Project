# #this is testing like if we get the same outputs then our rms is correct and we can now move on hehe
# import torch
# from model  import RMSNorm
# from transformers.models.qwen2.modeling_qwen2 import Qwen2RMSNorm

# for dt in (torch.float32, torch.float16):
#     x = torch.randn(2, 8, 896, dtype=dt)
#     hf = Qwen2RMSNorm(896, eps=1e-6).to(dt)
#     hf.weight.data = torch.randn(896, dtype=dt)
#     mine = RMSNorm(896, eps=1e-6).to(dt)
#     mine.weight.data.copy_(hf.weight.data)
#     print(dt, torch.allclose(hf(x), mine(x), atol=1e-3))


