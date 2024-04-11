import torch
import torch.nn.functional as F
from LLM4Bio.utils import clip


encoded = torch.randn(2, 7, 3)

print(encoded.shape)
print(encoded.flatten(start_dim=1).shape)

print(encoded)
print(encoded.flatten(start_dim=1))
