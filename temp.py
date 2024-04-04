import torch
import torch.nn as nn


a = [[1, 2], [3, 4], [5, 6]]
a = torch.tensor(a, dtype=torch.float32)
print(a)
print(a.shape)
b = nn.functional.pad(a, (0, 0, 0, 1))
print(b)
print(b.shape)
