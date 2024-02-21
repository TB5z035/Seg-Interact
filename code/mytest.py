import numpy as np
import torch
import torch.nn as nn

a = [torch.rand(5,2,3), torch.rand(5,3,3)]
print(torch.concat(a, dim=1).shape)
