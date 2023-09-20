import numpy as np
import os
import os.path as osp
import re
import torch
import torch.nn as nn
from itertools import repeat
import argparse
import os.path as osp
import yaml
'''
lin = nn.Linear(3,9)
m = torch.rand((2,3,3))
print(m)
qkv = lin(m)
print(qkv.shape)
#             B N 3 H C          3 B H N C
qkv = qkv.reshape(2,3,3,1,3).permute(2,0,3,1,4)
q = qkv[0]
k = qkv[1]
print(k.shape)
k = k.transpose(-2,-1)
print(k.shape)
s = q @ k
print(s.shape)
'''
a = torch.rand((2, 1, 2, 3))
b = torch.rand((2, 1, 2, 3))
print(b.shape)
b = b.transpose(-2, -1)
print(b.shape)
c = a @ b
print(c.shape)
