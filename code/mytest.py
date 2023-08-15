import numpy as np
import os
import os.path as osp
import re
import torch
'''
x = np.array([[1, 2, 3], [4, 5, 6]])
y = np.divide(x[:, 2], 2)
y = np.expand_dims(y, axis=1)
print(x[:, :2])
print(y)
x = np.concatenate((x[:, :2], y), axis=1)
print(x)
'''
x = np.array([[1, 2, 3], [4, 5, 6]])
y = x[:, 1:]
print(y)
y[:, 0] = 0
print(y)

print(9 % 8)
