import os

import numpy as np
import torch

'''a = np.array([1,2,3])
os.makedirs('/home/Guest/caiz/Seg-Interact/configs/testing', exist_ok=True)
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.array([2,2,3])
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.load('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy')
print(a)'''

a = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
print(*a)
x, y, z = list(zip(*a))
print(x)
