import numpy as np
import os
import torch
'''a = np.array([1,2,3])
os.makedirs('/home/Guest/caiz/Seg-Interact/configs/testing', exist_ok=True)
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.array([2,2,3])
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.load('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy')
print(a)'''

a = np.array([2, 1, 5, 3, 4])
c = np.array([1, 2, 3, 4, 5, 6])
b = np.array([0, 1, 2])
a[b] = c[b]
print(a)
