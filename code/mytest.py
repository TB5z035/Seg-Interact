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

a = np.array([2,1,5,3,4])
k = 3

c = np.argpartition(a,k-1)[:k]
d = a[c]
print(a)
print(c)
print(np.sort(d))