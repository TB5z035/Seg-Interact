import numpy as np
import os
import os.path as osp
import re
import torch
'''a = np.array([1,2,3])
os.makedirs('/home/Guest/caiz/Seg-Interact/configs/testing', exist_ok=True)
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.array([2,2,3])
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.load('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy')
print(a)'''
a = np.array([1, 2, 3, 4, 5, 6])
b = np.array([0, 0, 1])
c = np.delete(np.arange(len(a)), b)
print(c)
