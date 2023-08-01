import numpy as np
import os
import re
import torch
'''a = np.array([1,2,3])
os.makedirs('/home/Guest/caiz/Seg-Interact/configs/testing', exist_ok=True)
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.array([2,2,3])
np.save('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy', a)
a = np.load('/home/Guest/caiz/Seg-Interact/configs/testing/my.npy')
print(a)'''


def f7(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


print(f7([1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 6, 6, 6, 6, 7, 7, 7]))
