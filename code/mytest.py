import numpy as np
import os
import os.path as osp
import re
import torch
from itertools import repeat
import argparse
import os.path as osp
import yaml

class a():
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def method(self):
        print(1)

class b(a):
    def __init__(self):
        pass

    def method2(self):
        print(2)
        super().__init__(3,4)


this = b()
this.method2()