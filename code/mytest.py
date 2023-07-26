import torch

a = torch.tensor([1,2,3,4,5,6])
b = torch.tensor([True,True,True,True,False,False])
print(a[b])