from __future__ import print_function
import torch

"""Tensor"""
print("Tensors -------------")
# Tensors are similar to NumPy’s ndarrays, with the addition being that Tensors can also be used on a GPU to accelerate computing.
## Construct a 5x3 matrix, uninitialized:
x = torch.empty(5, 3)
print(x)

## Construct a randomly initialized matrix:
x = torch.rand(5, 3)
print(x)

## Construct a matrix filled zeros and of dtype long:
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

## Construct a tensor directly from data:
x = torch.tensor([5.5, 3])
print(x)


x = x.new_ones(5, 3, dtype=torch.double)      # new_* methods take in sizes
print(x)

x = torch.randn_like(x, dtype=torch.float)    # override dtype!
print(x)                                      # result has the same size
print(x.size())


"""Operations"""
print("Operations ----------------------------")
## addition:
y = torch.rand(5, 3)
print(x + y)
print(torch.add(x, y))
### providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

### in-place: adds x to y
# note:Any operation that mutates a tensor in-place is post-fixed with an _. For example: x.copy_(y), x.t_(), will change x.
y.add_(x)
print(y)

print(x[:, 1])

## resizing: If you want to resize/reshape tensor, you can use torch.view:
x = torch.randn(4, 4)
y = x.view(16)
z = x.view((-1, 8)) # the size -1 is inferred from other
print(x.size(), y.size(), z.size())


# If you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

# Read later:
#
# 100+ Tensor operations, including transposing, indexing, slicing, mathematical operations, linear algebra,
# random numbers, etc., are described https://pytorch.org/docs/stable/torch.html

"""numpy bridge"""
print("numpy bridge -----------------")
# The Torch Tensor and NumPy array will share their underlying memory locations (if the Torch Tensor is on CPU), and changing one will change the other.
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# See how the numpy array changed in value
a.add_(1)
print(a)
print(b)

import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.


"""CUDA Tensors"""
print("CUDA Tensors --------")
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
