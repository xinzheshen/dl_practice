import torch

"""
Central to all neural networks in PyTorch is the autograd package. Let’s first briefly visit this, 
and we will then go to training our first neural network.

The autograd package provides automatic differentiation for all operations on Tensors. It is a define-by-run framework,
which means that your backprop is defined by how your code is run, and that every single iteration can be different.
"""

"""Tensor"""
# torch.Tensor is the central class of the package.

# Create a tensor and set requires_grad=True to track computation with it
x = torch.ones(2, 2, requires_grad=True)
print(x)

y = x + 2
print(y)
# y was created as a result of an operation, so it has a grad_fn.
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print("z", z)
print("out", out)

# .requires_grad_( ... ) changes an existing Tensor’s requires_grad flag in-place.
# The input flag defaults to False if not given.
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

"""Gradients"""
# Let’s backprop now. Because out contains a single scalar, out.backward() is equivalent to out.backward(torch.tensor(1.)).
out.backward()
# Print gradients d(out)/dx
print(x.grad)


x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2

print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)


# You can also stop autograd from tracking history on Tensors with .requires_grad=True by wrapping the code block in with torch.no_grad():
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)