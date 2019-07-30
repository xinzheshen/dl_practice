import numpy as np

a = np.arange(10)
print(a)
print("a.shape", a.shape)

b = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(b)
print("b.shape", b.shape)

c = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
print(c)
print("c.shape", c.shape)

print(str(c[0]))
print(str(c[0:5]))
print(str(c[0:5:2]))
print(str(c[:,2]))

d = c.reshape(5,-1)
print(d)
print("d.shape", d.shape)
print(str(d[0]))
print(str(d[0:5]))
print(str(d[0:5:2]))
print(str(d[:,1]))


a = np.array([[1,2],[3,4]])
b = np.array([[11,12],[13,14]])
print(np.dot(a,b))
