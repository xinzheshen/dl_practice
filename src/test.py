import sys
tmp1 = sys.modules
import numpy as np
tmp2 = sys.modules

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


print("最大索引-------------")
a = np.array([3, 1, 2, 4, 6, 1])
print(np.argmax(a))
print("-----")
a = np.array([[1, 5, 5, 2],
              [9, 6, 2, 8],
              [3, 7, 9, 1]])
# 沿着axis=0， 相当于在各个行之间比较相同列的大小，结果[1 2 2 1]
print(np.argmax(a, axis=0))
print("--")
# 沿着axis=1， 相当于在各个列之间比较相同行的大小，结果[1 0 2]
print(np.argmax(a, axis=1))

print("-----三维")
a = np.array([
    [
        [1, 5, 5, 2],
        [9, -6, 2, 8],
        [-3, 7, -9, 1]
    ],

    [
        [-1, 5, -5, 2],
        [9, 6, 2, 8],
        [3, 7, 9, 1]
    ]
])
print("shape", a.shape)
print(np.argmax(a, axis=0))
print("--")
print(np.argmax(a, axis=1))


np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
tmp = p.ravel()
index = np.random.choice([0, 1, 2, 3], p=tmp)
print("index", index)