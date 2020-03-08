# 一维线性回归

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn, optim
# 给定数据集
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], [9.779], [6.182], [7.59],
                    [2.167], [7.042], [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)
y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596], [2.53], [1.221],
                    [2.827], [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# 数据可视化
plt.plot(x_train, y_train, 'ro', label='Original data')

# 将训练集由numpy.array转换为Tensor类型
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)

# 建立模型
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)   # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out

# 如果GPU加速，可以通过model.cuda()将模型放到GPU上
if torch.cuda.is_available():
    model = LinearRegression().cuda()
else:
    model = LinearRegression()

# 定义损失函数
criterion = nn.MSELoss()   # 平方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 开始训练
num_epoches = 100
for epoch in range(num_epoches):
    # 将数据变成Variable放入计算图中
    if torch.cuda.is_available():
        inputs = Variable(x_train).cuda()
        target = Variable(y_train).cuda()
    else:
        inputs = Variable(x_train)
        target = Variable(y_train)

    # forward
    out = model(inputs)
    loss = criterion(out, target)
    # backward
    optimizer.zero_grad()   # 梯度归零
    loss.backward()    # 反向传播，自动求导得到每个参数的梯度
    optimizer.step()     # 梯度做进一步参数更新

    if (epoch+1) % 20 == 0:
        print("Epoch[{}/{}], loss: {:.6f}".format(epoch+1, num_epoches, loss.data))

# 开始做预测
# 将模型变成测试模式，这是因为有一些层操作，比如Dropout和BatchNormalization在训练和测试时是不一样的，所以我们需要通过这样一个操作来转换这些不一的层
model.eval()
# 计算预测值
predict = model(Variable(x_train))
# 将预测值从Tensor转换为numpy类型
predict = predict.data.numpy()
# 画出数据线
plt.plot(x_train.numpy(), predict, label='Fitting Line')
plt.show()
