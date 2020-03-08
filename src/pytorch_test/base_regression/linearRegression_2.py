import torch
from torch import nn, optim
from torch.autograd import Variable

# 构造特征
def make_features(x):
    x = x.unsqueeze(1)
    return torch.cat([x**i for i in range(1, 4)], 1)

# 定义好真实的函数
W_target = torch.FloatTensor([0.5, 3, 2.4]).unsqueeze(1)    # 将原来的tensor大小由3变成(3, 1)
b_target = torch.FloatTensor([0.9])
print(W_target)

def f(x):
    return x.mm(W_target) + b_target

# 得到训练集合
def get_batch(batch_size=1):
    random = torch.randn(batch_size)
    x = make_features(random)
    y = f(x)
    if torch.cuda.is_available():
        return Variable(x).cuda(), Variable(y).cuda()
    else:
        return Variable(x), Variable(y)

# 构建模型
class poly_model(nn.Module):
    def __init__(self):
        super(poly_model, self).__init__()
        self.poly = nn.Linear(3, 1)
    def forward(self, x):
        out = self.poly(x)
        return out

if torch.cuda.is_available():
    model = poly_model().cuda()
else:
    model = poly_model()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# 开始训练
epoch = 0
while True:
    # get data
    batch_x, batch_y = get_batch()
    # forward
    output = model(batch_x)

    loss = criterion(output, batch_y)
    print(loss)
    print_loss = loss.data.float()
    # Reset gradients
    optimizer.zero_grad()
    # backward
    loss.backward()
    # update parameters
    optimizer.step()
    epoch += 1
    if print_loss < 1e-4:
        print("Loss: {} after {} batches".format(print_loss, epoch))
        print("Actual function: y = 0.90 + 0.50x + 3.00*x^2 + 2.40*x^3")
        for name,param in model.named_parameters():
            print(name, param)
        break


# 预测
model.eval()

# x_test, y_test = get_batch(20)
# y_pred = model(x_test)

x, _ = torch.randn(20).sort()
x_test = make_features(x)
y_test = f(x_test)
y_pred = model(Variable(x_test))
# 数据可视化
import matplotlib.pyplot as plt

plt.plot(x.numpy(), y_test.detach().numpy(), 'ro', label='Original data')
plt.plot(x.numpy(), y_pred.detach().numpy())
plt.show()