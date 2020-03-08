import torch
from torch import nn, optim
from torch.autograd import Variable

from src.course1_02.lr_utils import load_dataset



train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

train_set_x_orig_flatten = torch.from_numpy(train_set_x_orig).view(train_set_x_orig.shape[0], -1)
test_set_x_orig_flatten = torch.from_numpy(test_set_x_orig).view(test_set_x_orig.shape[0], -1)

input_size = train_set_x_orig_flatten.size()[-1]


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression, self).__init__()
        self.lr = nn.Linear(input_size, 1)
        self.sm = nn.Sigmoid()

    def forward(self, x):
        z = self.lr(x)
        a = self.sm(z)
        return a


logistic_model = LogisticRegression()
# 定义损失函数和优化器
criterion = nn.BCELoss()      # nn.BCELoss是二分类的损失函数
optimizer = optim.SGD(logistic_model.parameters(), lr=1e-3, momentum=0.9)

# 训练模型
for epoch in range(2000):
    # 转换为Variable
    x_train = train_set_x_orig_flatten.float()
    y_train = torch.from_numpy(train_set_y).transpose(0, 1).float()
    # forward
    out = logistic_model(x_train)
    loss = criterion(out, y_train)
    # 计算准确率
    mask = out.ge(0.5).float()
    correct = (mask == y_train).sum().float()
    acc = correct/x_train.size(0)
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(loss.data)
    if (epoch+1) % 100 == 0:
        print('*'*10)
        print("eopch {}, loss is {:.4f}, acc is {:.4f}".format(epoch+1, loss, acc))


# 预测
logistic_model.eval()
x_test = test_set_x_orig_flatten.float()
y_test = torch.from_numpy(test_set_y).transpose(0, 1).float()
out = logistic_model(x_test)
mask = out.ge(0.5).float()
correct = (mask == y_test).sum().float()
acc = correct/y_test.size(0)
print("测试集准确率：", acc)
