import numpy as np
import matplotlib.pyplot as plt

# 考虑一个True和False的一维向量，你要为其计算序列中“False to True”转换的数量
np.random.seed(444)

x = np.random.choice([False, True], size=100000)
print(np.count_nonzero(x[:-1] < x[1:]))

# 假定一只股票的历史价格是一个序列，假设你只允许进行一次购买和一次出售，那么可以获得的最大利润是多少？
# 例如，假设价格=(20，18，14，17，20，21，15)，最大利润将是7，从14买到21卖。

prices = np.full(100, fill_value=np.nan)
prices[[0, 25, 60, -1]] = [80., 30., 75., 50.]

x = np.arange(len(prices))
is_valid = ~np.isnan(prices)
prices = np.interp(x=x, xp=x[is_valid], fp=prices[is_valid])
prices += np.random.randn(len(prices)) * 2
print(prices)

mn = np.argmin(prices)
mx = np.argmax(prices)
kwargs = {'markersize': 12, 'linestyle': ""}

fig, ax = plt.subplots()
ax.plot(prices)
ax.set_title('Price History')
ax.set_xlabel('Time')
ax.set_ylabel('Price')
ax.plot(mn, prices[mn], color='green', **kwargs)
ax.plot(mx, prices[mx], color='red', **kwargs)
plt.show()