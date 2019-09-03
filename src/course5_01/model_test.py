import numpy as np
import rnn_utils

import random
import time
import cllm_utils

def rnn_cell_forward(xt, a_prev, parameters):
    """
    根据图2实现RNN单元的单步前向传播

    参数：
        xt -- 时间步“t”输入的数据，维度为（n_x, m）
        a_prev -- 时间步“t - 1”的隐藏状态，维度为（n_a, m）
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a_next -- 下一个隐藏状态，维度为（n_a， m）
        yt_pred -- 在时间步“t”的预测，维度为（n_y， m）
        cache -- 反向传播需要的元组，包含了(a_next, a_prev, xt, parameters)
    """

    # 从“parameters”获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 使用上面的公式计算下一个激活值
    a_next = np.tanh(np.dot(Waa, a_prev) + np.dot(Wax, xt) + ba)

    # 使用上面的公式计算当前单元的输出
    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

    # 保存反向传播需要的值
    cache = (a_next, a_prev, xt, parameters)

    return a_next, yt_pred, cache


def rnn_forward(x, a0, parameters):
    """
    根据图3来实现循环神经网络的前向传播

    参数：
        x -- 输入的全部数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为 (n_a, m)
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y_pred -- 所有时间步的预测，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """

    # 初始化“caches”，它将以列表类型包含所有的cache
    caches = []

    # 获取x 与Wya 的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wya"].shape

    # 使用0 来初始化 "a" and "y"
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    # 初始化next
    a_next = a0

    # 遍历所有时间步
    for t in range(T_x):
        ## 1.使用rnn_cell_forward函数来更新“next”隐藏状态与cache。
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

        ## 2.使用 a 来保存“next”隐藏状态（第 t ）个位置。
        a[:, :, t] = a_next

        ## 3.使用 y 来保存预测值。
        y_pred[:, :, t] = yt_pred

        ## 4.把cache保存到“caches”列表中。
        caches.append(cache)

    # 保存反向传播所需要的参数
    caches = (caches, x)

    return a, y_pred, caches


def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    根据图4实现一个LSTM单元的前向传播。

    参数：
        xt -- 在时间步“t”输入的数据，维度为(n_x, m)
        a_prev -- 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
        c_prev -- 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
        parameters -- 字典类型的变量，包含了：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    返回：
        a_next -- 下一个隐藏状态，维度为(n_a, m)
        c_next -- 下一个记忆状态，维度为(n_a, m)
        yt_pred -- 在时间步“t”的预测，维度为(n_y, m)
        cache -- 包含了反向传播所需要的参数，包含了(a_next, c_next, a_prev, c_prev, xt, parameters)

    注意：
        ft/it/ot表示遗忘/更新/输出门，cct表示候选值(c tilda)，c表示记忆值。
    """

    # 从“parameters”中获取相关值
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # 获取 xt 与 Wy 的维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 1.连接 a_prev 与 xt
    contact = np.zeros([n_a + n_x, m])
    contact[: n_a, :] = a_prev
    contact[n_a :, :] = xt

    # 2.根据公式计算ft、it、cct、c_next、ot、a_next

    ## 遗忘门，公式1
    ft = rnn_utils.sigmoid(np.dot(Wf, contact) + bf)

    ## 更新门，公式2
    it = rnn_utils.sigmoid(np.dot(Wi, contact) + bi)

    ## 更新单元，公式3
    cct = np.tanh(np.dot(Wc, contact) + bc)

    ## 更新单元，公式4
    #c_next = np.multiply(ft, c_prev) + np.multiply(it, cct)
    c_next = ft * c_prev + it * cct
    ## 输出门，公式5
    ot = rnn_utils.sigmoid(np.dot(Wo, contact) + bo)

    ## 输出门，公式6
    #a_next = np.multiply(ot, np.tan(c_next))
    a_next = ot * np.tanh(c_next)
    # 3.计算LSTM单元的预测值
    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)

    # 保存包含了反向传播所需要的参数
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


def lstm_forward(x, a0, parameters):
    """
    根据图5来实现LSTM单元组成的的循环神经网络

    参数：
        x -- 所有时间步的输入数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为(n_a, m)
        parameters -- python字典，包含了以下参数：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y -- 所有时间步的预测值，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """

    # 初始化“caches”
    caches = []

    # 获取 xt 与 Wy 的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # 使用0来初始化“a”、“c”、“y”
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])

    # 初始化“a_next”、“c_next”
    a_next = a0
    c_next = np.zeros([n_a, m])

    # 遍历所有的时间步
    for t in range(T_x):
        # 更新下一个隐藏状态，下一个记忆状态，计算预测值，获取cache
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:,:,t], a_next, c_next, parameters)

        # 保存新的下一个隐藏状态到变量a中
        a[:, :, t] = a_next

        # 保存预测值到变量y中
        y[:, :, t] = yt_pred

        # 保存下一个单元状态到变量c中
        c[:, :, t] = c_next

        # 把cache添加到caches中
        caches.append(cache)

    # 保存反向传播需要的参数
    caches = (caches, x)

    return a, y, c, caches


def clip(gradients, maxValue):
    """
    使用maxValue来修剪梯度

    参数：
        gradients -- 字典类型，包含了以下参数："dWaa", "dWax", "dWya", "db", "dby"
        maxValue -- 阈值，把梯度值限制在[-maxValue, maxValue]内

    返回：
        gradients -- 修剪后的梯度
    """
    # 获取参数
    dWaa, dWax, dWya, db, dby = gradients['dWaa'], gradients['dWax'], gradients['dWya'], gradients['db'], gradients['dby']

    # 梯度修剪
    for gradient in [dWaa, dWax, dWya, db, dby]:
        np.clip(gradient, -maxValue, maxValue, out=gradient)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients


def sample(parameters, char_to_is, seed):
    """
    根据RNN输出的概率分布序列对字符序列进行采样

    参数：
        parameters -- 包含了Waa, Wax, Wya, by, b的字典
        char_to_ix -- 字符映射到索引的字典
        seed -- 随机种子

    返回：
        indices -- 包含采样字符索引的长度为n的列表。
    """

    # 从parameters 中获取参数
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]

    # 步骤1
    ## 创建独热向量x
    x = np.zeros((vocab_size,1))

    ## 使用0初始化a_prev
    a_prev = np.zeros((n_a,1))

    # 创建索引的空列表，这是包含要生成的字符的索引的列表。
    indices = []

    # IDX是检测换行符的标志，我们将其初始化为-1。
    idx = -1

    # 循环遍历时间步骤t。在每个时间步中，从概率分布中抽取一个字符，
    # 并将其索引附加到“indices”上，如果我们达到50个字符，
    #（我们应该不太可能有一个训练好的模型），我们将停止循环，这有助于调试并防止进入无限循环
    counter = 0
    newline_character = char_to_ix["\n"]

    while (idx != newline_character and counter < 50):
        # 步骤2：使用公式1、2、3进行前向传播
        a = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b)
        z = np.dot(Wya, a) + by
        y = cllm_utils.softmax(z)

        # 设定随机种子
        np.random.seed(counter + seed)

        # 步骤3：从概率分布y中抽取词汇表中字符的索引
        idx = np.random.choice(list(range(vocab_size)), p=y.ravel())

        # 添加到索引中
        indices.append(idx)

        # 步骤4:将输入字符重写为与采样索引对应的字符。
        x = np.zeros((vocab_size,1))
        x[idx] = 1

        # 更新a_prev为a
        a_prev = a

        # 累加器
        seed += 1
        counter +=1

    if(counter == 50):
        indices.append(char_to_ix["\n"])

    return indices


def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    执行训练模型的单步优化。

    参数：
        X -- 整数列表，其中每个整数映射到词汇表中的字符。
        Y -- 整数列表，与X完全相同，但向左移动了一个索引。
        a_prev -- 上一个隐藏状态
        parameters -- 字典，包含了以下参数：
                        Wax -- 权重矩阵乘以输入，维度为(n_a, n_x)
                        Waa -- 权重矩阵乘以隐藏状态，维度为(n_a, n_a)
                        Wya -- 隐藏状态与输出相关的权重矩阵，维度为(n_y, n_a)
                        b -- 偏置，维度为(n_a, 1)
                        by -- 隐藏状态与输出相关的权重偏置，维度为(n_y, 1)
        learning_rate -- 模型学习的速率

    返回：
        loss -- 损失函数的值（交叉熵损失）
        gradients -- 字典，包含了以下参数：
                        dWax -- 输入到隐藏的权值的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏到隐藏的权值的梯度，维度为(n_a, n_a)
                        dWya -- 隐藏到输出的权值的梯度，维度为(n_y, n_a)
                        db -- 偏置的梯度，维度为(n_a, 1)
                        dby -- 输出偏置向量的梯度，维度为(n_y, 1)
        a[len(X)-1] -- 最后的隐藏状态，维度为(n_a, 1)
    """

    # 前向传播
    loss, cache = cllm_utils.rnn_forward(X, Y, a_prev, parameters)

    # 反向传播
    gradients, a = cllm_utils.rnn_backward(X, Y, parameters, cache)

    # 梯度修剪，[-5 , 5]
    gradients = clip(gradients,5)

    # 更新参数
    parameters = cllm_utils.update_parameters(parameters,gradients,learning_rate)

    return loss, gradients, a[len(X)-1]


def model(data, ix_to_char, char_to_ix, num_iterations=3500,
          n_a=50, dino_names=7, vocab_size=27):
    """
    训练模型并生成恐龙名字

    参数：
        data -- 语料库
        ix_to_char -- 索引映射字符字典
        char_to_ix -- 字符映射索引字典
        num_iterations -- 迭代次数
        n_a -- RNN单元数量
        dino_names -- 每次迭代中采样的数量
        vocab_size -- 在文本中的唯一字符的数量

    返回：
        parameters -- 学习后了的参数
    """

    # 从vocab_size中获取n_x、n_y
    n_x, n_y = vocab_size, vocab_size

    # 初始化参数
    parameters = cllm_utils.initialize_parameters(n_a, n_x, n_y)

    # 初始化损失
    loss = cllm_utils.get_initial_loss(vocab_size, dino_names)

    # 构建恐龙名称列表
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]

    # 打乱全部的恐龙名称
    np.random.seed(0)
    np.random.shuffle(examples)

    # 初始化LSTM隐藏状态
    a_prev = np.zeros((n_a,1))

    # 循环
    for j in range(num_iterations):
        # 定义一个训练样本
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]

        # 执行单步优化：前向传播 -> 反向传播 -> 梯度修剪 -> 更新参数
        # 选择学习率为0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)

        # 使用延迟来保持损失平滑,这是为了加速训练。
        loss = cllm_utils.smooth(loss, curr_loss)

        # 每2000次迭代，通过sample()生成“\n”字符，检查模型是否学习正确
        if j % 2000 == 0:
            print("第" + str(j+1) + "次迭代，损失值为：" + str(loss))

            seed = 0
            for name in range(dino_names):
                # 采样
                sampled_indices = sample(parameters, char_to_ix, seed)
                cllm_utils.print_sample(sampled_indices, ix_to_char)

                # 为了得到相同的效果，随机种子+1
                seed += 1

            print("\n")
    return parameters


if __name__ == '__main__':
    # np.random.seed(1)
    # xt = np.random.randn(3, 10)
    # a_prev = np.random.randn(5, 10)
    # Waa = np.random.randn(5, 5)
    # Wax = np.random.randn(5, 3)
    # Wya = np.random.randn(2, 5)
    # ba = np.random.randn(5, 1)
    # by = np.random.randn(2, 1)
    # parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
    #
    # a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
    # print("a_next[4] = ", a_next[4])
    # print("a_next.shape = ", a_next.shape)
    # print("yt_pred[1] =", yt_pred[1])
    # print("yt_pred.shape = ", yt_pred.shape)


    # np.random.seed(1)
    # x = np.random.randn(3,10,4)
    # a0 = np.random.randn(5,10)
    # Waa = np.random.randn(5,5)
    # Wax = np.random.randn(5,3)
    # Wya = np.random.randn(2,5)
    # ba = np.random.randn(5,1)
    # by = np.random.randn(2,1)
    # parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
    #
    # a, y_pred, caches = rnn_forward(x, a0, parameters)
    # print("a[4][1] = ", a[4][1])
    # print("a.shape = ", a.shape)
    # print("y_pred[1][3] =", y_pred[1][3])
    # print("y_pred.shape = ", y_pred.shape)
    # print("caches[1][1][3] =", caches[1][1][3])
    # print("len(caches) = ", len(caches))

    # np.random.seed(1)
    # x = np.random.randn(3,10,7)
    # a0 = np.random.randn(5,10)
    # Wf = np.random.randn(5, 5+3)
    # bf = np.random.randn(5,1)
    # Wi = np.random.randn(5, 5+3)
    # bi = np.random.randn(5,1)
    # Wo = np.random.randn(5, 5+3)
    # bo = np.random.randn(5,1)
    # Wc = np.random.randn(5, 5+3)
    # bc = np.random.randn(5,1)
    # Wy = np.random.randn(2,5)
    # by = np.random.randn(2,1)
    #
    # parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
    #
    # a, y, c, caches = lstm_forward(x, a0, parameters)
    # print("a[4][3][6] = ", a[4][3][6])
    # print("a.shape = ", a.shape)
    # print("y[1][4][3] =", y[1][4][3])
    # print("y.shape = ", y.shape)
    # print("caches[1][1[1]] =", caches[1][1][1])
    # print("c[1][2][1]", c[1][2][1])
    # print("len(caches) = ", len(caches))

    # 获取名称
    data = open("dinos.txt", "r").read()

    # 转化为小写字符
    data = data.lower()

    # 转化为无序且不重复的元素列表
    chars = list(set(data))

    # 获取大小信息
    data_size, vocab_size = len(data), len(chars)

    print(chars)
    print("共计有%d个字符，唯一字符有%d个"%(data_size,vocab_size))

    char_to_ix = {ch:i for i, ch in enumerate(sorted(chars))}
    ix_to_char = {i:ch for i, ch in enumerate(sorted(chars))}

    print(char_to_ix)
    print(ix_to_char)


    #开始时间
    start_time = time.clock()

    #开始训练
    parameters = model(data, ix_to_char, char_to_ix, num_iterations=3500)

    #结束时间
    end_time = time.clock()

    #计算时差
    minium = end_time - start_time

    print("执行了：" + str(int(minium / 60)) + "分" + str(int(minium%60)) + "秒")
