import pandas as pd
from matplotlib import pyplot as plt
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch
import os

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

EPOCH = 1000
#%%数据载入
data_csv = pd.read_csv(r'D:\故障特征提取\python路径\RNN-特征分类\1.csv')
print(data_csv.shape)
data_csv=data_csv[0:100]

# 首先我们进行预处理，将数据中 na 的数据去掉，然后将数据标准化到 0 ~ 1 之间。
data_csv = data_csv.dropna()

plt.plot(data_csv)
plt.show()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value
dataset = list(map(lambda x: x / scalar, dataset))

'''
接着我们进行数据集的创建，我们想通过前面几个月的流量来预测当月的流量，
比如我们希望通过前两个月的流量来预测当月的流量，我们可以将前两个月的流量
当做输入，当月的流量当做输出。同时我们需要将我们的数据集分为训练集和测试
集，通过测试集的效果来测试模型的性能，这里我们简单的将前面几年的数据作为
训练集，后面两年的数据作为测试集。
'''


def create_dataset(dataset, look_back=10):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)


# 创建好输入输出
data_X, data_Y = create_dataset(dataset)
print(data_X)
print(data_Y)
#print(data_X)

# 划分训练集和测试集，70% 作为训练集
train_size = int(len(data_X) * 0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]
#print(train_X)
'''
最后，我们需要将数据改变一下形状，因为 RNN 读入的数据维度是 
(seq, batch, feature)，所以要重新改变一下数据的维度，这里只有一个序列，
所以 batch 是 1，而输入的 feature 就是我们希望依据的几个月份，这里我们
定的是两个月份，所以 feature 就是 2.
'''

train_X = train_X.reshape(-1, 1, 10)
train_Y = train_Y.reshape(-1, 1, 1)
test_X = test_X.reshape(-1, 1, 10)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
test_x = torch.from_numpy(test_X)
#print(train_X)


#%%RNN网络构建

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(RNN, self).__init__()

        self.rnn = nn.LSTM(input_size,
                           hidden_size,
                           num_layers
                           )
        self.reg = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x, _ = self.rnn(x)
        s, b, h = x.shape  # (seq, batch, hidden)
        x = x.view(s * b, h)  # 转化为线性层的输入方式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x


#%%创建模型
model = RNN(10, 10, 1, 2)
lossf = nn.MSELoss() #定义交叉熵函数
optim = torch.optim.Adam(model.parameters(), lr=0.001) #采用Adam算法

#%%模型训练
for epoch in tqdm(range(EPOCH), desc='训练进度', ncols=100):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = model(var_x)
    loss = lossf(out, var_y)
    # 反向传播
    optim.zero_grad()
    loss.backward()
    optim.step()
    if (epoch + 1) % 100 == 0:
        print('Epoch:{}, Loss:{:.5f}'.format(epoch + 1, loss.item()))

# 训练完成之后，我们可以用训练好的模型去预测后面的结果
model = model.eval()
print(model)
data_X = data_X.reshape(-1, 1, 10)
data_X = torch.from_numpy(data_X)
var_data = Variable(data_X)
pred_test = model(var_data)  # 测试集的预测结果

# 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()

# 画出实际结果和预测的结果
plt.plot(pred_test, 'r', label='prediction')
plt.plot(dataset, 'b', label='real')
plt.legend(loc='best')
plt.show()
