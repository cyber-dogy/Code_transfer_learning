import pandas as pd
import torch
import time
import torch.nn as nn
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from datetime import datetime, timedelta  # 用于计算时间
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter

#%% 定义常量

INPUT_SIZE = 10  # 定义输入的特征数
HIDDEN_SIZE = 1  # 定义一个LSTM单元有多少个神经元
BATCH_SIZE = 50  # batch
EPOCH = 100  # 学习次数
LR = 0.001  # 学习率
DROP_RATE = 0  # drop out概率
LAYERS = 2  # 有多少隐层，一个隐层一般放一个LSTM单元
MODEL = 'LSTM'  # 模型名字

#%% tensorboard
writer = SummaryWriter('./logs/')

#%% 定义一些常用函数
# 保存日志
# fname是要保存的位置，s是要保存的内容
def log(fname, s):
    f = open(fname, 'a')
    f.write(str(datetime.now()) + ': ' + s + '\n')
    f.close()

#%% 设置GPU
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
# 设置随机种子
#torch.manual_seed(0)

#%% 进行归一化，分割数据集
#%% 制作数据集

data_csv = pd.read_csv(r'D:\故障特征提取\python路径\RNN-特征分类\1.csv')
print(data_csv.shape)
data_csv = data_csv[0:100]

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

def create_dataset(dataset, look_back=5):
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
X_train = data_X[:train_size]
y_train = data_Y[:train_size]
X_test = data_X[train_size:]
y_test = data_Y[train_size:]

tr_l = len(X_train)
te_l = len(y_train)

data_tf = torch.tensor


class myDataset(Dataset):  # 这是一个Dataset子类
    def __init__(self, train=True, transform=None, target_transform=None):

        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data = X_train  # 特征向量集合,特征是2维表示一段文本
            self.train_label = y_train  # 标签是1维,表示文本类别
        else:
            self.test_data = X_test  # 特征向量集合,特征是2维表示一段文本
            self.test_label = y_test  # 标签是1维,表示文本类别

    def __getitem__(self, index):
        if self.train:
            train, label = self.train_data[index], self.train_label[index]
        else:
            train, label = self.test_data[index], self.test_label[index]
        if self.transform is not None:
            train = self.transform(train)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return train, label  # 返回标签

    def __len__(self):
        if self.train:
            return tr_l
        else:
            return te_l


train_data = myDataset(train=True, transform=data_tf)
test_data = myDataset(train=False, transform=data_tf)


print('train_data.train_data.size():', train_data.train_data.shape)    # 打印训练集特征的size
print('train_data.train_labels.size():', train_data.train_label.shape)    # 打印训练集标签的size


#%% 先归一化，在分割数据，为节约时间，只取了部分

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE,  shuffle=True)
#valid_loader = DataLoader(data_valid, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE,  shuffle=False)

#%% 定义LSTM网络的结构

class rnn(nn.Module):
    def __init__(self):
        super(rnn, self).__init__()
        self.rnn = nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=LAYERS,
            dropout=DROP_RATE,
            batch_first=True  # 如果为True，输入输出数据格式是(batch, seq_len, feature)
            # 为False，输入输出数据格式是(seq_len, batch, feature)，
        )
        self.hidden_out = nn.Linear(HIDDEN_SIZE, 10)  # 最后一个时序的输出接一个全连接层
        self.h_s = None
        self.h_c = None

    def forward(self, x):  # x是输入数据集
        r_out, (h_s, h_c) = self.rnn(x)  # 如果不导入h_s和h_c，默认每次都进行0初始化
        #  h_s和h_c表示每一个隐层的上一时间点输出值和输入细胞状态
        # h_s和h_c的格式均是(num_layers * num_directions, batch, HIDDEN_SIZE)
        # 如果是双向LSTM，num_directions是2，单向是1
        output = self.hidden_out(r_out)
        return output

model = rnn()  # 使用GPU或CPU
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # optimize all cnn parameters
lossf = nn.CrossEntropyLoss()  # 分类问题

#%% 定义学习率衰减点，训练到50%和75%时学习率缩小为原来的1/10
mult_step_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                           milestones=[EPOCH // 2, EPOCH // 4 * 3], gamma=0.1)



#%% 训练+验证
# 训练+验证
avg_acc=np.empty(())

min_valid_loss = np.inf
for epoch in tqdm(range(EPOCH), desc='训练进度', ncols=100):
    total_train_loss = []
    start_time = time.time()
    model.train()  # 进入训练模式
    for step, (data, labels) in enumerate(train_loader):
        var_x = Variable(data)
        var_y = Variable(labels)
        # 前向传播
        out = model(var_x)
        out =out.reshape(-1,1,1)
        loss = lossf(out, var_y.long())
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'LOSS: {loss.item():.4f}',)
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time - start_time)}')

    mult_step_scheduler.step()  # 学习率更新
    # 服务器一般用的世界时，需要加8个小时，可以视情况把加8小时去掉
    print(str(datetime.now() + timedelta(hours=8)) + ': ')


#%%预测器

model = model.eval()
print(model)
data_X = data_X.reshape(-1, 1, 20)
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





#%% tensorboard
'''
tensorboard --logdir=python路径/RNN-特征分类/logs/ --port 9000
'''

0