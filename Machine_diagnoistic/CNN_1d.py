import time
import numpy as np
import torch.nn as nn
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


EPOCH = 30
BATCH_SIZE = 64
LR = 0.01

#%% tensorboard
writer = SummaryWriter('./logs/')

#%% 制作数据集

path = r'D:\故障特征提取\python路径\CNN-特征分类\X_1d_3000_raw.mat'
file = io.loadmat(path)
data_e = file.get("array_x_3000")#可修改位置
X = data_e
y_l = np.empty((0))
r_n = 3840
num = X.shape[0]
for i in range(X.shape[0]):
    if i < r_n:
        y_l = np.append(y_l, 1)
    else:
        y_l = np.append(y_l, 0)


y_l = y_l.reshape(num,1)
X_train, X_test, y_train, y_test = train_test_split(X, y_l, stratify=y_l, random_state=42)
tr_l = X_train.shape[0]
te_l = X_test.shape[0]

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

        return train,label  # 返回标签

    def __len__(self):
        if self.train:
            return tr_l
        else:
            return te_l


train_file = myDataset(train=True, transform=data_tf)
test_file = myDataset(train=False, transform=data_tf)


#%% 制作数据加载器
train_loader = DataLoader(
    dataset=train_file,
    batch_size=BATCH_SIZE,
    shuffle=True
)
test_loader = DataLoader(
    dataset=test_file,
    batch_size=BATCH_SIZE,
    shuffle=False
)

#%% 定义网络结构
class MyCNN(nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256,
                      kernel_size=3, stride=1),
            nn.BatchNorm1d(398),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(256, 512, 3, 1),
            nn.BatchNorm1d(196),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )

        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv3 = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=64,
                      kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm1d(97),
            nn.ReLU(),
        )

        # fully connected layer
        #self.fc = nn.Linear(400, 128)
        self.fc = nn.Linear(97, 128)

    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

#%% 创建模型
model = MyCNN()
#print(model)
optim = torch.optim.Adam(model.parameters(), LR)
lossf = nn.CrossEntropyLoss()

#%%loss及acc
def calc(data_loader):
    loss = 0
    total = 0
    correct = 0
    with torch.no_grad():
        for data, targets in data_loader:
            data = data
            data = data.type(torch.FloatTensor)
            targets = targets
            targets = targets.reshape(-1)
            output = model(data)
            loss += lossf(output, targets.long())
            correct += (output.argmax(1) == targets).sum()
            total += data.size(0)
    loss = loss.item()/len(data_loader)
    acc = correct.item()/total
    return loss, acc

#%% 训练过程打印函数
def show():
    # 定义全局变量
    if epoch == 0:
        global model_saved_list
        global temp
        temp = 0
    # 打印训练的EPOCH和STEP信息

    header_list = [
        f'EPOCH: {epoch+1:0>{len(str(EPOCH))}}/{EPOCH}',
        f'STEP: {step+1:0>{len(str(len(train_loader)))}}/{len(train_loader)}'
    ]
    header_show = ' '.join(header_list)
    print(header_show, end=' ')
    # 打印训练的LOSS和ACC信息
    loss, acc = calc(train_loader)
    writer.add_scalar('loss', loss, epoch+1)
    writer.add_scalar('acc', acc, epoch+1)

    train_list = [
        f'LOSS: {loss:.4f}',
        f'ACC: {acc:.4f}'
    ]
    train_show = ' '.join(train_list)
    #print(train_show, end=' ')
    # 打印测试的LOSS和ACC信息
    val_loss, val_acc = calc(test_loader)
    writer.add_scalar('val_loss', val_loss, epoch+1)
    writer.add_scalar('val_acc', val_acc, epoch+1)
    test_list = [
        f'VAL-LOSS: {val_loss:.4f}',
        f'VAL-ACC: {val_acc:.4f}'
    ]
    test_show = ' '.join(test_list)
    print(test_show, end=' ')
    # 保存最佳模型
    if val_acc > temp:
        model_saved_list = header_list+train_list+test_list
        torch.save(model.state_dict(), 'model_1d.pt')
        temp = val_acc

#%% 训练模型
for epoch in tqdm(range(EPOCH),desc = '训练进度',ncols=100):
    time.sleep(0.05)
    start_time = time.time()
    for step, (data, targets) in enumerate(train_loader):
        optim.zero_grad()
        data = data
        data = data.type(torch.FloatTensor)
        #data = data.cuda()
        targets = targets.reshape(-1)
        output = model(data)
        output = output.float()
        loss = lossf(output, targets.long())
        acc = (output.argmax(1) == targets).sum().item()/BATCH_SIZE
        loss.backward()
        optim.step()

    print(f'LOSS: {loss.item():.4f}',f'ACC: {acc:.4f}')
    show()
    end_time = time.time()
    print(f'TOTAL-TIME: {round(end_time-start_time)}')



#%% 打印并保存最优模型的信息
model_saved_show = ' '.join(model_saved_list)
print('| BEST-MODEL | '+model_saved_show)
with open('model_1d.txt', 'a') as f:
    f.write(model_saved_show+'\n')
#%% tensorboard
'''
tensorboard --logdir=python路径/CNN-特征分类/logs/ --port 9000
'''

