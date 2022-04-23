import os
import pandas as pd
import numpy as np
import scipy.io as scio
from torch.utils.data import Dataset
import torchvision
from sklearn.model_selection import train_test_split
import numpy as np
import random

from tqdm import tqdm

random.seed(0)

np.random.seed(0)



path_norm = "D:\signal_deal\signal\西储大学轴承数据中心网站\\Normal Baseline Data\\97.mat"
file_norm = scio.loadmat(path_norm)
data_norm = file_norm['X097_DE_time']
data_norm = np.array(data_norm)
path_fault = "D:\signal_deal\signal\西储大学轴承数据中心网站\\12k Drive End Bearing Fault Data\\105.mat"
file_fault = scio.loadmat(path_fault)
data_fault = file_fault['X105_DE_time']
data_fault = np.array(data_fault)


#%% 制作数据集

data_1s = np.empty((0, 100, 9))
data_2s = np.empty((0, 100, 9))
for z in range(76):
    data_n = np.empty((100, 0))
    data_f = np.empty((100, 0))
    for i in range(9):
        st1 = np.random.randint(0, data_norm.shape[0] - 750)
        st2 = np.random.randint(0, data_fault.shape[0] - 750)
        data_ncol = data_norm[st1:(100 + st1), 0].reshape(-1, 1)
        data_fcol = data_fault[st2:(100 + st2), 0].reshape(-1, 1)
        data_n = np.append(data_n, data_ncol, axis=1)
        data_f = np.append(data_f, data_fcol, axis=1)
    data_1s = np.append(data_1s, data_n.reshape(1, 100, -1),axis=0)
    data_2s = np.append(data_2s, data_f.reshape(1, 100, -1),axis=0)
data_all = np.append(data_1s, data_2s, axis=0)




y = np.zeros((data_all.shape[0], 2))
for i in range(data_1s.shape[0]):
    y[i, 0] = 1
    y[i+data_1s.shape[0], 1] = 1

X_train, X_test, y_train, y_test = train_test_split(data_all, y, test_size=0.4, stratify=y, random_state=42)
tr_l = X_train.shape[0]
te_l = X_test.shape[0]

data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5),(0.5))
    ]
)


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


train_file = myDataset(train=True, transform=data_tf)
test_file = myDataset(train=False, transform=data_tf)
