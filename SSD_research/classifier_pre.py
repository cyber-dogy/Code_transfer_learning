import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torchvision
from sklearn.model_selection import train_test_split
import random


path = "D:\SSD故障预警\MC1_data"
col = ['S9','S12','S170', 'S173','S174','S188','S194','S196', 'S198']#'y171','y172','y183','y184','y194','y195','y197','y199','y206'
data_files = os.listdir(path)
pre_imp = np.zeros((len(col), 0))

#%% 制作数据集
data_health_all = np.empty((0,100,len(col)))
data_damage_all = np.empty((0,100,len(col)))
for fname in data_files:
    file_path = os.path.join(path, fname)
    data_r = pd.read_csv(file_path)
    data = np.zeros((len(data_r), 0))
    for i in range(len(col)):
        data_col = data_r[col[i]].fillna(method='pad')
        data_p = np.array(data_col).reshape(-1, 1)
        data = np.append(data, data_p, axis=1)
    l_d = data.shape[0]
    rad_num1 = np.random.randint(0, int(0.85*l_d) - 100)
    rad_num2 = np.random.randint(int(0.85 * l_d), l_d-100)
    data_health = data[rad_num1:(rad_num1+100), :]
    data_damage = data[rad_num2:(rad_num2+100), :]
    data_health_all = np.append(data_health_all, data_health.reshape(1, data_health.shape[0], data_health.shape[1]), axis=0)
    data_damage_all = np.append(data_damage_all, data_damage.reshape(1, data_damage.shape[0], data_damage.shape[1]), axis=0)


y = np.zeros((data_health_all.shape[0]+data_damage_all.shape[0], 2))
for i in range(data_health_all.shape[0]):
    y[i, 0] = 1
    y[i+data_health_all.shape[0], 1] = 1
label_health = y[:int(data_health_all.shape[0]), :]
label_damage = y[int(data_damage_all.shape[0]):, :]

data_all = np.append(data_health_all, data_damage_all, axis=0)
label_all = np.append(label_health, label_damage, axis=0)


X_train, X_test, y_train, y_test = train_test_split(data_all, label_all, stratify=label_all, random_state=42)
tsn_x = np.append(X_train,X_test,axis=0)
tsn_y = np.append(y_train,y_test,axis=0)
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


