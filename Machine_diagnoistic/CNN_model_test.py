#%% 导入模块
import os
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split

from model import MyCNN
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
import numpy as np
#%% 数据准备

path = 'D:\故障特征提取\python路径\CNN-特征分类\X_3000_2d_ss.mat'
file = io.loadmat(path)
data = file.get("array_3000_x")#可修改位置
X = data
y_l = np.zeros((data.shape[0], 1))
r_n = 196
for i in range(data.shape[0]):
    if i < r_n:
        y_l[i, 0] = 1

data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5),(0.5))
    ]
)

imgs = []
labels = []

for i in [1,56,18,4,45,72]:
    img = X[i]
    img = data_tf(img)
    imgs.append(img)
    labels.append(int(y_l[i]))
imgs = torch.stack(imgs, 0)

#%% 加载模型
model = MyCNN()
model.load_state_dict(torch.load('model_good.pt', map_location=torch.device('cpu')))
model.eval()
#%% 测试模型

with torch.no_grad():
    imgs = imgs.type(torch.FloatTensor)
    output = model(imgs)
#%% 打印结果
pred = output.argmax(1)
true = torch.LongTensor(labels)
print(pred)
print(true)
#%% 结果显示
plt.figure(figsize=(18, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(f'pred {pred[i]} | true {true[i]}')
    plt.axis('off')
    plt.imshow(imgs[i].squeeze(0), cmap='gray')
plt.savefig('test_good.png')
plt.show()

