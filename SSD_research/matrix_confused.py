import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import torch
import torchvision
from classifier_pre import tsn_x, tsn_y
from classifier_pre import X_train, X_test, y_train, y_test
import numpy as np
import SNEplt
from classifier_model import Convolution

# confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams



#%%计算预测值
model = Convolution()# 导入网络结构
model.load_state_dict(torch.load('model_good.pt')) # 导入网络的参数
model.eval()
data_all = tsn_x[:,np.newaxis,:,:].astype('float32')

with torch.no_grad():
    # tests = tests .type(torch.FloatTensor)
    data = torch.tensor(data_all)
    predict = torch.tensor(model(data))
    y_predict = np.argmax(predict, 1).numpy()
y_test = tsn_y.argmax(1)



def TN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict==0))   # 注意这里是一个‘&’

TN(y_test, y_predict)   # 403

def FP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 0) & (y_predict==1))

FP(y_test, y_predict)   # 2

def FN(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict==0))

FN(y_test, y_predict)   # 9

def TP(y_true, y_predict):
    assert len(y_true) == len(y_predict)
    return np.sum((y_true == 1) & (y_predict==1))

TP(y_test, y_predict)   # 36

def confusion_matrix(y_true, y_predict):
    return np.array([
        [TN(y_true, y_predict), FP(y_true, y_predict)],
        [FN(y_true, y_predict), TP(y_true, y_predict)]
    ])

CM = confusion_matrix(y_test, y_predict)
print(CM)


def precision_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fp = FP(y_true, y_predict)
    try:
        return tp / (tp + fp)
    except:   # 处理分母为0的情况
        return 0.0

precision_score(y_test, y_predict)


def recall_score(y_true, y_predict):
    tp = TP(y_true, y_predict)
    fn = FN(y_true, y_predict)

    try:
        return tp / (tp + fn)
    except:
        return 0.0

recall_score(y_test, y_predict)



classes = ['true', 'predict']
# 输入特征矩阵CM
proportion = []
for i in CM:
        for j in i:
                temp = j / (np.sum(i))
                proportion.append(temp)
# print(np.sum(confusion_matrix[0]))
# print(proportion)
pshow = []
for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
proportion = np.array(proportion).reshape(2, 2)  # reshape(列的长度，行的长度)
pshow = np.array(pshow).reshape(2, 2)
# print(pshow)
config = {
        "font.family": 'Times New Roman',  # 设置字体类型
}
rcParams.update(config)
plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues)  # 按照像素显示出矩阵
# (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
# 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
plt.title('confusion_matrix')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, fontsize=12)
plt.yticks(tick_marks, classes, fontsize=12)

thresh = np.max(CM) / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器
iters = np.reshape([[[i, j] for j in range(2)] for i in range(2)], (CM.size, 2))
for i, j in iters:
        if (i == j):
                plt.text(j, i - 0.12, format(CM[i, j]), va='center', ha='center', fontsize=12,
                         color='white', weight=5)  # 显示对应的数字
                plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12, color='white')
        else:
                plt.text(j, i - 0.12, format(CM[i, j]), va='center', ha='center', fontsize=12)  # 显示对应的数字
                plt.text(j, i + 0.12, pshow[i, j], va='center', ha='center', fontsize=12)

plt.ylabel('True label', fontsize=16)
plt.xlabel('Predict label', fontsize=16)
plt.tight_layout()
plt.show()

