import os
import torch
import torchvision
from PIL import Image
from sklearn.model_selection import train_test_split

from classifier_model import Convolution
from data_test import X_test, y_test, test_file
from classifier_pre import train_file,test_file
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import scipy.io as io
import numpy as np



#%% 加载模型
model = Convolution()
model.load_state_dict(torch.load('model_good.pt', map_location=torch.device('cpu')))
model.eval()
#%% 测试模型

test_loader = DataLoader(
    dataset=test_file,
    batch_size=1,
    shuffle=True
)
preds = []
trues = []
datas = []
acc=0
for i in range(6):
    for step, (data, label) in enumerate(test_loader):
        data_1 = data.numpy()
        #test = X_test[i]
        #test = data_tf(data)
        #tests.append(data)
        #labels.append(label)
        with torch.no_grad():
            # tests = tests .type(torch.FloatTensor)
            data = data.type(torch.FloatTensor)
            output = model(data)

        datas.append(data_1.reshape(data.size(2), data.size(3)))
        pred = output.argmax(1)
        true = label.argmax(1)
        if pred == true:
            acc = acc+1
            acc_p = acc/((step+1)*(i+1))
        preds.append(pred)
        trues.append(true)
    #tests = torch.stack(tests, 0)


#%% 打印结果
print(preds)
print(trues)
print(' Prediction: '+str(float(acc_p*100))+'%')
#%% 结果显示
plt.figure(figsize=(18, 4))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.title(f'pred {preds[i]} | true {trues[i]}')
    plt.axis('off')
    plt.imshow(datas[i], cmap='gray')
plt.savefig('test_good_SSD.png')
plt.show()

