import torch
import torchvision

from classifier_pre import X_train, X_test, y_train, y_test
import numpy as np
import SNEplt
from classifier_model import Convolution

x_train = X_train
x_test = X_test[:int(0.5*X_test.shape[0])]
x_valid = X_test[int(0.5*X_test.shape[0]):]
y_valid = y_test[int(0.5*X_test.shape[0]):]
data_tf = torchvision.transforms.Compose(
    [
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.5),(0.5))
    ]
)

# 原始数据可视化
label_1 = np.argmax(y_valid, 1)
SNEplt.plot_embedding(data=x_valid.reshape(19,-1), label=label_1)
# 将网络最后一层的数据单独可视化
#x_train, x_valid, x_test = x_train[:, :, np.newaxis], x_valid[:, :, np.newaxis], x_test[:, :, np.newaxis]

model = Convolution()# 导入网络结构
model.load_state_dict(torch.load('model_good.pt')) # 导入网络的参数
model.eval()
x_valid = x_valid[:,np.newaxis,:,:].astype('float32')
with torch.no_grad():
    # tests = tests .type(torch.FloatTensor)
    data = torch.tensor(x_valid)
    y = torch.tensor(model(data))
#model_last_layer = Model(inputs=model.inputs, outputs=model.layers[13].output)
#y = model_last_layer.predict(x_valid)



label = np.argmax(y, 1)
label = label.numpy()
SNEplt.plot_embedding(data=y, label=label)
SNEplt.plot_data(data=y)
