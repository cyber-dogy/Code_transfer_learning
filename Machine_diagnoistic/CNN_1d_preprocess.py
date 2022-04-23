import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt
import scipy.io as scio


path = r'D:\故障特征提取\参考\总数据.mat'
file = io.loadmat(path)
data_raw = file.get("X_3000")#可修改位置
group = data_raw.shape[0]
a = data_raw.shape[1]
print(a)
right = 6
X_mtf = np.empty((0, 400))
for i in range(int(a/400)):
    data_e = data_raw[:, i*400:400+i*400]
    for k in range(group):
        data = data_e[k, :].reshape(1, 400)
        X_mtf = np.append(X_mtf, data, axis=0)


scio.savemat('X_1d_3000_raw.mat', {'array_x_3000': X_mtf})
