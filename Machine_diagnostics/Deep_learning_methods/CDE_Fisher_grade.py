import os
import numpy as np
import scipy.io as io
import scipy.io as scio

path = r'D:\故障特征提取\python路径\data_x_4600_CDE.mat'
data = scio.loadmat(path)['array_x_4600']
data = np.array(data)
print(data.shape)

normnum = 6
FS = np.empty((0, 0))
for i in range(10):
    feature_1 = data[0:normnum, 1280 * i :1280 * (i + 1)]
    feature_2 = data[normnum:data.shape[0], 1280 * i :1280 * (i + 1)]
    m1 = np.mean(feature_1, axis=1)
    m2 = np.mean(feature_2, axis=1)
    m_n = np.mean(m1)
    m_f = np.mean(m2)
    Sb1 = (m1-m_n)*(m1-m_n).T
    Sb2 = (m2-m_f)*(m2-m_f).T
    Sw = (m_n-m_f)*(m_n-m_f).T
    Sb = np.zeros(())
    for k in range(len(Sb1)):
        Sb = Sb + Sb1[k]
    for k in range(len(Sb2)):
        Sb = Sb + Sb2[k]
    Sb = Sb/(len(Sb1)+len(Sb2))
    Fs = Sb/Sw
    print(Fs)
    FS = np.append(FS, Fs)








