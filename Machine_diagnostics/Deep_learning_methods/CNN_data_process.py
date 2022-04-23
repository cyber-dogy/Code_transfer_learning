import numpy as np
import scipy.io as io
from matplotlib import pyplot as plt
from pyts.image import GASF,MTF
import scipy.io as scio


path = r'D:\故障特征提取\python路径\data_X_4600_CDA.mat'
file = io.loadmat(path)
data_e = file.get("array_x_4600")#可修改位置
X_mtf = np.empty((18, 0, 20, 20))
for i in range(int(data_e.shape[1]/400)):
    print(i)
    data = data_e[:, 400*i:400+400*i]
    # MTF transformation
    mtf = GASF(image_size=20)
    x_mtf = mtf.fit_transform(data)
    x_mtf = x_mtf.reshape(18, 1, 20, 20)
    X_mtf = np.append(X_mtf[:][:], x_mtf, axis=1)





scio.savemat('X_4600_3d.mat', {'array_x_4600': X_mtf})

# Show the image for the first time series
plt.figure(figsize=(5, 5))
plt.imshow(X_mtf[0][0], cmap='rainbow', origin='lower')
plt.title('Markov Transition Field', fontsize=18)
plt.colorbar(fraction=0.0457, pad=0.04)
plt.tight_layout()
plt.show()
