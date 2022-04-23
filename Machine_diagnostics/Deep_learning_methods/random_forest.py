import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier


import scipy.io as io
from sklearn.model_selection import train_test_split

path = r'D:\故障特征提取\python路径\data_x_3000_CDE.mat'
file = io.loadmat(path)
data = file.get("array_x_3000")#可修改位置

data1 = np.array(data)
data = data1[:, 0:5]
y = np.zeros((data.shape[0], 1))
r_n = 6
for i in range(data.shape[0]):
    if i < r_n:
        y[i, 0] = 1
print(y)
X = data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
forest = RandomForestClassifier(n_estimators=13, criterion="entropy",
                             max_depth=5, random_state=0)
forest.fit(X_train, y_train)
start_time = time.time()
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
elapsed_time = time.time() - start_time
print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
wlt = ['haar', 'sym2', 'coif1', 'bior1.3', 'rbio1.3']
forest_importances = pd.Series(importances, index=wlt)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=std, ax=ax)
ax.set_title("Feature importances using MDI")
ax.set_ylabel("Mean decrease in impurity")
fig.tight_layout()
plt.show()
#print(forest.predict([[0, 0, 0, 0]]))