import time

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


import scipy.io as io
from sklearn.model_selection import train_test_split

path = "D:\SSD故障预警\MC1_data"
save_p = "D:\SSD故障预警\\random forest选优MC1"
col = ['y1', 'y5', 'y9','y12','y170', 'y173','y174','y180', 'y187','y188','y194','y196', 'y198',]#'y171','y172','y183','y184','y195','y197','y199','y206'
data_files = os.listdir(path)
pre_imp = np.zeros((len(col), 0))

for fname in data_files:
    file_path = os.path.join(path, fname)
    data_r = pd.read_csv(file_path)
    data = np.zeros((len(data_r), 0))
    for i in range(len(col)):
        data_col = data_r[col[i]].fillna(method='pad')
        data_p = np.array(data_col).reshape(-1, 1)
        data = np.append(data, data_p, axis=1)
    l_d = data.shape[0]
    data_health = data[:int(0.7*l_d), :]
    data_damage = data[int(0.7*l_d):, :]
    y = np.zeros((data.shape[0], 1))
    for i in range(int(0.7*l_d)):
        y[i, 0] = 1

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
    forest_importances = pd.Series(importances, index=col)
    pre_im = np.array(forest_importances).reshape(-1,1)
    pre_imp = np.append(pre_imp, pre_im, axis=1)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    #plt.show()
    save_pp = os.path.join(save_p, str(fname) + '_Feature importances.png')
    f = plt.gcf()
    f.savefig(save_pp)
    f.clear()
    # print(forest.predict([[0, 0, 0, 0]]))
pre_importance = np.mean(pre_imp, axis=1)
for k in range(len(col)):
    print(col[k])
    print(pre_importance[k])
plt.bar(range(len(col)), pre_importance,align="center",tick_label=col,color="b",alpha=0.6)
fig.tight_layout()
plt.show()



