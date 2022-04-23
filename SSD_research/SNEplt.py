from sklearn.manifold import TSNE
from classifier_pre import X_train, X_test, y_train, y_test
import numpy as np
from time import time
import matplotlib.pyplot as plt


def plot_embedding(data, label,):
    """
    :param data: 待处理的高维数据，np.array的形式
    :param label: 数据的标签，整数标签。不是one—hot编码！！！
    :param title: 图标的，名称
    :return:
    """

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    for i in range(result.shape[0]):
        plt.text(result[i, 0], result[i, 1], str(label[i]),
                 color=plt.cm.Set1(label[i] * 10.),
                 fontdict={'weight': 'bold', 'size': 10})
    title = 't-SNE embedding of the digits (time %.2fs)' % (time() - t0)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return
def plot_data(data,):
    """
    :param data: 待处理的高维数据，np.array的形式
    :param label: 数据的标签，整数标签。不是one—hot编码！！！
    :param title: 图标的，名称
    :return:
    """

    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    x_min, x_max = np.min(result, 0), np.max(result, 0)
    result = (result - x_min) / (x_max - x_min)
    fig = plt.figure()
    ax = plt.subplot(111)
    label=range(result.shape[0])
    for i in range(result.shape[0]):
        plt.text(result[i, 0], result[i, 1],str(label[i]),
                 color=plt.cm.Set1(label[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 10})
    title = 't-SNE embedding of the digits (time %.2fs)' % (time() - t0)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.show()
    return
