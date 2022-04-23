
from sklearn.manifold import TSNE
import torch
import numpy as np
from classifier_model import Convolution
import matplotlib.pyplot as plt
from classifier_pre import tsn_x, tsn_y
import SNEplt
# 载入数据
model = Convolution()# 导入网络结构
model.load_state_dict(torch.load('model_good.pt')) # 导入网络的参数
model.eval()
data_all = tsn_x[:,np.newaxis,:,:].astype('float32')
with torch.no_grad():
    # tests = tests .type(torch.FloatTensor)
    data = torch.tensor(data_all)
    X = torch.tensor(model(data))
    Y = np.argmax(X, 1).numpy()
# 加载数据
def get_data():
	"""
	:return: 数据集、标签、样本数量、特征数量
	"""
	#digits = datasets.load_digits(n_class=10)
	digits=2
	data = X#digits.data		# 图片特征
	label = Y#digits.target		# 图片标签
	n_samples=X.shape[0]#对应reshape中的行数
	n_features =X.shape[1] #对应reshape中的列数
	return data, label, n_samples, n_features


# 对样本进行预处理并画图
def plot_embedding(data, label, title):
	"""
	:param data:数据集
	:param label:样本标签
	:param title:图像标题
	:return:图像
	"""
	x_min, x_max = np.min(data, 0), np.max(data, 0)
	data = (data - x_min) / (x_max - x_min)		# 对数据进行归一化处理
	fig = plt.figure()		# 创建图形实例
	ax = plt.subplot(111)		# 创建子图，经过验证111正合适，尽量不要修改
	# 遍历所有样本
	for i in range(data.shape[0]):
		# 在图中为每个数据点画出标签
		plt.text(data[i, 0], data[i, 1], str(label[i]), color=plt.cm.Set1(label[i] * 10),
				 fontdict={'weight': 'bold', 'size': 7})
	plt.xticks()		# 指定坐标的刻度
	plt.yticks()
	plt.title(title, fontsize=14)
	# 返回值
	return fig


# 主函数，执行t-SNE降维
data, label, n_samples, n_features = get_data()  # 调用函数，获取数据集信息
print('Starting compute t-SNE Embedding...')
ts = TSNE(n_components=2, init='pca', random_state=0)
# t-SNE降维
reslut = ts.fit_transform(data)
# 调用函数，绘制图像
fig1 = plot_embedding(reslut, label, 't-SNE Embedding of digits')
# 显示图像
plt.show()
da=tsn_x.reshape(152,-1)
dy=tsn_y.argmax(1)
SNEplt.plot_embedding(data=X, label=Y)
SNEplt.plot_embedding(data=da, label=dy)
#fig2 = plot_embedding(da, dy, 'Raw Data')
#plt.show()





