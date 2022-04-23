import math
import numpy as np
import scipy.io as io
import random
from scipy.fftpack import fft
import matplotlib.pyplot as plt
import pywt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
import scipy.io as scio

def split(m, step, length):
    n = random.randint(0, len(exp_1))
    data_s = np.empty((0, length))
    ###########################################
    if n > (len(exp_1)-(length+m*step)):
        n = n-(len(exp_1)-(length+m*step))
    for i in range(m):
        data_z = exp_1[n:(n+length)].reshape(1, length)
        data_s = np.append(data_s, data_z, axis=0)
        n = n+step
    print('当前数据切分' + str(m) + '组,范围：  ' + str(n) + '——' + str(n+length))
    return data_s
    ##########################################
    #这里存在问题，回头掉过头来看，数据切分逻辑定义的不完全对#

def draw(x1, y1, x2, y2):
    fig = plt.figure(figsize=(10, 5), dpi=150)
    #dpi调整清晰度，大于300后程序会卡死，默认80
    fig_1 = fig.add_subplot(1, 2, 1)
    fig_2 = fig.add_subplot(1, 2, 2)
    fig_1.plot(x1, y1)
    plt.xlabel('t/s')
    plt.ylabel('Amplitude')
    fig_2.plot(x2, y2)
    plt.xlabel('Hz')
    plt.ylabel('Amplitude')
    plt.show()
    return

def fftdef(data_s, Fs):
    length = data_s.shape[1]
    t = np.empty((0, length))
    X = np.empty((0, length))
    fs = np.empty((0, int(length/2)))
    Y = np.empty((0, int(length/2)))
    for i in range(data_s.shape[0]):
        fft_data = fft(data_s[i, :])
        fft_data = np.abs(fft_data)
        tt = np.arange(0, length / Fs, 1 / Fs)
        XX = data_s[i, :]
        fss = np.arange(0, Fs / 2, Fs / length)
        YY = fft_data[range(int(length / 2))]
        t = np.append(t, tt.reshape(1, length), axis=0)
        X = np.append(X, XX.reshape(1, length), axis=0)
        fs = np.append(fs, fss.reshape(1, int(length/2)), axis=0)
        Y = np.append(Y, YY.reshape(1, int(length/2)), axis=0)
        ####################
        if i > 20:
            break
        ####################
        #当组数过多时，防止内存不够卡死#
    return t, X, fs, Y

def waveletdef(data_s, Fs, lev_num):
    length = data_s.shape[1]
    level = 6
    t_t = np.empty((0, length))
    X_t = np.empty((0, length))
    t_w = np.empty((0, length))
    X_w = np.empty((0, length))
    f_w = np.empty((0, length))
    Y_w = np.empty((0, length))
    wlt_num = 0
    #lev_num = 0
    for i in range(data_s.shape[0]):
        XX = data_s[i, :]
        wp = pywt.WaveletPacket(data=XX, wavelet=wlt[wlt_num], mode='symmetric', maxlevel=level)
        wlt_num = wlt_num
        ##############################
        if wlt_num >= 7:
            wlt_num = wlt_num-7
        ##############################
        #控制小波变化
        level_f = lev[lev_num]
        data_wp = wp[level_f].data
        #'aaaaa'代表第6层输出的逼近信号;'aaaad'代表第6层细节信号
        str_len = len(level_f)+1
        X_ww = data_wp
        Y_ww = fft(X_ww)
        Y_ww = np.abs(Y_ww)
        Y_ww = Y_ww[range(int(len(Y_ww) / 2))]
        f_start = int(Fs / (2 ** str_len)*lev_num)
        f_end = Fs / (2 ** str_len)+f_start
        f_ww = np.arange(f_start, f_end, int((f_end-f_start)/len(Y_ww)))
        #############################
        f_w = f_w[:, 0:len(Y_ww)]
        f_w = np.append(f_w, f_ww.reshape(1, len(Y_ww)), axis=0)
        Y_w = Y_w[:, 0:len(Y_ww)]
        Y_w = np.append(Y_w, Y_ww.reshape(1, len(Y_ww)), axis=0)
        t_ww = np.arange(0, length / Fs, (length/Fs)/len(data_wp))
        t_tt = np.arange(0, length / Fs, 1 / Fs)
        t_t = np.append(t_t, t_tt.reshape(1, length), axis=0)
        X_t = np.append(X_t, XX.reshape(1, length), axis=0)
        ###########################
        #处理小波变化时序组
        ###########################
        t_w = t_w[:, 0:len(data_wp)]
        t_w = np.append(t_w, t_ww.reshape(1, len(data_wp)), axis=0)
        X_w = X_w[:, 0:len(data_wp)]
        X_w = np.append(X_w, X_ww.reshape(1, len(data_wp)), axis=0)
        ##########################
        #处理小波变换时频组
    print('分解层数' + str(str_len) + ';  频带范围:' + str(f_start) + '——' + str(f_end) + 'Hz')
    print(Y_w.shape)
    return t_t, X_t, t_w, X_w, f_w, Y_w

def DE_extraction(data, deta, y_l):
    DE = np.empty((1,0))
    for j in range(data.shape[0]):
        data_ds = data[j, :]
        y_i = np.empty((0, y_l))
        for k in range(len(data_ds)):
            y_ii = data_ds[k:k+y_l]
            y_ii = y_ii.reshape(1,len(y_ii))
            y_i = np.append(y_i, y_ii, axis=0)
            if k ==len(data_ds)-y_l:
                break
        Dm = np.empty((1, 0))
        for m in range(y_i.shape[0]):
            a=y_i[m, :]
            b=y_i[m+1, :]
            a = a.reshape(1,len(a))
            b = b.reshape(1,len(b))
            c=np.sum(a*b)
            d=math.sqrt(np.dot(a,a.T))*(math.sqrt(np.dot(b,b.T)))
            dm = c/d
            Dm = np.append(Dm, dm)
            if m ==y_i.shape[0]-2:
                break
        deta_ll = -1
        deta_l = np.empty((1, 0))
        for n in range(deta):
            deta_ll = deta_ll+2/deta
            deta_l = np.append(deta_l, deta_ll)
        P = np.zeros((1, deta))
        Dm = Dm.reshape(1, len(Dm))
        deta_l = deta_l.reshape(1, len(deta_l))
        for o in range(deta_l.shape[1]):
            for p in range(Dm.shape[1]):
                if Dm[0, p] <= deta_l[0, o]:
                    P[0, o] = P[0, o]+1
        Pk = P/np.sum(P)
        P_S = 0
        for q in range(Pk.shape[1]):
            if Pk[0,q] != 0:
                P_s = Pk[0, q] * (math.log(Pk[0, q]))
                P_S = P_S + P_s

        De = (-1/math.log(deta))*P_S
        DE = np.append(DE,De)

    return DE



def flatten(data_df):
    l = data_df.shape[1]
    r = data_df.shape[0]
    data_f = data_df.reshape(1, l*r)
    return data_f


path= r'D:\故障特征提取\总数据.mat'#可修改位置
file = io.loadmat(path)
data = file.get("X_4600")#可修改位置
wlt = ['haar', 'sym2', 'coif1', 'bior1.3', 'rbio1.3']
lev = ['aaa', 'aad', 'ada', 'add', 'daa', 'dad', 'dda', 'ddd']

group = 10#可修改位置
step = 1000#可修改位置
length_data = 2560#可修改位置
Fs = 25600#可修改位置
data_x_3000 = np.empty((int(Fs / 2), 0))
label_x_3000 = np.empty((0, 0))

# num=1
# 等待替换成大循环处理整篇数据
DE_i = np.empty((0, group))
for num in range(data.shape[0]):
    exp_1 = data[num, :]
    # 取第一次试验数据进行切分
    N = len(exp_1)
    # 主程序部分
    #############################################
    # 定义输入参数;group-实验组；step-组内重采样步长；length-单次采样数据长度；Fs-采样频率
    data_s = split(group, step, length_data)
    Y_exp_1 = np.empty((group, 0))
    for lev_num in range(len(lev)):
        t_t, X_t, t_w, X_w, f_w, Y_w = waveletdef(data_s, Fs, lev_num)
        Y_exp_1 = np.append(Y_exp_1, Y_w, axis=1)
    DE_ii = DE_extraction(Y_exp_1, deta=30, y_l=Y_exp_1.shape[1]-20)
    DE_ii = DE_ii.reshape(1, group)
    DE_i = np.append(DE_i, DE_ii, axis=0)
    data_exp_1_f = flatten(Y_exp_1)
    # 展平小波变换后序列
    data_x_3000 = np.append(data_x_3000, data_exp_1_f.T, axis=1)
    label_x = 0
    if num >= 3:
        label_x = 1 + label_x
    label_x_3000 = np.append(label_x_3000, label_x)
data_x_3000 = data_x_3000.T
#label_x_3000 = label_x_3000.T
print('X_3000处理后同心分离小波变换试验数据' + str(data_x_3000.shape))
#scio.savemat('data_X_CDA.mat', {'array_x': data_x_3000})#可修改位置
#scio.savemat('data_X_4600_CDE.mat', {'array_x_4600': DE_i})

plt.clf()
plt.figure(1, figsize=(8, 6))
X_3000_ext = PCA(n_components=2).fit_transform(data_x_3000)
#x_min, x_max = X_3000_ext[:, 0].min() - 0.5, X_3000_ext[:, 0].max() + 0.5
#y_min, y_max = X_3000_ext[:, 1].min() - 0.5, X_3000_ext[:, 1].max() + 0.5
plt.scatter(X_3000_ext[:, 0], X_3000_ext[:, 1], c=label_x_3000, cmap=plt.cm.Set1, edgecolor="k")

#plt.xlim(x_min, x_max)
#plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()

fig = plt.figure(2, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
X_3000_ext = PCA(n_components=2).fit_transform(DE_i)
ax.scatter(
    X_3000_ext[:, 0],
    X_3000_ext[:, 1],
    c=label_x_3000,
    cmap=plt.cm.Set1,
    edgecolor="k",
    s=40,
)
ax.set_title("PCA directions")
ax.set_xlabel("1st ")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd ")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd ")
ax.w_zaxis.set_ticklabels([])

plt.show()
#draw(t_t[1, :], X_t[1, :], t_w[1, 5:t_w.shape[1]-5], X_w[1, 5:t_w.shape[1]-5])
draw(t_t[1, :], X_t[1, :], f_w[1, :], Y_w[1, :])
#t,X,fs,Y=fftdef(data_s, Fs)
#傅里叶变换输出参数
#print(t.shape)
#draw(t[1,:],X[1,:],fs[1,:],Y[1,:])




