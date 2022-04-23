# CDE同心分离集熵方法特征提取

熵值变化可以作为旋转机械的健康指标，目前常用的熵方法包括有：样本熵、模糊熵、置换熵、MSE：多尺度熵；MPE：多维置换熵；MFE：多维模糊熵；MDE：多维多样性熵；IMFs：基本模态分量。MDE优点：一致性好，计算效率高，健壮性强，但MDE实际上是基于Haar小波的低通滤波器，忽略了高频带的丰富信息，HDE：利用小波熵和高频分集来同步提取故障。通过文章阅读后了解到CDE方法——同心分集熵法，即把原信号利用重采样的方法分解为局部细化的时间序列LRTS，再对每一段时间序列采用不同小波基函数提取特征，并对不同时间序列不同小波提取出的频率特征进行特征融合，提高特征的故障分类能力。

![image](https://github.com/Code_transfer_learning/Machine_diagnostics/文章复现/CDE_flexible feature extraction/CDE.jpg)


# 论文链接
这里：https://doi.org/10.1016/j.ymssp.2022.108934


