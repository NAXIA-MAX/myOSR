import numpy as np

# 加载 .npy 文件
labels = np.load('/workspace/OSR/jammingData/3channels/labels_3channel.npy')  
data=np.load("/workspace/OSR/ID_OODdata/OODimages_6_9.npy")
# 查看标签的唯一值
unique_labels, counts = np.unique(labels, return_counts=True)
print("dtype:", data.ndim)
print(data.dtype)
# 打印所有标签
print("标签总数：", len(unique_labels))
print("所有标签名称：", unique_labels)
print("每个标签对应的数据量：", dict(zip(unique_labels, counts)))
print(labels[1])
