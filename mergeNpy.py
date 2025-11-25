import numpy as np

# 加载图像和标签
images = np.load("jammingData/data.npy")
labels = np.load("jammingData/labels.npy")

# 筛选标签为 0~6 的样本
mask_0_6 = (labels >= 0) & (labels <= 5)
images_0_6 = images[mask_0_6]
labels_0_6 = labels[mask_0_6]

# 筛选标签为 7~9 的样本
mask_7_9 = (labels >= 6) & (labels <= 9)
images_7_9 = images[mask_7_9]
labels_7_9 = labels[mask_7_9]

np.save('ID_OODdata/IDimages_0_5.npy', images_0_6)
np.save('ID_OODdata/IDlabels_0_5.npy', labels_0_6)

np.save('ID_OODdata/OODimages_6_9.npy', images_7_9)
np.save('ID_OODdata/OODlabels_6_9.npy', labels_7_9)