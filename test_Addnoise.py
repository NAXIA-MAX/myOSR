import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
from resnetTrain import NPYDatasetWithAugment
# 自定义高斯噪声类
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.1):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std + self.mean
        return torch.clamp(tensor + noise, 0., 1.)  # 保证范围在[0,1]

    def __repr__(self):
        return self.__class__.__name__ + f'(mean={self.mean}, std={self.std})'


# 转换流程（无噪声 / 有噪声）
transform_clean = transforms.Compose([
    transforms.ToTensor()
])

transform_noisy = transforms.Compose([
    transforms.ToTensor(),
    AddGaussianNoise(mean=0.0, std=0.03)
])

savepthHead="trainedPth/"
data=np.load("/workspace/OSR/imageData/SCW(1.023)_50.npy")
labels=np.load("ID_OODdata/IDlabels_0_5.npy")
cleanData=NPYDatasetWithAugment(data, labels, transform=None)
train_dataset = NPYDatasetWithAugment(data, labels, transform=transform_noisy)

# 随便选一张图像
img_clean, _ = cleanData[0]
img_noisy, _ = train_dataset[0]

# 展示图像（灰度图）
def show_image(img_tensor, title):
    img = img_tensor.squeeze().numpy()
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.axis('off')

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
show_image(img_clean, "Clean Image")
plt.subplot(1, 2, 2)
show_image(img_noisy, "Noisy Image")
plt.savefig("mnist", dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()
