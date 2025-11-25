import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
data = np.load('imageSet_3/CW_20.npy')
print("数组形状 (shape):", data.shape)

# 提取第一张图
first_image = data[0]

# 判断是否是 RGB 图或灰度图
if first_image.ndim == 3 and first_image.shape[0] == 3:
    # 通道为 [3, H, W]
    channels = first_image.astype(np.uint8)

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    for i in range(3):
        axs[i].imshow(channels[i], cmap='gray')  # 单通道灰度图
        axs[i].axis('off')

    plt.tight_layout()
    plt.savefig('images/test.png', bbox_inches='tight', pad_inches=0)
    plt.show()

elif first_image.ndim == 2:
    plt.imshow(first_image, cmap='gray')
    plt.axis('off')
    plt.savefig('images/first_image_gray.png', bbox_inches='tight', pad_inches=0)
    plt.show()
else:
    raise ValueError(f"Unsupported image shape: {first_image.shape}")
