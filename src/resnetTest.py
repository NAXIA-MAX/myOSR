import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import resnet
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader,Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

filenameList = ["normal","CW","SCW(0.1023)","SCW(1.023)","AGWN(0.1023)","AGWN(1.023)",
                "PI(1ms_0.3)","PI(1ms_0.7)","PI(10ms_0.3)","PI(10ms_0.7)"]
#定义一个继承类，继承Dataset
class NPYDatasetWithAugment(Dataset):
    def __init__(self, data_array, label_array, transform=None):
        self.data = data_array
        self.labels = label_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]  # shape: [H, W] or [1, H, W]
        label = self.labels[idx]
        # 保证 image 是 [H, W]
        if image.ndim == 3 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)  # [1, H, W] → [H, W]
        image_pil =Image.fromarray(image) 
        image_pil = image_pil.convert('L')  # 单通道灰度图

        if self.transform:
            image_tensor = self.transform(image_pil)  
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        return image_tensor, label
    
test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

def remodConv(model,inputChannel):
    # 获取原始 conv1 的参数
    conv1 = model.conv1
    out_channels = conv1.out_channels
    kernel_size = conv1.kernel_size
    stride = conv1.stride
    padding = conv1.padding
    bias = (conv1.bias is not None) 

    # 创建新的 conv1 层
    model.conv1 = nn.Conv2d(inputChannel,
                            out_channels,
                            kernel_size=kernel_size,
                            stride=stride,
                            padding=padding,
                            bias=bias)
    
testData=np.load("jammingData/X_test.npy")
testLabels=np.load("jammingData/y_test.npy")
test_dataset = NPYDatasetWithAugment(testData, testLabels, transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

model_name="resnet20" #模型名称
numClasses=10         #训练种类数
inputChannel=1
models=resnet.build_resnet(model_name,numClasses)
#如果输入的通道数不为3，则进行调整
if inputChannel!=3:
    remodConv(models,inputChannel)
models.load_state_dict(torch.load('resnet20_best.pth'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models.to(device)
models.eval()
#验证
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs = models(inputs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
correct = sum(p == t for p, t in zip(all_preds, all_labels))
total = len(all_labels)
print(f"验证集准确率: {correct / total * 100:.2f}%")

# 计算混淆矩阵
cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=filenameList)

# 绘图
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(ax=ax, cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png", bbox_inches='tight')
plt.show()