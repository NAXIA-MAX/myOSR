import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
import resnet
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.optim as optim
import random,os
from sklearn.model_selection import train_test_split
""""
标签对应：
filenameList=["normal","CW","SCW(0.1023)","SCW(1.023)","AGWN(0.1023)","AGWN(1.023)",
              "PI(1ms_0.3)","PI(1ms_0.7)","PI(10ms_0.3)","PI(10ms_0.7)"]
"""
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
    
def setTransform(useAugment=False,useNoise=False,noisemean=0.0,noiseStd=0.1):
    transform_list = []
    if useAugment:
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomRotation(15))
    transform_list.append(transforms.ToTensor())

    if useAugment and useNoise:
        transform_list.append(AddGaussianNoise(mean=noisemean, std=noiseStd))

    transform_list.append(transforms.Normalize(mean=[0.5], std=[0.5]))
    return transforms.Compose(transform_list)

def creatDataLoaderNPY(trainData,testData,trainLabels,testLabels,train_transform,test_transform):
    #数据预处理
    train_dataset = NPYDatasetWithAugment(trainData, trainLabels, transform=train_transform)
    test_dataset = NPYDatasetWithAugment(testData, testLabels, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    return train_loader,test_loader

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

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    set_seed()

    model_name = "resnet20"#模型名称
    numClasses = 6        #训练种类数
    inputChannel = 1      #输入通道数
    useNoiseAug=True      #使用噪声增强
    NoiseStd=0.1
    savepthHead="trainedPth/"
    data=np.load("ID_OODdata/IDimages_0_5.npy")
    labels=np.load("ID_OODdata/IDlabels_0_5.npy")
    # 划分训练测试集
    trainData, testData, trainLabels, testLabels = train_test_split(
        data,labels , test_size=0.2, random_state=42, stratify=labels
    )
    train_transform=setTransform(useAugment=True,useNoise=useNoiseAug,NoiseStd=NoiseStd)
    test_transform=setTransform(useAugment=False)
    train_loader, test_loader = creatDataLoaderNPY(trainData, testData, trainLabels, testLabels,train_transform,test_transform)
    models = resnet.build_resnet(model_name, numClasses)

    if inputChannel != 3:
        remodConv(models, inputChannel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(models.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    epochs = 50
    best_acc = 0.0
    if useNoiseAug:
        withNoise=f"useNoiseAug_{NoiseStd}"
    else :
        withNoise="None"
    save_path = f"{savepthHead}{model_name}_best_{numClasses}_{withNoise}.pth"
    if not os.path.exists(savepthHead):
        raise FileNotFoundError(f"路径不存在: {savepthHead}")
    for epoch in range(epochs):
        models.train()
        print(f"Epoch {epoch+1} training mode: {models.training}")

        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = models(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if (i + 1) % 100 == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, 当前批次训练损失: {loss.item():.4f}")

        epoch_loss_train = running_loss / total_train
        epoch_acc_train = correct_train / total_train

        models.eval()
        print(f"Epoch {epoch+1} evaluation mode: {not models.training}")

        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = models(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_loss_val = val_running_loss / total_val
        epoch_acc_val = correct_val / total_val

        val_losses.append(epoch_loss_val)
        val_accuracies.append(epoch_acc_val)
        train_losses.append(epoch_loss_train)
        train_accuracies.append(epoch_acc_train)

        # 学习率调度器更新
        scheduler.step(epoch_loss_val)

        # 打印学习率
        for param_group in optimizer.param_groups:
            print(f"Current learning rate: {param_group['lr']}")

        if epoch_acc_val > best_acc:
            best_acc = epoch_acc_val
            torch.save(models.state_dict(), save_path)
            print(f"模型已保存至 {save_path}，验证准确率: {best_acc:.4f}")

        print(f"Epoch [{epoch+1:02d}/{epochs}], "
              f"训练损失: {epoch_loss_train:.4f}, 训练准确率: {epoch_acc_train:.4f} | "
              f"验证损失: {epoch_loss_val:.4f}, 验证准确率: {epoch_acc_val:.4f}")

if __name__ == "__main__":
    # 定义这些变量防止未定义报错
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    main()