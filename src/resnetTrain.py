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
        image = self.data[idx]  # shape: [H, W], [1, H, W], [3, H, W], etc.
        label = self.labels[idx]

        # 1. 单通道图像：可能是 [1, H, W] 或 [H, W]
        if image.ndim == 2:
            image_pil = Image.fromarray(image.astype(np.uint8)).convert('L')  # [H, W]
        elif image.ndim == 3:
            if image.shape[0] == 1:
                # [1, H, W] → [H, W]
                image = np.squeeze(image, axis=0)
                image_pil = Image.fromarray(image.astype(np.uint8)).convert('L')
            elif image.shape[0] == 3:
                # [3, H, W] → [H, W, 3]
                image = np.transpose(image, (1, 2, 0))
                image_pil = Image.fromarray(image.astype(np.uint8)).convert('RGB')
            else:
                raise ValueError(f"Unsupported channel count (first dim): {image.shape[0]} at idx {idx}")
        else:
            raise ValueError(f"Unsupported image shape: {image.shape} at idx {idx}")

        # 应用 transform
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

    transform_list.append(transforms.Normalize(mean=[0.558,0.537,0.886], std=[0.469,0.129,0.245]))#3通道的mean、std
    return transforms.Compose(transform_list)

def creatDataLoaderNPY(trainData, trainLabels, train_transform, 
                       testData=None, testLabels=None, test_transform=None): 
    #数据预处理
    train_dataset = NPYDatasetWithAugment(trainData, trainLabels, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=8, pin_memory=True) 

    if testData is not None and testLabels is not None and test_transform is not None:
        test_dataset = NPYDatasetWithAugment(testData, testLabels, transform=test_transform) 
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True) 
    else:
        test_loader = None

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
    numClasses = 10        #训练种类数
    inputChannel = 3      #输入通道数
    useNoiseAug=False      #使用噪声增强
    noiseStd=0.05
    savepthHead="trainedPth/"
    addTailname="_jam30_40_50"
    data=np.load("jammingData/data_3channel_30_40_50.npy")
    labels=np.load("jammingData/labels_3channel_30_40_50.npy")
    # 划分训练测试集
    trainData, testData, trainLabels, testLabels = train_test_split(
        data,labels , test_size=0.2, random_state=42, stratify=labels
    )
    train_transform=setTransform(useAugment=True,useNoise=useNoiseAug,noiseStd=noiseStd)
    test_transform=setTransform(useAugment=False)
    train_loader, test_loader = creatDataLoaderNPY(trainData, trainLabels, train_transform,testData,testLabels,test_transform)
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
    patience = 5
    no_improve_count = 0
    save_start_epoch = 30
    if useNoiseAug:
        withNoise=f"NoiseStd_{noiseStd}"
    else :
        withNoise="None"
    save_path = f"{savepthHead}{model_name}_{inputChannel}channels_{numClasses}classes_{withNoise}{addTailname}.pth"
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

        if epoch >= save_start_epoch:
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                no_improve_count = 0
                torch.save(models.state_dict(), save_path)
                print(f"模型已保存至 {save_path}，验证准确率: {best_acc:.4f}")
            else:
                no_improve_count += 1
                print(f"验证准确率未提升，连续 {no_improve_count} 次")
        else:
            print(f"第 {epoch+1} 轮，尚未开始保存模型，未触发early stopping")

        print(f"Epoch [{epoch+1:02d}/{epochs}], "
            f"训练损失: {epoch_loss_train:.4f}, 训练准确率: {epoch_acc_train:.4f} | "
            f"验证损失: {epoch_loss_val:.4f}, 验证准确率: {epoch_acc_val:.4f}")

        if epoch >= save_start_epoch and no_improve_count >= patience:
            print(f"验证准确率连续 {patience} 次未提升，提前停止训练。")
            break

if __name__ == "__main__":
    # 定义这些变量防止未定义报错
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    main()