import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import random, os
import resnet
from resnetTrain import NPYDatasetWithAugment, setTransform, creatDataLoaderNPY, remodConv  # 假设你把Dataset和transform函数放dataset.py里

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_model(Tailname, trainData, trainLabels, testData, testLabels,
                saveStart_epoch=30,epochNum=50,
                model_name="resnet20", numClasses=10, inputChannel=3,
                useNoiseAug=False, noiseStd=0.05, savepthHead="trainedPth/"):

    set_seed()

    train_transform = setTransform(useAugment=True, useNoise=useNoiseAug, noiseStd=noiseStd)
    test_transform = setTransform(useAugment=False)

    train_loader, test_loader = creatDataLoaderNPY(trainData, trainLabels, train_transform,
                                                   testData, testLabels, test_transform)

    model = resnet.build_resnet(model_name, numClasses)
    if inputChannel != 3:
        remodConv(model, inputChannel)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.1, verbose=True)

    epochs = epochNum
    best_acc = 0.0
    patience = 5
    no_improve_count = 0
    save_start_epoch = saveStart_epoch

    if useNoiseAug:
        withNoise = f"Noise_{noiseStd}"
    else:
        withNoise = "None"
    save_path = f"{savepthHead}{model_name}_{inputChannel}channels_{numClasses}classes_{withNoise}{Tailname}.pth"
    os.makedirs(savepthHead, exist_ok=True)

    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_loss_train = running_loss / total_train
        epoch_acc_train = correct_train / total_train

        model.eval()
        val_running_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_loss_val = val_running_loss / total_val
        epoch_acc_val = correct_val / total_val
        print(f"Epoch [{epoch+1:02d}/{epochs}], "
            f"训练损失: {epoch_loss_train:.4f}, 训练准确率: {epoch_acc_train:.4f} | "
            f"验证损失: {epoch_loss_val:.4f}, 验证准确率: {epoch_acc_val:.4f}")
        val_losses.append(epoch_loss_val)
        val_accuracies.append(epoch_acc_val)
        train_losses.append(epoch_loss_train)
        train_accuracies.append(epoch_acc_train)

        scheduler.step(epoch_loss_val)

        if epoch >= save_start_epoch:
            if epoch_acc_val > best_acc:
                best_acc = epoch_acc_val
                no_improve_count = 0
                torch.save(model.state_dict(), save_path)
                print(f"[Epoch {epoch+1}] 模型已保存，验证准确率: {best_acc:.4f}")
            else:
                no_improve_count += 1

        if epoch >= save_start_epoch and no_improve_count >= patience:
            print(f"[Epoch {epoch+1}] 验证准确率连续 {patience} 轮未提升，提前停止训练。")
            break

    return save_path
