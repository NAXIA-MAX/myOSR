import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from resnetTrain import creatDataLoaderNPY,NPYDatasetWithAugment
from sklearn.model_selection import train_test_split
def compute_mean_std(loader):
    mean = torch.zeros(3)
    std = torch.zeros(3)
    total_images = 0

    print("Computing mean and std...")
    for images, _ in tqdm(loader):
        batch_samples = images.size(0)  # batch size (B)
        images = images.view(batch_samples, 3, -1)  # flatten H*W

        mean += images.mean(2).sum(0)  # sum over batch
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    return mean.tolist(), std.tolist()
def main():
    data=np.load("/workspace/OSR/jammingData/3channels/X_3channels_train.npy")
    labels=np.load("/workspace/OSR/jammingData/3channels/y_3channels_train.npy")
    print("start..")
    transform = transforms.Compose([
        transforms.ToTensor()  # Converts to [0,1] and shape [C, H, W]
    ])
    train_loader,_ = creatDataLoaderNPY(data, labels,transform)
    mean, std = compute_mean_std(train_loader)
    print("Mean:", mean)
    print("Std:", std)
if __name__ == "__main__":
    main()