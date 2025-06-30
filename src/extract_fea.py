import torch
import resnet
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader,Dataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
"""
filenameList = ["normal","CW","SCW(0.1023)","SCW(1.023)","AGWN(0.1023)","AGWN(1.023)",
                "PI(1ms_0.3)","PI(1ms_0.7)","PI(10ms_0.3)","PI(10ms_0.7)"]
"""
#定义一个继承类，继承Dataset
class NPYDatasetWithAugment(Dataset):
    def __init__(self, data_array, label_array=None, transform=None):
        self.data = data_array
        self.labels = label_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        
        # 保证 image 是 [H, W]
        if image.ndim == 3 and image.shape[0] == 1:
            image = np.squeeze(image, axis=0)  # [1, H, W] → [H, W]

        image_pil = Image.fromarray(image).convert('L')  # 灰度图
        image_tensor = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)

        if self.labels is not None:
            label = self.labels[idx]
            return image_tensor, label
        else:
            return image_tensor
    

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

# 设置数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])  
])

class settings:
    def __init__(self,dataName,labelName,trainedModel,model_name="resnet20",num_classes=7,inputChannel=1 ):
        self.model_name=model_name
        self.num_classes=num_classes
        self.inputChannel=inputChannel
        self.dataName=dataName
        self.labelName=labelName
        self.trainedModel=trainedModel

def main(s):
    model_name=s.model_name#模型名称
    num_classes=s.num_classes         #训练种类数
    inputChannel=s.inputChannel
    dataName=s.dataName
    labelName=s.labelName
    # 加载数据集
    fileFolder="ID_OODdata/"
    trainedfolder="trainedPth/"
    testData=np.load(f"{fileFolder}{dataName}")
    testLabels=np.load(f"{fileFolder}{labelName}")
    datashpae=testData[0].shape
    testset = NPYDatasetWithAugment(testData, testLabels,transform=transform)
    testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

    # 加载模型
    model =resnet.build_resnet(model_name,num_classes)
    #如果输入的通道数不为3，则进行调整
    if inputChannel!=3:
        remodConv(model,inputChannel)
    model.load_state_dict(torch.load(f"{trainedfolder}{s.trainedModel}"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 注册hooks，提取多层特征
    feature_outputs = []
    def hook_fn(module, input, output):
        feature_outputs.append(output)
    # 注册 hook 函数
    hooks = []
    for layer in [model.layer1, model.layer2, model.layer3]:  # 选择需要提取的层
        hooks.append(layer.register_forward_hook(hook_fn))
    # 使用 dummy 输入获取特征维度
    dummy_input = torch.zeros(1, inputChannel, datashpae[0],datashpae[1] ).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
        featdims = [F.adaptive_avg_pool2d(feat, 1).view(1, -1).shape[1] for feat in feature_outputs]

    # 清空 feature_outputs
    feature_outputs.clear()

    # 预分配存储
    num_samples = len(testset)
    feat_log = np.zeros((num_samples, sum(featdims)))
    score_log = np.zeros((num_samples, num_classes))
    label_log = np.zeros(num_samples)

    print(sum(featdims))  # 输出提取的层数

    # 正式提取特征
    batch_size = 64
    with torch.no_grad():
        idx = 0
        for batch_idx, (inputs, labels) in enumerate(testloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 提取并处理特征
            pooled_features = []
            for feat in feature_outputs:
                pooled = F.adaptive_avg_pool2d(feat, 1).view(feat.size(0), -1)
                pooled_features.append(pooled)
            out = torch.cat(pooled_features, dim=1)

            bsz = inputs.size(0)
            feat_log[idx:idx+bsz, :] = out.cpu().numpy()
            score_log[idx:idx+bsz, :] = outputs.cpu().numpy()
            label_log[idx:idx+bsz] = labels.cpu().numpy()

            feature_outputs.clear()  # 清空一次，避免堆叠

            if batch_idx % 10 == 0:
                print(f"{batch_idx}/{len(testloader)} processed.")

            idx += bsz

    # 保存结果
    savename = dataName.split('_', 1)[1].rsplit('.', 1)[0]
    np.save(f"extractLog/feat_log_{savename}.npy", feat_log)
    np.save(f"extractLog/score_log_{savename}.npy", score_log)
    np.save(f"extractLog/label_log_{savename}.npy", label_log)

    print("Done! Features, scores, labels saved.")

if __name__ == "__main__":
    dataName="OODimages_6_9.npy"
    labelName="OODlabels_6_9.npy"
    trainedModel="resnet20_best_6.pth"
    num_classes=6
    s=settings(dataName,labelName,trainedModel,num_classes=num_classes)
    main(s)