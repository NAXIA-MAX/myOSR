import os
import numpy as np
import faiss
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import resnet
import argparse
import re

# 设置随机种子
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class NPYDatasetWithAugment(Dataset):
    """数据集类，用于加载numpy格式的图像数据"""
    def __init__(self, data_array, label_array=None, transform=None):
        self.data = data_array
        self.labels = label_array
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        
        # 处理不同维度的图像
        if image.ndim == 2:
            image_pil = Image.fromarray(image.astype(np.uint8)).convert('L')
        elif image.ndim == 3:
            if image.shape[0] == 1:
                image = np.squeeze(image, axis=0)
                image_pil = Image.fromarray(image.astype(np.uint8)).convert('L')
            elif image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
                image_pil = Image.fromarray(image.astype(np.uint8)).convert('RGB')
            else:
                raise ValueError(f"不支持的通道数: {image.shape[0]} at idx {idx}")
        else:
            raise ValueError(f"不支持的图像形状: {image.shape} at idx {idx}")

        image_tensor = self.transform(image_pil) if self.transform else transforms.ToTensor()(image_pil)

        if self.labels is not None:
            label = self.labels[idx]
            return image_tensor, label
        else:
            return image_tensor

def remodConv(model, inputChannel):
    """修改模型的第一层卷积层以适应不同的输入通道数"""
    conv1 = model.conv1
    out_channels = conv1.out_channels
    kernel_size = conv1.kernel_size
    stride = conv1.stride
    padding = conv1.padding
    bias = (conv1.bias is not None)

    model.conv1 = nn.Conv2d(inputChannel, out_channels, kernel_size=kernel_size, 
                           stride=stride, padding=padding, bias=bias)

def parse_model_filename(model_path):
    """
    解析模型文件名，提取未知类别信息
    例如: resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth
    返回: {'unknown_classes': [5, 7, 6], 'cv_fold': 0}
    """
    filename = os.path.basename(model_path)
    
    # 提取未知类别信息
    unknown_match = re.search(r'unknown_(\d+(?:_\d+)*)', filename)
    if unknown_match:
        unknown_str = unknown_match.group(1)
        unknown_classes = [int(x) for x in unknown_str.split('_')]
    else:
        raise ValueError(f"无法从文件名中解析未知类别: {filename}")
    
    # 提取CV折数
    cv_match = re.search(r'cv(\d+)', filename)
    cv_fold = int(cv_match.group(1)) if cv_match else 0
    
    return {
        'unknown_classes': unknown_classes,
        'cv_fold': cv_fold
    }

def get_known_classes(unknown_classes, num_total_classes=10):
    """
    获取已知类别列表
    """
    known_classes = [i for i in range(num_total_classes) if i not in unknown_classes]
    return known_classes

def extract_features(model, data_loader, device, input_channel=3):
    """从模型中提取特征"""
    model.eval()
    feature_outputs = []
    
    def hook_fn(module, input, output):
        feature_outputs.append(output)
    
    # 注册hook函数
    hooks = []
    for layer in [model.layer1, model.layer2, model.layer3]:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # 使用dummy输入获取特征维度
    # 根据输入通道数确定图像尺寸
    if input_channel == 3:
        dummy_input = torch.zeros(1, input_channel, data_loader.dataset.data[0].shape[1], data_loader.dataset.data[0].shape[2]).to(device)
    elif input_channel == 1:
        dummy_input = torch.zeros(1, input_channel, data_loader.dataset.data[0].shape[0], data_loader.dataset.data[0].shape[1]).to(device)
    else:
        raise ValueError(f"不支持的输入通道数: {input_channel}")
    
    with torch.no_grad():
        _ = model(dummy_input)
        featdims = [F.adaptive_avg_pool2d(feat, 1).view(1, -1).shape[1] for feat in feature_outputs]
    
    feature_outputs.clear()
    
    # 预分配存储
    num_samples = len(data_loader.dataset)
    feat_log = np.zeros((num_samples, sum(featdims)))
    score_log = np.zeros((num_samples, len(known_classes)))
    label_log = np.zeros(num_samples)
    
    print(f"特征维度: {sum(featdims)}")
    
    # 正式提取特征
    with torch.no_grad():
        idx = 0
        for batch_idx, (inputs, labels) in enumerate(data_loader):
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
            
            feature_outputs.clear()
            
            if batch_idx % 10 == 0:
                print(f"处理进度: {batch_idx}/{len(data_loader)}")
            
            idx += bsz
    
    # 移除hooks
    for hook in hooks:
        hook.remove()
    
    return feat_log, score_log, label_log

def knn_osr_detection(feat_train, feat_test, k=50):
    """
    KNN开放集识别检测
    Args:
        feat_train: 训练集特征 (已知类别)
        feat_test: 测试集特征 (包含已知和未知类别)
        k: 近邻数量
    Returns:
        scores: 检测分数 (距离分数)
    """
    # 构建KNN索引
    index = faiss.IndexFlatL2(feat_train.shape[1])
    index.add(feat_train.astype(np.float32))
    
    # 搜索近邻
    D, _ = index.search(feat_test.astype(np.float32), k)
    scores = -D[:, -1]  # 取第k近邻的距离，负号因为距离小越好
    
    return scores

def calculate_auroc(scores_in, scores_out):
    """计算AUROC指标"""
    labels = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    scores = np.concatenate([scores_in, scores_out])
    
    auroc = roc_auc_score(labels, scores)
    return auroc, labels, scores

def plot_results(scores_in, scores_out, auroc, model_name, save_dir="results"):
    """绘制结果图表"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制分数分布
    plt.figure(figsize=(10, 6))
    plt.hist(scores_in, bins=100, alpha=0.5, label='已知类别 (In-Distribution)', color='blue', density=True)
    plt.hist(scores_out, bins=100, alpha=0.5, label='未知类别 (Out-of-Distribution)', color='red', density=True)
    plt.xlabel('KNN分数 (越高表示越可能是已知类别)')
    plt.ylabel('密度')
    plt.legend()
    plt.title(f'KNN OOD检测分数分布 (AUROC={auroc*100:.2f}%)')
    plt.grid(True)
    
    dist_path = os.path.join(save_dir, f'{model_name}_score_distribution.png')
    plt.savefig(dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 绘制ROC曲线
    labels = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_out)])
    scores = np.concatenate([scores_in, scores_out])
    
    fpr, tpr, thresholds = roc_curve(labels, scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUROC = {auroc*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (FPR)')
    plt.ylabel('真正率 (TPR)')
    plt.title(f'ROC曲线 - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    roc_path = os.path.join(save_dir, f'{model_name}_roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"分布图已保存至: {dist_path}")
    print(f"ROC曲线已保存至: {roc_path}")

def main():
    parser = argparse.ArgumentParser(description='测试KNN-OSR开放集识别效果')
    parser.add_argument('--model_path', type=str, required=True,
                       help='训练好的模型路径')
    parser.add_argument('--data_path', type=str, required=True,
                       help='完整数据集路径 (.npy)')
    parser.add_argument('--label_path', type=str, required=True,
                       help='完整标签集路径 (.npy)')
    parser.add_argument('--input_channel', type=int, default=3,
                       help='输入通道数')
    parser.add_argument('--k', type=int, default=50,
                       help='KNN近邻数量')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='批次大小')
    parser.add_argument('--test_ratio', type=float, default=0.5,
                       help='ID数据中用于测试的比例')
    
    args = parser.parse_args()
    
    # 解析模型文件名
    model_info = parse_model_filename(args.model_path)
    unknown_classes = model_info['unknown_classes']
    cv_fold = model_info['cv_fold']
    
    print(f"模型信息:")
    print(f"  未知类别: {unknown_classes}")
    print(f"  CV折数: {cv_fold}")
    
    # 获取已知类别
    global known_classes
    known_classes = get_known_classes(unknown_classes)
    print(f"  已知类别: {known_classes}")
    print(f"  未知类别: {unknown_classes}")
    
    # 加载完整数据
    print("加载完整数据...")
    all_data = np.load(args.data_path)
    all_labels = np.load(args.label_path)
    
    print(f"总数据: {len(all_data)} 样本")
    
    # 根据已知类别和未知类别拆分数据
    id_mask = np.isin(all_labels, known_classes)
    ood_mask = np.isin(all_labels, unknown_classes)
    
    # ID数据（已知类别）
    id_data = all_data[id_mask]
    id_labels_original = all_labels[id_mask]
    
    # OOD数据（未知类别）
    ood_data = all_data[ood_mask]
    ood_labels = all_labels[ood_mask]
    
    print(f"ID数据（已知类别）: {len(id_data)} 样本")
    print(f"OOD数据（未知类别）: {len(ood_data)} 样本")
    
    # 将ID数据进一步分为训练集和测试集（标签映射只是为了数据加载，KNN-OSR不需要）
    from sklearn.model_selection import train_test_split
    id_train_data, id_test_data, id_train_labels, id_test_labels = train_test_split(
        id_data, id_labels_original, test_size=args.test_ratio, random_state=42, stratify=id_labels_original
    )
    
    print(f"ID训练数据: {len(id_train_data)} 样本")
    print(f"ID测试数据: {len(id_test_data)} 样本")
    
    # 设置数据预处理
    if args.input_channel == 3:
        mean = [0.558, 0.537, 0.886]
        std = [0.469, 0.129, 0.245]
    else:
        mean = [0.5]
        std = [0.5]
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    # 创建数据加载器（标签只是为了数据加载，实际KNN-OSR检测不需要标签）
    # ID训练数据用于构建KNN索引
    id_train_dataset = NPYDatasetWithAugment(id_train_data, id_train_labels, transform=transform)
    id_train_loader = DataLoader(id_train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # ID测试数据用于计算ID分数
    id_test_dataset = NPYDatasetWithAugment(id_test_data, id_test_labels, transform=transform)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # OOD数据用于计算OOD分数
    ood_dataset = NPYDatasetWithAugment(ood_data, ood_labels, transform=transform)
    ood_loader = DataLoader(ood_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 加载模型
    print("加载模型...")
    model = resnet.build_resnet("resnet20", len(known_classes))
    if args.input_channel != 3:
        remodConv(model, args.input_channel)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 提取特征
    print("提取ID训练特征（用于构建KNN索引）...")
    feat_id_train, _, _ = extract_features(model, id_train_loader, device, args.input_channel)
    
    print("提取ID测试特征...")
    feat_id_test, _, _ = extract_features(model, id_test_loader, device, args.input_channel)
    
    print("提取OOD特征...")
    feat_ood, _, _ = extract_features(model, ood_loader, device, args.input_channel)
    
    # 特征归一化
    def normalize_features(feat):
        return feat / (np.linalg.norm(feat, ord=2, axis=-1, keepdims=True) + 1e-10)
    
    feat_id_train_norm = normalize_features(feat_id_train)
    feat_id_test_norm = normalize_features(feat_id_test)
    feat_ood_norm = normalize_features(feat_ood)
    
    # KNN-OSR检测
    print("执行KNN-OSR检测...")
    
    # 使用ID训练数据构建KNN索引，对ID测试数据进行检测（应该得到较高的分数）
    scores_id = knn_osr_detection(feat_id_train_norm, feat_id_test_norm, args.k)
    
    # 使用ID训练数据构建KNN索引，对OOD数据进行检测（应该得到较低的分数）
    scores_ood = knn_osr_detection(feat_id_train_norm, feat_ood_norm, args.k)
    
    # 计算AUROC
    auroc, labels, scores = calculate_auroc(scores_id, scores_ood)
    
    print(f"\n结果:")
    print(f"KNN-OSR检测AUROC: {auroc * 100:.2f}%")
    print(f"ID数据平均分数: {np.mean(scores_id):.4f}")
    print(f"OOD数据平均分数: {np.mean(scores_ood):.4f}")
    
    # 绘制结果
    model_name = os.path.splitext(os.path.basename(args.model_path))[0]
    plot_results(scores_id, scores_ood, auroc, model_name)
    
    return auroc

if __name__ == "__main__":
    main()
