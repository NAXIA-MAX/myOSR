import numpy as np
import random
import torch
import torch.nn.functional as F
from train_model import train_model
import resnet
from extract_fea import  remodConv, settings as ExtractSettings
from resnetTrain import NPYDatasetWithAugment,setTransform
import os
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# 创建必要的目录
os.makedirs("extractLog", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("cv_results", exist_ok=True)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def extract_features(model_path, X_data, y_data, data_name, num_classes, input_channel=3):
    """提取特征并保存"""
    # 创建数据集
    dataset = NPYDatasetWithAugment(X_data, y_data, transform=setTransform(useAugment=False))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # 加载模型
    model = resnet.build_resnet("resnet20", num_classes)
    if input_channel != 3:
        remodConv(model, input_channel)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # 注册hooks提取多层特征
    feature_outputs = []
    def hook_fn(module, input, output):
        feature_outputs.append(output)
    
    hooks = []
    for layer in [model.layer1, model.layer2, model.layer3]:
        hooks.append(layer.register_forward_hook(hook_fn))
    
    # 获取特征维度
    if input_channel==3:
        dummy_input = torch.zeros(1, input_channel, X_data[0].shape[1],X_data[0].shape[2]).to(device)
    elif input_channel==1:
        dummy_input = torch.zeros(1, input_channel, X_data[0].shape[0],X_data[0].shape[1]).to(device)
    with torch.no_grad():
        _ = model(dummy_input)
        featdims = [F.adaptive_avg_pool2d(feat, 1).view(1, -1).shape[1] for feat in feature_outputs]
    
    feature_outputs.clear()
    
    # 预分配存储
    num_samples = len(dataset)
    feat_log = np.zeros((num_samples, sum(featdims)), dtype=np.float32)
    score_log = np.zeros((num_samples, num_classes), dtype=np.float32)
    label_log = np.zeros(num_samples, dtype=np.int64)
    
    # 提取特征
    with torch.no_grad():
        idx = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
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
            idx += bsz
    
    # 保存特征
    np.save(f"extractLog/feat_log_{data_name}.npy", feat_log)
    np.save(f"extractLog/score_log_{data_name}.npy", score_log)
    np.save(f"extractLog/label_log_{data_name}.npy", label_log)
    
    # 清理hooks
    for hook in hooks:
        hook.remove()
    
    return feat_log, score_log, label_log

def knn_osr_detection(feat_id, feat_ood, label_id, label_ood, cv_round,unknown_classes):
    """使用KNN进行OSR检测"""
    import faiss
    
    # 预处理（归一化）
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,112)]))  # 使用最后一层特征
    
    fid = prepos_feat(feat_id)
    food = prepos_feat(feat_ood)
    
    # 构建KNN索引（使用ID数据）
    index = faiss.IndexFlatL2(fid.shape[1])
    index.add(fid)
    
    # KNN搜索
    K = 50
    D_id, _ = index.search(fid, K)
    D_ood, _ = index.search(food, K)
    
    # 计算分数（负距离，距离越小分数越高）
    scores_id = -D_id[:, -1]  # ID数据分数
    scores_ood = -D_ood[:, -1]  # OOD数据分数
    
    # 合并分数和标签
    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])  # 1表示ID，0表示OOD
    
    # 计算AUROC
    auroc = roc_auc_score(labels, scores)
    
    # 绘制结果
    plt.figure(figsize=(12, 5))
    
    # 子图1：分数分布
    plt.subplot(1, 2, 1)
    known_scores = scores_id
    unknown_scores = scores_ood
    
    plt.hist(known_scores, bins=50, alpha=0.5, label='ID Classes', color='blue', density=True)
    plt.hist(unknown_scores, bins=50, alpha=0.5, label='OOD Classes', color='red', density=True)
    plt.xlabel('KNN Score')
    plt.ylabel('Density')
    plt.legend()
    plt.title(f'CV Round {cv_round+1} - Score Distribution')
    plt.grid(True)
    
    # 子图2：ROC曲线
    plt.subplot(1, 2, 2)
    fpr, tpr, thresholds = roc_curve(labels, scores)
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'CV Round {cv_round+1} - ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"images/CVimages/cv_round_{cv_round+1}_unkonw_{'_'.join(map(str, unknown_classes))}.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    return auroc, scores, labels


def run_cross_validation(num_unknown=3, num_cv=5, data_path="/workspace/OSR/jammingData/3channels/", 
                        labels_file="labels_3channel_10000.npy", data_file="data_3channel_10000.npy"):
    """
    运行交叉验证实验
    
    Args:
        num_unknown (int): 未知类别数量，默认3
        num_cv (int): 交叉验证轮数，默认5
        data_path (str): 数据文件路径
        labels_file (str): 标签文件名
        data_file (str): 数据文件名
    """
    # 加载数据
    originLabels = np.load(os.path.join(data_path, labels_file))  
    data = np.load(os.path.join(data_path, data_file))
    
    
    if data[0].ndim == 3 and data[0].shape[0] == 3:
        input_channel = 3
    else:
        input_channel = 1
    
    num_classes = 10
    all_classes = list(range(num_classes))
    
    # 存储所有交叉验证结果
    cv_results = []

    for cv_round in range(num_cv):
        random.seed(cv_round + 42)
        unknown_classes = random.sample(all_classes, num_unknown)
        known_classes = [c for c in all_classes if c not in unknown_classes]

        print(f"\n=== CV Round {cv_round+1}/{num_cv} ===")
        print(f"Unknown classes = {unknown_classes}")
        print(f"Known classes = {known_classes}")

        # ID数据：只包含已知类
        id_idx = np.isin(originLabels, known_classes)
        X_id = data[id_idx]
        y_id = originLabels[id_idx]
        # 构建标签映射：已知类别 -> 0~numClasses-1
        mapping = {cls: i for i, cls in enumerate(sorted(known_classes))}
        y_id_mapped = np.array([mapping[l] for l in y_id])
        
        # ID数据进一步分为训练集和验证集
        trainData, valData, trainLabels, valLabels = train_test_split(
            X_id, y_id_mapped, test_size=0.2, random_state=42, stratify=y_id_mapped
        )
        print(f"ID数据训练种类：{np.unique(y_id_mapped)}")
        
        # OOD数据：只包含未知类
        ood_idx = np.isin(originLabels, unknown_classes)
        X_ood = data[ood_idx]
        y_ood = originLabels[ood_idx]
        print(f"OOD数据类别：{np.unique(y_ood)}")

        tailname = f"_cv{cv_round}_unknown_{'_'.join(map(str, unknown_classes))}"

        # 步骤1：训练模型
        print("步骤1：训练模型...")
        save_path = train_model(tailname, trainData, trainLabels, valData, valLabels,
                              saveStart_epoch=25, epochNum=50, numClasses=len(known_classes), 
                              savepthHead="trainedPth/trainedCVpth/")
        
        # 步骤2：提取ID数据特征
        print("步骤2：提取ID数据特征...")
        model_path = save_path
        feat_id, score_id, label_id = extract_features(
            model_path, X_id, y_id_mapped, f"id_cv{cv_round}", len(known_classes), input_channel
        )
        
        # 步骤3：提取OOD数据特征
        print("步骤3：提取OOD数据特征...")
        feat_ood, score_ood, label_ood = extract_features(
            model_path, X_ood, y_ood, f"ood_cv{cv_round}", len(known_classes), input_channel
        )
        
        # 步骤4：KNN OSR检测
        print("步骤4：进行KNN OSR检测...")
        auroc, scores, labels = knn_osr_detection(
            feat_id, feat_ood, label_id, label_ood, cv_round,unknown_classes
        )
        
        # 保存当前轮次结果
        num_id_samples = np.sum(labels == 1)
        num_ood_samples = np.sum(labels == 0)
        result = {
            'cv_round': cv_round + 1,
            'unknown_classes': unknown_classes,
            'known_classes': known_classes,
            'auroc': auroc,
            'num_id_samples': num_id_samples,
            'num_ood_samples': num_ood_samples
        }
        cv_results.append(result)
        
        print(f"CV Round {cv_round+1} 完成 - AUROC: {auroc:.4f}")
        print(f"ID样本数: {num_id_samples}, OOD样本数: {num_ood_samples}")

    # 汇总所有交叉验证结果
    print("\n=== 交叉验证结果汇总 ===")
    auroc_scores = [result['auroc'] for result in cv_results]
    mean_auroc = np.mean(auroc_scores)
    std_auroc = np.std(auroc_scores)

    print(f"平均AUROC: {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"各轮次AUROC: {[f'{score:.4f}' for score in auroc_scores]}")

    # 保存结果到文件
    results_summary = {
        'mean_auroc': mean_auroc,
        'std_auroc': std_auroc,
        'individual_results': cv_results
    }

    np.save(f"cv_results/cross_validation_results_{num_unknown}.npy", results_summary)

    # 绘制总体结果图
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.bar(range(1, num_cv+1), auroc_scores, alpha=0.7, color='skyblue')
    plt.axhline(y=mean_auroc, color='red', linestyle='--', label=f'Mean: {mean_auroc:.4f}')
    plt.xlabel('CV Round')
    plt.ylabel('AUROC')
    plt.title('Cross-Validation AUROC Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.hist(auroc_scores, bins=5, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(x=mean_auroc, color='red', linestyle='--', label=f'Mean: {mean_auroc:.4f}')
    plt.xlabel('AUROC')
    plt.ylabel('Frequency')
    plt.title('AUROC Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"images/CVimages/cv_summary_{num_unknown}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n结果已保存到:")
    print(f"- 详细结果: cv_results/cross_validation_results_{num_unknown}.npy")
    print(f"- 汇总图表: images/CVimages/cv_summary.png")
    print(f"- 各轮次图表: images/cv_round_*.png")
    
    return results_summary


def main():
    """主函数"""
    import sys
    
    # 默认参数
    num_unknown = 3
    num_cv = 5
    
    # 检查命令行参数
    if len(sys.argv) >= 2:
        num_unknown = int(sys.argv[1])
    if len(sys.argv) >= 3:
        num_cv = int(sys.argv[2])
    
    print("=== 交叉验证实验启动 ===")
    print(f"未知类别数量: {num_unknown}")
    print(f"交叉验证轮数: {num_cv}")
    
    # 运行交叉验证
    results = run_cross_validation(num_unknown=num_unknown, num_cv=num_cv)
    
    print("\n=== 实验完成 ===")
    return results


if __name__ == "__main__":
    main()
