import os
import numpy as np
import faiss
import torch

# 设置随机种子
torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
IDdata="0_5"
OODdata="6_9"
# 加载数据
feat_log = np.load(f"extractLog/feat_log_{IDdata}.npy").astype(np.float32)
score_log = np.load(f"extractLog/score_log_{IDdata}.npy").astype(np.float32)
label_log = np.load(f"extractLog/label_log_{IDdata}.npy")

feat_log_ood = np.load(f"extractLog/feat_log_{OODdata}.npy").astype(np.float32)
score_log_ood = np.load(f"extractLog/score_log_{OODdata}.npy").astype(np.float32)
label_log_ood = np.load(f"extractLog/label_log_{OODdata}.npy")
print("data has downloaded")
# 预处理（归一化）
# def normalize(x):
#     return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-10)
#feat_log = feat_log[:20000, :]#取前10000个样本
normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)

prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,112)]))# Last Layer only
ftrain = prepos_feat(feat_log)        # 分布内样本 (用来建KNN库)
food = prepos_feat(feat_log_ood)      # 分布外样本 (用来测试OOD)

#构建 KNN 索引
index = faiss.IndexFlatL2(ftrain.shape[1])  # 使用 L2 距离
index.add(ftrain)

# KNN并打分
K = 50  # 近邻数量，可以调整

# 分布内得分
D_in, _ = index.search(ftrain, K)
scores_in = -D_in[:, -1]   # 取第K近邻的距离，负号因为距离小越好

# 分布外得分
D_ood, _ = index.search(food, K)
scores_ood = -D_ood[:, -1]

# AUROC指标计算
from sklearn.metrics import roc_auc_score

labels = np.concatenate([np.ones_like(scores_in), np.zeros_like(scores_ood)])
scores = np.concatenate([scores_in, scores_ood])

auroc = roc_auc_score(labels, scores)
print(f"KNN Detection AUROC = {auroc * 100:.2f}%")

# KNN绘图
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.hist(scores_in, bins=100, alpha=0.5, label='In-Distribution', color='blue', density=True)
plt.hist(scores_ood, bins=100, alpha=0.5, label='Out-of-Distribution', color='red', density=True)
plt.xlabel('KNN Score (higher means more likely In)')
plt.ylabel('Density')
plt.legend()
plt.title(f'KNN OOD Detection (AUROC={auroc*100:.2f}%)')
plt.grid(True)

save_path = 'knn_ood_distribution.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Score distribution figure saved to {save_path}")

# 绘制 ROC 曲线并保存
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(labels, scores)  # labels和scores是前面用于计算auroc的
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUROC = {auroc*100:.2f}%)')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # 虚线表示随机猜测
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f"ROC curve ID({IDdata}) OOD({OODdata})")
plt.legend(loc='lower right')
plt.grid(True)

save_path_roc = f"roc_curve_ID({IDdata})_OOD({OODdata}).png"
plt.savefig(save_path_roc, dpi=300, bbox_inches='tight')
print(f"ROC curve figure saved to {save_path_roc}")

