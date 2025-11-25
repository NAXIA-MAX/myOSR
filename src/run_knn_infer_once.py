import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import faiss
import re

import resnet
from extract_fea import remodConv


def ensure_dirs():
    os.makedirs("extractLog", exist_ok=True)
    os.makedirs("images", exist_ok=True)


def build_model_cpu(model_name: str, num_classes: int, input_channel: int, model_path: str) -> torch.nn.Module:
    # 构建并加载模型到CPU
    model = resnet.build_resnet(model_name, num_classes)
    if input_channel != 3:
        remodConv(model, input_channel)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def extract_features_cpu(model: torch.nn.Module,
                         X_data: np.ndarray,
                         y_data: np.ndarray,
                         input_channel: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 使用CPU提取多层特征
    from extract_fea import NPYDatasetWithAugment as NpyDS
    from extract_fea import transform as default_transform
    from torch.utils.data import DataLoader

    dataset = NpyDS(X_data, y_data, transform=default_transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=False)

    feature_outputs = []

    def hook_fn(module, input, output):
        feature_outputs.append(output)

    hooks = []
    for layer in [model.layer1, model.layer2, model.layer3]:
        hooks.append(layer.register_forward_hook(hook_fn))

    # 通过dummy输入拿到各层维度
    if X_data[0].ndim == 3 and X_data[0].shape[0] == input_channel:
        h, w = X_data[0].shape[-2], X_data[0].shape[-1]
    elif X_data[0].ndim == 2:
        h, w = X_data[0].shape[0], X_data[0].shape[1]
    else:
        # 回退：尝试从数据本身推断
        h = X_data.shape[-2]
        w = X_data.shape[-1]

    with torch.no_grad():
        _ = model(torch.zeros(1, input_channel, h, w))
        featdims = [F.adaptive_avg_pool2d(feat, 1).view(1, -1).shape[1] for feat in feature_outputs]

    feature_outputs.clear()

    num_samples = len(dataset)
    feat_log = np.zeros((num_samples, sum(featdims)), dtype=np.float32)
    score_log = np.zeros((num_samples, model.fc.out_features), dtype=np.float32)
    label_log = np.zeros(num_samples, dtype=np.int64)

    with torch.no_grad():
        idx = 0
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            outputs = model(inputs)

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

            if (batch_idx % 10 == 0) or (idx >= num_samples):
                print(f"feature extracting: {idx}/{num_samples}")

    for hook in hooks:
        hook.remove()

    return feat_log, score_log, label_log


def knn_osr_once(feat_id: np.ndarray,
                 feat_ood: np.ndarray,
                 labels_id: np.ndarray,
                 known_class_names: list[str],
                 K: int = 50,
                 fig_prefix: str = "once") -> tuple[float, np.ndarray, np.ndarray]:
    # 固定与原实现一致：仅用索引 [0,112) 的特征维度
    normalizer = lambda x: x / (np.linalg.norm(x, ord=2, axis=-1, keepdims=True) + 1e-10)
    prepos_feat = lambda x: np.ascontiguousarray(normalizer(x[:, range(0,112)]))

    fid = prepos_feat(feat_id).astype(np.float32)
    food = prepos_feat(feat_ood).astype(np.float32)

    index = faiss.IndexFlatL2(fid.shape[1])
    index.add(fid)

    D_id, _ = index.search(fid, K)
    D_ood, _ = index.search(food, K)

    scores_id = -D_id[:, -1]
    scores_ood = -D_ood[:, -1]

    scores = np.concatenate([scores_id, scores_ood])
    labels = np.concatenate([np.ones_like(scores_id), np.zeros_like(scores_ood)])

    # 已知/未知分布图
    plt.figure(figsize=(8, 6))
    plt.hist(scores_id, bins=80, alpha=0.5, label='Known (ID)', color='blue', density=True)
    plt.hist(scores_ood, bins=80, alpha=0.5, label='Unknown (OOD)', color='red', density=True)
    plt.xlabel('KNN Score')
    plt.ylabel('Density')
    plt.legend()
    plt.title('KNN Score Distribution: Known vs Unknown')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"images/{fig_prefix}_knn_known_unknown.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 各已知类的KNN分布（对每个类单独检索后绘制）
    plt.figure(figsize=(10, 7))
    unique_known = np.unique(labels_id)
    for k in unique_known:
        mask = labels_id == k
        if not np.any(mask):
            continue
        D_k, _ = index.search(fid[mask], K)
        scores_k = -D_k[:, -1]
        label_name = known_class_names[k] if k < len(known_class_names) else f'class_{int(k)}'
        plt.hist(scores_k, bins=60, alpha=0.4, density=True, label=label_name)
    plt.xlabel('KNN Score')
    plt.ylabel('Density')
    plt.title('KNN Score Distribution per Known Class')
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"images/{fig_prefix}_knn_per_class.png", dpi=300, bbox_inches='tight')
    plt.close()

    # ROC曲线与AUROC
    from sklearn.metrics import roc_auc_score, roc_curve
    auroc = roc_auc_score(labels, scores)
    fpr, tpr, _ = roc_curve(labels, scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUROC={auroc*100:.2f}%)')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve: Known vs Unknown')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"images/{fig_prefix}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()

    return auroc, scores, labels


def main():
    # 在此直接填写你的具体输入（数据是全量10类）
    model_path = r"trainedPth/resnet20_3channels_7classes_None_cv0_unknown_1_0_4.pth"
    data_path = r"jammingData"
    data_file = "data_3channel_10000.npy"
    labels_file = "labels_3channel_10000.npy"
    tag = "once"

    # 解析模型名：示例 resnet20_3channels_7classes_None_cv0_unknown_1_0_4
    base = os.path.basename(model_path)
    name_no_ext = os.path.splitext(base)[0]

    # 主干
    backbone_match = re.match(r"^(resnet\d+)_", name_no_ext)
    model_name = backbone_match.group(1) if backbone_match else "resnet20"

    # 通道
    input_channel = 3 if re.search(r"_3channels\b", name_no_ext) else None

    # 未知类列表
    unknown_match = re.search(r"unknown_([0-9_]+)$", name_no_ext)
    unknown_list = []
    if unknown_match:
        unknown_str = unknown_match.group(1)
        unknown_list = [int(x) for x in unknown_str.split('_') if x != '']

    # 由未知数确定类别数（总类10）
    total_classes = 10
    num_classes = total_classes - len(unknown_list)

    # 已知/未知划分（从全量10类中划分）
    known_cls_list = sorted(list(set(range(total_classes)) - set(unknown_list)))

    ensure_dirs()

    # 加载数据
    data = np.load(os.path.join(data_path, data_file))
    labels = np.load(os.path.join(data_path, labels_file))

    # 输入通道数判定
    if input_channel is None:
        if data[0].ndim == 3 and data[0].shape[0] == 3:
            in_ch = 3
        else:
            in_ch = 1
    else:
        in_ch = input_channel

    # 已知/未知划分
    unknown_mask = ~np.isin(labels, known_cls_list)
    known_mask = ~unknown_mask

    X_id = data[known_mask]
    y_id_original = labels[known_mask]
    X_ood = data[unknown_mask]
    y_ood_original = labels[unknown_mask]

    # 将已知类映射到0..K-1以适配训练好的FC (需与训练一致)
    mapping = {cls: i for i, cls in enumerate(known_cls_list)}
    y_id_mapped = np.array([mapping[v] for v in y_id_original], dtype=np.int64)

    # 模型类别数（由未知数决定）
    n_classes = int(num_classes)

    # 构建并加载模型到CPU
    model = build_model_cpu(model_name, n_classes, in_ch, model_path)

    # CPU提取特征
    feat_id, score_id, label_id = extract_features_cpu(model, X_id, y_id_mapped, in_ch)
    # OOD提取时标签原样存，仅用于记录
    if X_ood.shape[0] > 0:
        fake_map = {v: i for i, v in enumerate(sorted(np.unique(y_ood_original)))}
        y_ood_temp = np.array([fake_map[v] for v in y_ood_original], dtype=np.int64)
        feat_ood, score_ood, label_ood = extract_features_cpu(model, X_ood, y_ood_temp, in_ch)
    else:
        feat_ood = np.zeros((0, feat_id.shape[1]), dtype=np.float32)
        score_ood = np.zeros((0, n_classes), dtype=np.float32)
        label_ood = np.zeros(0, dtype=np.int64)

    # 保存提取日志
    np.save(f"extractLog/feat_log_id_{tag}.npy", feat_id)
    np.save(f"extractLog/score_log_id_{tag}.npy", score_id)
    np.save(f"extractLog/label_log_id_{tag}.npy", label_id)
    np.save(f"extractLog/feat_log_ood_{tag}.npy", feat_ood)
    np.save(f"extractLog/score_log_ood_{tag}.npy", score_ood)
    np.save(f"extractLog/label_log_ood_{tag}.npy", label_ood)

    # KNN-OSR 单轮评估与作图
    known_names = [str(c) for c in known_cls_list]
    auroc, scores, bin_labels = knn_osr_once(feat_id, feat_ood, label_id, known_names, fig_prefix=tag)

    # 打印与保存摘要
    print(f"AUROC = {auroc:.4f}")
    np.save(f"extractLog/knn_scores_{tag}.npy", scores)
    np.save(f"extractLog/knn_bin_labels_{tag}.npy", bin_labels)
    print("结果与图像已保存到 extractLog/ 与 images/ 目录。")


if __name__ == "__main__":
    main()


