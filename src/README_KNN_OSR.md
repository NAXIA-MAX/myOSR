# KNN-OSR 开放集识别测试工具

## 概述

这是一个用于测试已训练模型的KNN-OSR（K-Nearest Neighbor Open Set Recognition）开放集识别效果的Python工具。系统能够自动从完整数据集中根据模型文件名拆分ID和OOD数据，并进行标签映射。

## 主要功能

✅ **自动数据拆分**: 从包含所有类别(0-9)的完整数据集中自动拆分ID和OOD数据
✅ **智能标签映射**: 根据模型文件名自动处理原始标签到训练标签的映射
✅ **特征提取**: 从已训练模型中提取多层特征
✅ **KNN-OSR检测**: 实现基于KNN的开放集识别算法
✅ **AUROC评估**: 计算并输出AUROC指标
✅ **可视化结果**: 生成分数分布图和ROC曲线

## 文件说明

- `test_knn_osr.py`: 主要测试脚本
- `run_knn_test_example.py`: 使用示例和交互式运行脚本
- `README_KNN_OSR.md`: 使用说明文档

## 使用方法

### 基本用法

```bash
python test_knn_osr.py \
  --model_path /workspace/OSR/trainedPth/trainedCVpth/resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth \
  --data_path data/all_images.npy \
  --label_path data/all_labels.npy
```

### 完整参数

```bash
python test_knn_osr.py \
  --model_path <模型路径> \
  --data_path <完整数据集路径> \
  --label_path <完整标签集路径> \
  --input_channel 3 \
  --k 50 \
  --batch_size 64 \
  --test_ratio 0.5
```

### 参数说明

- `--model_path`: 训练好的模型文件路径
- `--data_path`: 包含所有类别(0-9)的完整数据集路径
- `--label_path`: 对应的完整标签集路径
- `--input_channel`: 输入通道数 (默认: 3)
- `--k`: KNN近邻数量 (默认: 50)
- `--batch_size`: 批次大小 (默认: 64)
- `--test_ratio`: ID数据中用于测试的比例 (默认: 0.5)

## 数据格式要求

### 数据集要求
- **完整数据集**: 包含所有类别(0-9)的图像数据
- **数据格式**: `.npy`文件，形状为`(N, H, W)`或`(N, C, H, W)`
- **标签格式**: `.npy`文件，形状为`(N,)`，包含类别标签0-9

### 模型文件名格式
模型文件名应包含未知类别信息，例如：
```
resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth
```
- 未知类别: `5, 7, 6`
- 已知类别: `0, 1, 2, 3, 4, 8, 9`
- CV折数: `0`

## 工作流程

1. **解析模型文件名**: 提取未知类别信息
2. **创建标签映射**: 将原始标签映射为训练标签
3. **数据拆分**: 从完整数据集中拆分ID和OOD数据
4. **ID数据细分**: 将ID数据进一步分为训练集和测试集
5. **特征提取**: 从模型中提取多层特征
6. **KNN-OSR检测**: 使用ID训练数据构建KNN索引，检测ID测试数据和OOD数据
7. **结果评估**: 计算AUROC指标并生成可视化结果

## 类别信息示例

对于模型文件名`resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth`:

```
已知类别: [0, 1, 2, 3, 4, 8, 9]
未知类别: [5, 7, 6]
```

**注意**: KNN-OSR检测只需要特征，不需要标签映射。标签仅用于数据加载和类别判断。

## 输出结果

运行完成后会生成：

1. **控制台输出**:
   - 模型信息和标签映射
   - 数据拆分统计
   - AUROC指标
   - ID和OOD数据平均分数

2. **可视化图表** (保存在`results/`目录):
   - `{model_name}_score_distribution.png`: 分数分布图
   - `{model_name}_roc_curve.png`: ROC曲线图

## 使用示例脚本

运行交互式示例：
```bash
python run_knn_test_example.py
```

该脚本会：
- 显示详细使用说明
- 创建必要的目录结构
- 提供交互式运行选项
- 显示示例命令

## 目录结构

```
src/
├── test_knn_osr.py              # 主要测试脚本
├── run_knn_test_example.py      # 使用示例脚本
├── README_KNN_OSR.md            # 使用说明
├── data/                        # 数据目录
│   ├── all_images.npy          # 完整数据集
│   └── all_labels.npy          # 完整标签集
└── results/                     # 结果目录
    ├── model_name_score_distribution.png
    └── model_name_roc_curve.png
```

## 注意事项

1. 确保所有依赖包已安装：`torch`, `numpy`, `faiss`, `matplotlib`, `sklearn`, `PIL`
2. 模型文件路径和数据文件路径要正确
3. 数据格式要与代码要求一致
4. 模型文件名要包含正确的未知类别信息
5. 系统会自动根据实际数据尺寸设置dummy输入，无需手动指定图像尺寸

## 故障排除

如果遇到问题，请检查：
- 模型文件是否存在且可访问
- 数据文件格式是否正确
- 模型文件名是否包含`unknown_X_Y_Z`格式的未知类别信息
- 输入通道数是否与模型匹配
- 是否有足够的内存和GPU资源
