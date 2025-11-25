#!/usr/bin/env python3
"""
KNN-OSR测试运行示例脚本
这个脚本展示了如何使用test_knn_osr.py测试已训练模型的开放集识别效果
"""

import subprocess
import sys
import os

def run_knn_test():
    """运行KNN-OSR测试的示例"""
    
    # 配置参数
    model_path = "/workspace/OSR/trainedPth/trainedCVpth/resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth"
    
    # 完整数据集路径 - 包含所有类别(0-9)的数据
    data_path = "data/all_images.npy"  # 完整数据集
    label_path = "data/all_labels.npy"  # 完整标签集
    
    # 其他参数
    input_channel = 3
    k = 50
    batch_size = 64
    
    # 检查文件是否存在
    files_to_check = [model_path, data_path, label_path]
    for file_path in files_to_check:
        if not os.path.exists(file_path):
            print(f"警告: 文件不存在 - {file_path}")
            print("请确保所有数据文件都存在，或修改路径配置")
            return False
    
    # 构建命令
    cmd = [
        sys.executable, "test_knn_osr.py",
        "--model_path", model_path,
        "--data_path", data_path,
        "--label_path", label_path,
        "--input_channel", str(input_channel),
        "--k", str(k),
        "--batch_size", str(batch_size)
    ]
    
    print("运行KNN-OSR测试...")
    print(f"命令: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        # 运行测试
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("测试输出:")
        print(result.stdout)
        
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        print("-" * 50)
        print("测试完成!")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"测试失败，返回码: {e.returncode}")
        print("标准输出:")
        print(e.stdout)
        print("错误输出:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"运行测试时发生错误: {e}")
        return False

def create_test_data_structure():
    """创建测试所需的目录结构"""
    directories = [
        "data",
        "results",
        "extractLog"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"创建目录: {directory}")
        else:
            print(f"目录已存在: {directory}")

def print_usage_instructions():
    """打印使用说明"""
    print("=" * 60)
    print("KNN-OSR测试使用说明")
    print("=" * 60)
    print()
    print("1. 数据准备:")
    print("   - 确保已训练模型文件存在")
    print("   - 准备完整数据集: 包含所有类别(0-9)的images和labels的.npy文件")
    print("   - 系统会根据模型文件名自动拆分ID和OOD数据")
    print()
    print("2. 模型文件名格式:")
    print("   resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth")
    print("   - 未知类别: 5, 7, 6")
    print("   - 已知类别: 0, 1, 2, 3, 4, 8, 9")
    print()
    print("3. 数据拆分:")
    print("   - 系统自动根据模型文件名拆分数据")
    print("   - ID数据: 已知类别的数据，进一步分为训练集和测试集")
    print("   - OOD数据: 未知类别的数据")
    print()
    print("4. 标签映射:")
    print("   原始标签 -> 训练标签")
    print("   0 -> 0, 1 -> 1, 2 -> 2, 3 -> 3, 4 -> 4")
    print("   8 -> 5, 9 -> 6")
    print()
    print("5. 运行测试:")
    print("   python test_knn_osr.py --model_path <模型路径> \\")
    print("                           --data_path <完整数据路径> \\")
    print("                           --label_path <完整标签路径>")
    print()
    print("5. 输出结果:")
    print("   - AUROC指标")
    print("   - 分数分布图")
    print("   - ROC曲线图")
    print("   - 结果保存在results/目录下")
    print()

if __name__ == "__main__":
    print_usage_instructions()
    
    # 创建目录结构
    create_test_data_structure()
    
    print("\n是否运行测试? (y/n): ", end="")
    response = input().strip().lower()
    
    if response in ['y', 'yes', '是']:
        success = run_knn_test()
        if success:
            print("\n测试成功完成!")
        else:
            print("\n测试失败，请检查配置和数据文件")
    else:
        print("测试已取消")
        print("\n手动运行命令示例:")
        print("python test_knn_osr.py \\")
        print("  --model_path /workspace/OSR/trainedPth/trainedCVpth/resnet20_3channels_7classes_None_cv0_unknown_5_7_6.pth \\")
        print("  --data_path data/all_images.npy \\")
        print("  --label_path data/all_labels.npy")
