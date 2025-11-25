#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交叉验证实验参数设置脚本
可以通过命令行参数或配置文件设置num_unknown和num_cv参数
"""

import argparse
import json
import os
import sys
from runCVtest import run_cross_validation


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='运行交叉验证实验')
    
    parser.add_argument('--num_unknown', type=int, default=3,
                       help='未知类别数量 (默认: 3)')
    parser.add_argument('--num_cv', type=int, default=10,
                       help='交叉验证轮数 (默认: 10)')
    parser.add_argument('--data_path', type=str, 
                       default="/workspace/OSR/jammingData/3channels/",
                       help='数据文件路径')
    parser.add_argument('--labels_file', type=str, default="labels_3channel.npy",
                       help='标签文件名')
    parser.add_argument('--data_file', type=str, default="data_3channel.npy",
                       help='数据文件名')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径 (JSON格式)')
    parser.add_argument('--save_config', type=str, default=None,
                       help='保存当前配置到指定文件')
    
    return parser.parse_args()


def load_config(config_path):
    """从配置文件加载参数"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"错误: 配置文件 {config_path} 不存在")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"错误: 配置文件 {config_path} 格式错误")
        sys.exit(1)


def save_config(config, save_path):
    """保存配置到文件"""
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"配置已保存到: {save_path}")
    except Exception as e:
        print(f"保存配置失败: {e}")


def main():
    """主函数"""
    args = parse_arguments()
    
    # 如果指定了配置文件，从配置文件加载参数
    if args.config:
        config = load_config(args.config)
        # 使用配置文件中的参数，命令行参数会覆盖配置文件
        num_unknown = config.get('num_unknown', args.num_unknown)
        num_cv = config.get('num_cv', args.num_cv)
        data_path = config.get('data_path', args.data_path)
        labels_file = config.get('labels_file', args.labels_file)
        data_file = config.get('data_file', args.data_file)
    else:
        # 使用命令行参数
        num_unknown = args.num_unknown
        num_cv = args.num_cv
        data_path = args.data_path
        labels_file = args.labels_file
        data_file = args.data_file
    
    # 打印实验参数
    print("=" * 50)
    print("交叉验证实验参数设置")
    print("=" * 50)
    print(f"未知类别数量: {num_unknown}")
    print(f"交叉验证轮数: {num_cv}")
    print(f"数据路径: {data_path}")
    print(f"标签文件: {labels_file}")
    print(f"数据文件: {data_file}")
    print("=" * 50)
    
    # 检查数据文件是否存在
    labels_path = os.path.join(data_path, labels_file)
    data_file_path = os.path.join(data_path, data_file)
    
    if not os.path.exists(labels_path):
        print(f"错误: 标签文件不存在 - {labels_path}")
        sys.exit(1)
    
    if not os.path.exists(data_file_path):
        print(f"错误: 数据文件不存在 - {data_file_path}")
        sys.exit(1)
    
    # 如果指定了保存配置，保存当前配置
    if args.save_config:
        config = {
            'num_unknown': num_unknown,
            'num_cv': num_cv,
            'data_path': data_path,
            'labels_file': labels_file,
            'data_file': data_file
        }
        save_config(config, args.save_config)
    
    # 确认开始实验
    print("\n准备开始交叉验证实验...")
    response = input("是否继续? (y/N): ").strip().lower()
    if response not in ['y', 'yes', '是']:
        print("实验已取消")
        sys.exit(0)
    
    # 运行交叉验证实验
    try:
        print("\n开始运行交叉验证实验...")
        results = run_cross_validation(
            num_unknown=num_unknown,
            num_cv=num_cv,
            data_path=data_path,
            labels_file=labels_file,
            data_file=data_file
        )
        
        print("\n" + "=" * 50)
        print("实验完成!")
        print("=" * 50)
        print(f"平均AUROC: {results['mean_auroc']:.4f} ± {results['std_auroc']:.4f}")
        print(f"各轮次AUROC: {[f'{r['auroc']:.4f}' for r in results['individual_results']]}")
        
    except Exception as e:
        print(f"实验运行出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

