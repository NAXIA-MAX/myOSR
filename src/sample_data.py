#!/usr/bin/env python3
import numpy as np

def main():
    # 加载数据
    data = np.load("/workspace/OSR/jammingData/3channels/data_3channel.npy")
    labels = np.load("/workspace/OSR/jammingData/3channels/labels_3channel.npy")
    
    # 找到标签0的索引
    class0_indices = np.where(labels == 0)[0]
    other_indices = np.where(labels != 0)[0]
    
    # 随机采样1000张标签0的数据
    np.random.seed(42)
    sample_size = min(1000, len(class0_indices))
    sampled_class0_indices = np.random.choice(class0_indices, size=sample_size, replace=False)
    
    # 组合数据：采样的标签0数据 + 所有其他标签数据
    final_indices = np.concatenate([sampled_class0_indices, other_indices])
    final_data = data[final_indices]
    final_labels = labels[final_indices]
    
    # 保存最终数据
    np.save("/workspace/OSR/jammingData/3channels/data_3channel_sampled.npy", final_data)
    np.save("/workspace/OSR/jammingData/3channels/labels_3channel_sampled.npy", final_labels)
    
    print(f"采样完成: 标签0采样{sample_size}张，其他标签保留，总计{len(final_data)}张数据")

if __name__ == "__main__":
    main()
