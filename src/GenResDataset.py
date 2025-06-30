import numpy as np
import matplotlib.pyplot as plt
import torch
import resnet
import glob
import re
import os
from sklearn.model_selection import train_test_split
data_dir = "imageData"

def combineNpy(dataname):

    file_list = glob.glob(os.path.join(data_dir, f"{dataname}_*.npy"))
    
    # 正则模式，匹配 target_strength 值
    pattern = re.compile(rf"{re.escape(dataname)}_(-?\d+)\.npy")

    group1_values = []  # 数据
    group2_values = []
    group1_labels = []  # 标签
    group2_labels = []

    for file in file_list:
        match = pattern.search(os.path.basename(file))
        if match:
            strength = int(match.group(1))
            try:
                data = np.load(file) # shape: (250, 224, 224)
            except Exception as e:
                print(f" 加载文件失败: {file}\n异常信息: {e}")
                continue  
            #-20~10归为正常，30~50归为异常
            if -20 <= strength <= 10:
                group1_values.append(data)
                group1_labels.append(np.full(data.shape[0], "normal"))
            elif 20 <= strength <= 50:
                group2_values.append(data)
                group2_labels.append(np.full(data.shape[0], dataname))

    # 拼接数据
    group1_array = np.concatenate(group1_values, axis=0) if group1_values else np.empty((0, 224, 224))
    group2_array = np.concatenate(group2_values, axis=0) if group2_values else np.empty((0, 224, 224))
    all_data = np.concatenate([group1_array, group2_array], axis=0)

    # 拼接标签
    group1_labels = np.concatenate(group1_labels, axis=0) if group1_labels else np.empty((0,))
    group2_labels = np.concatenate(group2_labels, axis=0) if group2_labels else np.empty((0,))
    all_labels = np.concatenate([group1_labels, group2_labels], axis=0)

    return all_data, all_labels

filenameList=["CW","SCW(0.1023)","SCW(1.023)","AGWN(0.1023)","AGWN(1.023)",
              "PI(1ms_0.3)","PI(1ms_0.7)","PI(10ms_0.3)","PI(10ms_0.7)"]

def main():
    allData = []
    allLabels = []

    for filename in filenameList:
        print(f"process:{filename}")
        data, labels = combineNpy(dataname=filename)
        if data is None or data.shape[0] == 0:
            continue
        allData.append(data)
        allLabels.append(labels)

    # 合并所有数据
    allData = np.concatenate(allData, axis=0)  # shape: (N, 224, 224)
    allLabels = np.concatenate(allLabels, axis=0)  # shape: (N,)

    # 标签转换为整数
    label_names = ['normal'] + filenameList 
    label2idx = {label: idx for idx, label in enumerate(label_names)}
    allLabels_int = np.vectorize(label2idx.get)(allLabels)  # shape: (N,)

    # 划分训练测试集
    X_train, X_test, y_train, y_test = train_test_split(
        allData, allLabels_int, test_size=0.2, random_state=42, stratify=allLabels_int
    )
    savepath="jammingData/"
    # 保存训练集
    np.save(f"{savepath}X_train.npy", X_train)
    np.save(f"{savepath}y_train.npy", y_train)

    # 保存测试集
    np.save(f"{savepath}X_test.npy", X_test)
    np.save(f"{savepath}y_test.npy", y_test)

if __name__ == '__main__':
    main()