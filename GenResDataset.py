import numpy as np
import matplotlib.pyplot as plt
import torch
import resnet
import glob
import re
import os
from sklearn.model_selection import train_test_split
data_dir = "imageSet_3"
def make_empty_array(shape, dtype):
    return np.empty((0,) + shape, dtype=dtype) if shape else np.empty((0,), dtype=dtype)
def combineNpy(dataname, data_dir=data_dir,normList=[-20,-10,0,10],jamList=[20,30,40,50]):
    file_list = glob.glob(os.path.join(data_dir, f"{dataname}_*.npy"))
    pattern = re.compile(rf"{re.escape(dataname)}_(-?\d+)\.npy")

    group1_values = []
    group2_values = []
    group1_labels = []
    group2_labels = []

    expected_shape = None  # 用于检查通道一致性

    for file in file_list:
        match = pattern.search(os.path.basename(file))
        if not match:
            continue
        strength = int(match.group(1))
        try:
            data = np.load(file).astype(np.uint8)
        except Exception as e:
            print(f"加载文件失败: {file}\n异常信息: {e}")
            continue


        # 统一通道维度判断
        if expected_shape is None:
            expected_shape = data.shape[1:]  # 除了batch维度的形状
        else:
            if data.shape[1:] != expected_shape:
                print(f"跳过通道数不一致数据: {file} shape={data.shape} != expected {expected_shape}")
                continue

        if strength in normList:
            group1_values.append(data)
            group1_labels.append(np.full(data.shape[0], "normal"))
        elif strength in jamList:
            group2_values.append(data)
            group2_labels.append(np.full(data.shape[0], dataname))

    if group1_values:
        group1_array = np.concatenate(group1_values, axis=0)
    else:
        group1_array = make_empty_array(expected_shape, dtype=np.uint8)

    if group2_values:
        group2_array = np.concatenate(group2_values, axis=0)
    else:
        group2_array = make_empty_array(expected_shape, dtype=np.uint8)

    all_data = np.concatenate([group1_array, group2_array], axis=0)

    group1_labels = np.concatenate(group1_labels, axis=0) if group1_labels else np.empty((0,))
    group2_labels = np.concatenate(group2_labels, axis=0) if group2_labels else np.empty((0,))
    all_labels = np.concatenate([group1_labels, group2_labels], axis=0)

    return all_data, all_labels

filenameList=["CW","SCW(0.1023)","SCW(1.023)","AGWN(0.1023)","AGWN(1.023)",
              "PI(1ms_0.3)","PI(1ms_0.7)","PI(10ms_0.3)","PI(10ms_0.7)"]

def main():
    allData = []
    allLabels = []
    normalList=[-20,-10,0,10]#默认[-20,-10,0,10]
    jammingList=[30,40,50]#默认[20,30,40,50]
    tailname = "_".join(str(x) for x in jammingList)
    for filename in filenameList:
        print(f"process:{filename}")
        data, labels = combineNpy(dataname=filename,normList=normalList,jamList=jammingList)
        print(data.shape)
        if data is None or data.shape[0] == 0:
            continue
        allData.append(data)
        allLabels.append(labels)
    for i, d in enumerate(allData):
        print(f"Index {i}: shape {d.shape}")
        assert d.ndim == 4, f"数据维度不对！Index {i} 维度为 {d.ndim}"
    # 合并所有数据
    allData = np.concatenate(allData, axis=0)  # shape: (N, 224, 224)
    allLabels = np.concatenate(allLabels, axis=0)  # shape: (N,)

    # 标签转换为整数
    label_names = ['normal'] + filenameList 
    label2idx = {label: idx for idx, label in enumerate(label_names)}
    allLabels_int = np.vectorize(label2idx.get)(allLabels)  # shape: (N,)

    # 划分训练测试集
    # X_train, X_test, y_train, y_test = train_test_split(
    #     allData, allLabels_int, test_size=0.2, random_state=42, stratify=allLabels_int
    # )
    savepath="jammingData/"
    np.save(f"{savepath}data_3channel_{tailname}.npy", allData)
    np.save(f"{savepath}labels_3channel_{tailname}.npy", allLabels_int)
    # # 保存训练集
    # np.save(f"{savepath}X_train.npy", X_train)
    # np.save(f"{savepath}y_train.npy", y_train)

    # # 保存测试集
    # np.save(f"{savepath}X_test.npy", X_test)
    # np.save(f"{savepath}y_test.npy", y_test)

if __name__ == '__main__':
    main()