# 交叉验证实验使用说明

## 文件说明

- `runCVtest.py`: 封装后的交叉验证主程序，包含main函数
- `run_cv_with_params.py`: 参数设置脚本，可以通过命令行或配置文件设置参数
- `config_example.json`: 配置文件示例

## 使用方法

### 方法1: 直接运行封装后的主程序

```bash
python runCVtest.py
```

这将使用默认参数运行实验：
- `num_unknown = 3`
- `num_cv = 5`

### 方法2: 使用参数设置脚本

#### 2.1 通过命令行参数

```bash
# 设置num_unknown=3, num_cv=10
python run_cv_with_params.py --num_unknown 3 --num_cv 10

# 设置自定义数据路径
python run_cv_with_params.py --num_unknown 3 --num_cv 10 --data_path "/path/to/your/data/"

# 查看所有可用参数
python run_cv_with_params.py --help
```

#### 2.2 通过配置文件

1. 复制并修改配置文件：
```bash
cp config_example.json my_config.json
```

2. 编辑 `my_config.json` 文件，设置你想要的参数：
```json
{
    "num_unknown": 3,
    "num_cv": 10,
    "data_path": "/workspace/OSR/jammingData/3channels/",
    "labels_file": "labels_3channel.npy",
    "data_file": "data_3channel.npy"
}
```

3. 使用配置文件运行：
```bash
python run_cv_with_params.py --config my_config.json
```

#### 2.3 保存当前配置

```bash
# 运行实验并保存当前配置
python run_cv_with_params.py --num_unknown 3 --num_cv 10 --save_config my_saved_config.json
```

## 参数说明

- `num_unknown`: 未知类别数量，默认为3
- `num_cv`: 交叉验证轮数，默认为10
- `data_path`: 数据文件所在目录路径
- `labels_file`: 标签文件名
- `data_file`: 数据文件名

## 输出结果

实验完成后会生成以下文件：

- `cv_results/cross_validation_results.npy`: 详细的交叉验证结果
- `images/cv_summary.png`: 交叉验证结果汇总图表
- `images/cv_round_*.png`: 各轮次详细结果图表
- `extractLog/`: 特征提取日志文件

## 注意事项

1. 确保数据文件路径正确
2. 确保有足够的磁盘空间存储结果
3. 实验可能需要较长时间，建议在后台运行
4. 如果使用GPU，确保CUDA环境配置正确

