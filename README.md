# fl-adaptive-dp

面向非独立同分布数据的自适应差分隐私联邦学习研究仓库。

当前仓库以“研究原型可复现”为目标，已经完成了基础工程搭建、数据集接入与划分、以及 `FedAvg / FedProx / FedNova` 三条最小联邦训练闭环。

## 当前进度

目前已经具备以下能力：

- 统一的 YAML 配置加载与实验输出目录管理
- `MNIST` 与 `CIFAR-10` 数据集接入
- IID / Dirichlet Non-IID 客户端划分
- 客户端标签分布统计与检查脚本
- 最小可运行的 `FedAvg`
- 最小可运行的 `FedProx`
- 最小可运行的 `FedNova`

## 目录结构

```text
fl-adaptive-dp/
├── configs/          # 数据、训练与实验配置
├── data/             # 本地下载的数据集
├── docs/             # 开题报告与规划文档
├── outputs/          # 实验日志、指标与结果
├── scripts/          # 运行和检查脚本
├── src/              # 核心代码
├── tests/            # 单元测试
├── README.md
├── requirements.txt
└── .gitignore
```

## 环境准备

建议使用 `conda`，Python 版本为 `3.10`：

```bash
conda create -n fl-adaptive-dp python=3.10
conda activate fl-adaptive-dp
pip install -r requirements.txt
```

如果已经创建过环境，只需要：

```bash
conda activate fl-adaptive-dp
```

## 当前可用脚本

检查数据集加载与 Non-IID 划分：

```bash
python scripts/inspect_data.py --config configs/datasets/mnist.yaml
```

运行最小 `FedAvg` 调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fedavg_mnist_debug.yaml
```

运行最小 `FedProx` 调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fedprox_mnist_debug.yaml
```

运行最小 `FedNova` 调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fednova_mnist_debug.yaml
```

## 运行结果

每次实验会在 `outputs/` 下生成一个时间戳目录，包含：

- `config.yaml`：本次实验的完整配置
- `summary.json`：实验摘要与最终指标
- `metrics.csv`：逐轮训练与评估指标
- `train.log`：完整日志

## 当前测试

运行当前测试集：

```bash
pytest tests
```

目前已覆盖：

- 数据划分正确性
- `FedAvg` 聚合逻辑
- `FedProx` 近端项
- `FedNova` 归一化聚合

## 下一步方向

接下来的重点是继续推进隐私模块：

- 固定噪声 DP
- 自适应差分隐私
- 通信压缩
- 批量实验与论文图表导出

