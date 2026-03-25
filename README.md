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
- 固定噪声差分隐私训练
- 可审计的自适应差分隐私调度
- `Top-k` 客户端更新压缩

## 当前推荐参数

下面这组参数是目前已经在 `MNIST` 上完成验证、可以作为后续正式实验起点的配置。

### 1. 公共训练底座

当前无隐私基线实验优先使用：

- `num_clients = 10`
- `rounds = 20`
- `local_epochs = 1`
- `batch_size = 64`
- `eval_batch_size = 256`
- `fraction_fit = 1.0`
- `seed = 42`
- `max_train_samples = 10000`
- `max_test_samples = 2000`

对于强异构实验，当前主设定为：

- `partition_mode = dirichlet`
- `dirichlet_alpha = 0.1`

### 2. 当前最优无隐私配置

`FedAvg`

- 推荐配置：
  - `algorithm = fedavg`
  - `learning_rate = 0.05`
  - `proximal_mu = 0.0`
- 当前已验证结果：
  - `MNIST + IID` 最终准确率 `94.40%`
  - `MNIST + Non-IID(alpha=0.1)` 最终准确率 `84.65%`

`FedProx`

- 推荐配置：
  - `algorithm = fedprox`
  - `learning_rate = 0.05`
  - `proximal_mu = 0.5`
- 当前已验证结果：
  - `MNIST + Non-IID(alpha=0.1)` 最终准确率 `88.05%`

`FedNova`

- 推荐配置：
  - `algorithm = fednova`
  - `learning_rate = 0.05`
  - `proximal_mu = 0.0`
- 当前已验证结果：
  - `MNIST + IID` 最终准确率 `94.40%`
  - `MNIST + Non-IID(alpha=0.1)` 最终准确率 `88.20%`

### 3. DP 与压缩的当前推荐起点

固定 DP：

- `dp_mode = fixed`
- `clip_norm = 1.0`
- `noise_multiplier = 0.4`
- `delta = 1e-5`

自适应 DP：

- `dp_mode = adaptive`
- `noise_schedule = round_based`
- `clip_norm = 1.0`
- `min_noise_multiplier = 0.2`
- `max_noise_multiplier = 0.8`
- `schedule_warmup_rounds = 1`

`Top-k` 压缩：

- `compression.mode = topk`
- `topk_ratio = 0.2`

说明：

- 上面这组 DP 与压缩参数目前是“可运行、可审计的推荐起点”
- 它们还不是最终论文里的最优参数，后续仍需要系统调参和正式对比实验

详细调参记录可参考：

- [FedAvg-IID调参记录.md](/Users/admin/Desktop/fl-adaptive-dp/docs/FedAvg-IID调参记录.md)
- [FedProx调参记录.md](/Users/admin/Desktop/fl-adaptive-dp/docs/FedProx调参记录.md)
- [FedNova调参记录.md](/Users/admin/Desktop/fl-adaptive-dp/docs/FedNova调参记录.md)
- [实验1-基线算法对比实验记录-正式版.md](/Users/admin/Desktop/fl-adaptive-dp/docs/实验1-基线算法对比实验记录-正式版.md)

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

运行固定噪声 DP 调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fedavg_mnist_fixed_dp_debug.yaml
```

运行自适应 DP 调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fedprox_mnist_adaptive_dp_debug.yaml
```

运行 `Top-k` 压缩调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fednova_mnist_topk_debug.yaml
```

运行自适应 DP + `Top-k` 联合调试实验：

```bash
python scripts/run_experiment.py --config configs/experiments/fednova_mnist_adaptive_dp_topk_debug.yaml
```

## 运行结果

每次实验会在 `outputs/` 下生成一个时间戳目录，包含：

- `config.yaml`：本次实验的完整配置
- `summary.json`：实验摘要与最终指标
- `metrics.csv`：逐轮训练与评估指标
- `train.log`：完整日志

当启用 DP 或压缩时，`metrics.csv` 还会额外记录：

- `epsilon_spent`
- `noise_multiplier`
- `clip_norm`
- `schedule_reason`
- `compression_ratio`
- `nnz_params`
- `pre_compression_payload_bytes`

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
- DP 梯度裁剪、预算累计与调度
- `Top-k` 压缩与更新恢复

## 下一步方向

接下来的重点是继续推进隐私模块：

- 固定噪声 DP 的批量实验
- 自适应差分隐私的正式对比实验
- 通信压缩与 DP 的组合实验
- 批量实验与论文图表导出
