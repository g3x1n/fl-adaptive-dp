# fl-adaptive-dp

面向非独立同分布数据的自适应差分隐私联邦学习研究仓库。

当前仓库已经从“零散原型 + 调参记录”整理为“可直接执行最终实验”的结构，核心目标是：

- 在 `MNIST` 和 `CIFAR-10` 上统一比较 9 种方法
- 使用服务器视角白盒梯度反演攻击验证安全性
- 保留足够细的实验日志，便于后续绘图和论文写作
- 默认兼容远程 `Ubuntu + RTX4090`

## 最近更新

最近一轮主要补了两类改动：

- `CIFAR-10 IID` 底座增强
  - 本地优化器增加 `momentum / weight_decay / nesterov`
  - `CIFAR-10` 训练增强加入 `RandomCrop + RandomHorizontalFlip`
  - `CIFAR10CNN` 升级为带 `BatchNorm / Dropout` 的更强版本
  - `FedProx` 默认 `proximal_mu` 调整为更合理的 `0.01`
- 远程设备兼容修复
  - 修复 `FedNova + Adaptive DP` 在 `CUDA` 环境下可能出现的 `cpu/cuda` 混用聚合错误
  - 为 `FedProx` 的 proximal term 增加显式设备对齐保护

相关记录见：

- [CIFAR10-IID本地调优记录-20260331.md](/Users/gexinjin/project/fl-adaptive-dp/docs/CIFAR10-IID本地调优记录-20260331.md)

## 当前能力

目前已经具备：

- `MNIST / CIFAR-10` 数据集接入
- IID / Dirichlet Non-IID 划分
- `FedAvg / FedProx / FedNova`
- `Fixed DP`
- `Adaptive DP`
- 客户端感知裁剪、客户端感知噪声、风险感知聚合
- 黑盒 MIA
- 白盒梯度反演攻击
- 统一配置继承
- 统一训练日志、轮级日志、客户端级日志
- 远程 `Ubuntu + CUDA` 运行时适配

## 最终实验方法

正式实验统一采用 9 种方法：

1. `FedAvg`
2. `FedProx`
3. `FedNova`
4. `FedAvg + Fixed DP`
5. `FedProx + Fixed DP`
6. `FedNova + Fixed DP`
7. `FedAvg + Adaptive DP`
8. `FedProx + Adaptive DP`
9. `FedNova + Adaptive DP`

正式实验设计文档见：

- [最终实验设计-v2.md](/Users/gexinjin/project/fl-adaptive-dp/docs/最终实验设计-v2.md)

## 配置结构

正式实验配置统一放在：

- [configs/experiments/final](/Users/gexinjin/project/fl-adaptive-dp/configs/experiments/final)

历史调参、debug、旧实验配置统一归档到：

- [configs/experiments/legacy](/Users/gexinjin/project/fl-adaptive-dp/configs/experiments/legacy)

`final/` 下的结构是：

- `common/`
  - 公共运行时、数据集、训练参数、方法模板
- 各实验目录
  - 按实验名组织
  - 再按数据集组织
  - 每个数据集目录下固定 9 个方法配置

示例：

- [exp_a_iid](/Users/gexinjin/project/fl-adaptive-dp/configs/experiments/final/exp_a_iid)
- [exp_a_noniid_alpha_0_1](/Users/gexinjin/project/fl-adaptive-dp/configs/experiments/final/exp_a_noniid_alpha_0_1)
- [exp_d_privacy_tradeoff](/Users/gexinjin/project/fl-adaptive-dp/configs/experiments/final/exp_d_privacy_tradeoff)

配置通过 `inherits` 机制复用公共模板，配置加载逻辑在 [config.py](/Users/gexinjin/project/fl-adaptive-dp/src/utils/config.py)。

## 脚本结构

通用入口：

- [run_experiment.py](/Users/gexinjin/project/fl-adaptive-dp/scripts/run_experiment.py)
- [run_mia_experiment.py](/Users/gexinjin/project/fl-adaptive-dp/scripts/run_mia_experiment.py)
- [run_whitebox_attack.py](/Users/gexinjin/project/fl-adaptive-dp/scripts/run_whitebox_attack.py)

按实验批量运行的脚本统一放在：

- [scripts/experiments](/Users/gexinjin/project/fl-adaptive-dp/scripts/experiments)

例如：

- [run_exp_a_iid.sh](/Users/gexinjin/project/fl-adaptive-dp/scripts/experiments/run_exp_a_iid.sh)
- [run_exp_a_noniid_alpha_0_1.sh](/Users/gexinjin/project/fl-adaptive-dp/scripts/experiments/run_exp_a_noniid_alpha_0_1.sh)
- [run_exp_b_alpha_sensitivity.sh](/Users/gexinjin/project/fl-adaptive-dp/scripts/experiments/run_exp_b_alpha_sensitivity.sh)
- [run_exp_d_privacy_tradeoff.sh](/Users/gexinjin/project/fl-adaptive-dp/scripts/experiments/run_exp_d_privacy_tradeoff.sh)
- [run_exp_c_whitebox.sh](/Users/gexinjin/project/fl-adaptive-dp/scripts/experiments/run_exp_c_whitebox.sh)

历史脚本归档到：

- [scripts/legacy](/Users/gexinjin/project/fl-adaptive-dp/scripts/legacy)

## 环境准备

建议使用 `conda`，Python `3.10`：

```bash
conda create -n fl-adaptive-dp python=3.10
conda activate fl-adaptive-dp
pip install -r requirements.txt
```

如果是远程 `Ubuntu + RTX4090`，仓库已经支持：

- `device: auto` 自动优先选择 `cuda`
- CUDA 下自动启用 `pin_memory`
- `num_workers > 0` 时自动启用 `persistent_workers`
- `TF32` 与 `cudnn_benchmark`

详细说明见：

- [Ubuntu-RTX4090运行说明.md](/Users/gexinjin/project/fl-adaptive-dp/docs/Ubuntu-RTX4090运行说明.md)

如果远程机之前在 `FedNova + Adaptive DP` 上报过：

```text
Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu
```

请先同步到包含本次修复的最新提交后再重跑。

## 常用命令

检查数据集划分：

```bash
python scripts/inspect_data.py --config configs/datasets/mnist.yaml
```

运行单个正式配置：

```bash
python scripts/run_experiment.py \
  --config configs/experiments/final/exp_a_iid/mnist/fedavg.yaml
```

运行黑盒 MIA：

```bash
python scripts/run_mia_experiment.py \
  --config configs/experiments/final/exp_d_privacy_tradeoff/mnist/fednova_adaptive_dp.yaml
```

运行白盒梯度反演攻击：

```bash
python scripts/run_whitebox_attack.py \
  --baseline-config configs/experiments/final/exp_d_privacy_tradeoff/mnist/fednova.yaml \
  --dp-config configs/experiments/final/exp_d_privacy_tradeoff/mnist/fednova_adaptive_dp.yaml
```

按实验批量运行：

```bash
scripts/experiments/run_exp_a_iid.sh
scripts/experiments/run_exp_a_noniid_alpha_0_1.sh
scripts/experiments/run_exp_b_alpha_sensitivity.sh
scripts/experiments/run_exp_d_privacy_tradeoff.sh
scripts/experiments/run_exp_c_whitebox.sh
```

## 日志与输出

每次实验会在 `outputs/` 下生成一个时间戳目录，正式实验默认保留：

- `config.yaml`
- `summary.json`
- `metrics.csv`
- `round_summary.jsonl`
- `client_metrics.jsonl`
- `train.log`

用途建议：

- `metrics.csv`
  - 常规收敛曲线
- `round_summary.jsonl`
  - 逐轮详细统计
- `client_metrics.jsonl`
  - 客户端级箱线图、散点图、分布图

## 测试

运行测试：

```bash
pytest tests
```

当前已覆盖：

- 配置继承
- 设备与运行时参数解析
- `FedAvg / FedNova / FedProx`
- DP 裁剪、预算和调度
- 黑盒 MIA
- 白盒梯度反演

## 相关文档

- [最终实验设计-v2.md](/Users/gexinjin/project/fl-adaptive-dp/docs/最终实验设计-v2.md)
- [MIA安全性实验记录.md](/Users/gexinjin/project/fl-adaptive-dp/docs/MIA安全性实验记录.md)
- [白盒梯度反演攻击实验记录.md](/Users/gexinjin/project/fl-adaptive-dp/docs/白盒梯度反演攻击实验记录.md)
- [AdaptiveDP-NonIID三方法联合调参记录.md](/Users/gexinjin/project/fl-adaptive-dp/docs/AdaptiveDP-NonIID三方法联合调参记录.md)
- [CIFAR10-IID本地调优记录-20260331.md](/Users/gexinjin/project/fl-adaptive-dp/docs/CIFAR10-IID本地调优记录-20260331.md)
