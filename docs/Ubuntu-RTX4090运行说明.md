# Ubuntu RTX4090 运行说明

## 1. 目标

本仓库已经调整为可同时兼容：

- 本地 `Mac / Apple Silicon`
- 远程 `Ubuntu + NVIDIA RTX4090`

重点改动包括：

- `device: auto` 时优先选择 `cuda`
- 支持 `pin_memory`
- 支持 `persistent_workers`
- 支持 CUDA `TF32`
- 支持 `cudnn_benchmark`

## 2. 推荐运行时配置

在 `RTX4090` 机器上，建议优先使用：

```yaml
runtime:
  device: auto
  num_workers: 8
  pin_memory: auto
  persistent_workers: auto
  allow_tf32: true
  cudnn_benchmark: true
  matmul_precision: high
```

说明：

- `device: auto`
  会优先选择 `cuda`
- `pin_memory: auto`
  在 `cuda` 下自动启用
- `persistent_workers: auto`
  当 `num_workers > 0` 时自动启用
- `allow_tf32: true`
  对 `RTX4090` 这类 CUDA 设备通常有利于吞吐

## 3. 代码入口

这些入口已经适配新的运行时参数：

- `src/fl/runner.py`
- `scripts/run_mia_experiment.py`
- `scripts/run_whitebox_attack.py`

相关设备工具：

- `src/utils/device.py`

## 4. 配置建议

如果你要在远程机上跑正式实验，最少只需要改两类参数：

1. `runtime`
   - `device`
   - `num_workers`
   - `pin_memory`
   - `persistent_workers`

2. 输出目录与数据目录
   - `dataset.root`
   - `experiment.output_root`

例如：

```yaml
dataset:
  root: data

experiment:
  output_root: outputs
```

这样配置在 `Ubuntu` 上是可移植的，不依赖本地 `Mac` 路径。

## 5. 建议做法

在远程机上建议先跑一个小实验确认环境：

```bash
python scripts/run_experiment.py --config configs/experiments/fednova_mnist_debug.yaml
```

如果能正常识别 `cuda`，再继续跑正式实验。
