# FedAvg IID 调参记录

## 1. 目标

本次调参的目标是先将 `FedAvg + MNIST + IID` 调整到一个合理的基线水平，为后续：

- `FedProx / FedNova` 对比
- 固定 DP
- 自适应 DP
- 通信压缩

提供一个更可信的公共参数底座。

## 2. 初始问题

在第一轮基线实验中，`FedAvg + MNIST + IID` 的最终准确率约为 `79.95%`，偏低，说明当时的公共参数设置过于保守，不适合作为后续所有实验的基线。

初始设置大致为：

- `rounds = 10`
- `learning_rate = 0.01`
- `fraction_fit = 0.5`
- `local_epochs = 1`
- `max_train_samples = 10000`
- `max_test_samples = 2000`

## 3. 本轮调参思路

优先调对收敛影响最大的公共参数：

- `rounds`
- `learning_rate`
- `fraction_fit`
- `local_epochs`

其中：

- `fraction_fit` 先提高到 `1.0`，避免客户端采样带来的额外波动
- `learning_rate` 主要比较 `0.01 / 0.02 / 0.05`
- `rounds` 主要比较 `20 / 30`
- `local_epochs` 额外测试 `2`

## 4. 当前已完成的结果

### 4.1 已完成配置

| 配置名 | 主要参数 | 当前结果 |
| --- | --- | --- |
| `tune_fedavg_iid_r20_lr001_f10` | `rounds=20, lr=0.01, fraction_fit=1.0` | 最终 `86.30%` |
| `tune_fedavg_iid_r20_lr002_f10` | `rounds=20, lr=0.02, fraction_fit=1.0` | 最终 `89.95%` |
| `tune_fedavg_iid_r20_lr005_f10` | `rounds=20, lr=0.05, fraction_fit=1.0` | 最终 `94.40%` |

### 4.2 仍在后台运行或尚未完成收集的配置

- `tune_fedavg_iid_r30_lr002_f10`
- `tune_fedavg_iid_r30_lr002_f05`
- `tune_fedavg_iid_r20_lr002_f10_e2`

## 5. 当前阶段性结论

### 5.1 明确成立的结论

1. 将 `fraction_fit` 从 `0.5` 提高到 `1.0` 后，训练明显更稳定。
2. 将 `rounds` 从 `10` 提高到 `20` 后，准确率显著提升。
3. 在当前实现中，`learning_rate = 0.05` 明显优于 `0.01` 和 `0.02`。

### 5.2 当前最佳工作点

截至目前，表现最好的参数组合是：

- `rounds = 20`
- `learning_rate = 0.05`
- `fraction_fit = 1.0`
- `local_epochs = 1`

在 `MNIST + IID + 10000/2000` 的设置下，最终准确率达到：

- `94.40%`

这已经明显优于第一轮实验中的 `79.95%`。

## 6. 当前建议

基于目前结果，建议将 `FedAvg + IID` 的公共参数暂时更新为：

- `rounds = 20`
- `learning_rate = 0.05`
- `fraction_fit = 1.0`
- `local_epochs = 1`

并将其作为：

- 后续 `FedProx / FedNova` 调参的起点
- 后续隐私实验的默认无隐私基线

## 7. 后续建议

在当前工作点基础上，建议继续补两类确认：

1. 更大数据量确认  
   例如 `max_train_samples = 20000`、`max_test_samples = 5000`

2. 多随机种子确认  
   例如：
   - `seed = 42`
   - `seed = 123`
   - `seed = 2026`

这样可以避免最终论文结果过度依赖单次运行。
