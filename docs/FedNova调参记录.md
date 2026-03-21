# FedNova 调参记录

## 1. 调参目的

本轮调参的目标是验证：在已经改良过的公共训练参数下，`FedNova` 是否能够在强 Non-IID 场景中明显优于此前的保守设置，并找到适合作为后续正式对比实验的推荐工作点。

之前的基线实验中，`FedNova` 在 `MNIST + Non-IID(alpha=0.1)` 下最终只有 `48.70%`，但其中间轮次一度达到更高峰值，说明它不是完全无效，而是很可能对训练参数更敏感。因此，这轮调参重点考察学习率、训练轮次和本地训练强度。

## 2. 公共实验设置

除被调参数外，本轮实验统一采用：

- 数据集：`MNIST`
- 数据分布：Dirichlet Non-IID
- `alpha = 0.1`
- `num_clients = 10`
- `batch_size = 64`
- `fraction_fit = 1.0`
- `seed = 42`
- 模型：`mnist_cnn`

这样做的目的是尽量与最近的 `FedAvg / FedProx` 调参底座保持一致，避免不同算法使用不同公共设置导致结论失真。

## 3. 调参范围

本轮选择了 5 组高价值配置：

1. 固定 `rounds = 20`、`local_epochs = 1`，扫描学习率：
   - `learning_rate = 0.01`
   - `learning_rate = 0.02`
   - `learning_rate = 0.05`
2. 固定 `learning_rate = 0.02`，测试更长训练：
   - `rounds = 30`
3. 固定 `learning_rate = 0.02`，测试更强本地训练：
   - `local_epochs = 2`

对应配置文件：

- [tune_fednova_noniid_r20_lr001_f10.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fednova_noniid_r20_lr001_f10.yaml)
- [tune_fednova_noniid_r20_lr002_f10.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fednova_noniid_r20_lr002_f10.yaml)
- [tune_fednova_noniid_r20_lr005_f10.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fednova_noniid_r20_lr005_f10.yaml)
- [tune_fednova_noniid_r30_lr002_f10.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fednova_noniid_r30_lr002_f10.yaml)
- [tune_fednova_noniid_r20_lr002_f10_e2.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fednova_noniid_r20_lr002_f10_e2.yaml)

## 4. 实验结果汇总

| 配置 | 最佳轮次 | 最佳准确率 | 最终轮次 | 最终准确率 |
| --- | ---: | ---: | ---: | ---: |
| `r20, lr=0.01, e1` | 20 | 81.15% | 20 | 81.15% |
| `r20, lr=0.02, e1` | 16 | 83.50% | 20 | 83.45% |
| `r20, lr=0.05, e1` | 17 | 89.60% | 20 | 88.20% |
| `r30, lr=0.02, e1` | 30 | 87.25% | 30 | 87.25% |
| `r20, lr=0.02, e2` | 20 | 88.05% | 20 | 88.05% |

对应结果目录：

- [20260321_152142_tune_fednova_noniid_r20_lr001_f10](/Users/admin/Desktop/fl-adaptive-dp/outputs/20260321_152142_tune_fednova_noniid_r20_lr001_f10)
- [20260321_152256_tune_fednova_noniid_r20_lr002_f10](/Users/admin/Desktop/fl-adaptive-dp/outputs/20260321_152256_tune_fednova_noniid_r20_lr002_f10)
- [20260321_152413_tune_fednova_noniid_r20_lr005_f10](/Users/admin/Desktop/fl-adaptive-dp/outputs/20260321_152413_tune_fednova_noniid_r20_lr005_f10)
- [20260321_152530_tune_fednova_noniid_r30_lr002_f10](/Users/admin/Desktop/fl-adaptive-dp/outputs/20260321_152530_tune_fednova_noniid_r30_lr002_f10)
- [20260321_152724_tune_fednova_noniid_r20_lr002_f10_e2](/Users/admin/Desktop/fl-adaptive-dp/outputs/20260321_152724_tune_fednova_noniid_r20_lr002_f10_e2)

## 5. 结果分析

### 5.1 学习率是当前最关键的参数

这轮结果最明显的结论是：`FedNova` 在当前实现中非常吃学习率。

- `lr = 0.01` 只能到 `81.15%`
- `lr = 0.02` 提升到 `83.45%`
- `lr = 0.05` 直接提升到 `88.20%`

也就是说，之前 `FedNova` 表现不理想，并不只是算法本身问题，更大可能是因为原始设置过于保守。

### 5.2 拉长轮次有帮助，但不如直接把学习率调对

把 `lr = 0.02` 的训练从 `20` 轮拉到 `30` 轮后，最终准确率从 `83.45%` 提升到 `87.25%`，这说明 `FedNova` 确实受益于更充分的全局收敛过程。

但即便如此，它仍然没有超过：

- `r20, lr=0.05, e1` 的 `88.20%`

这说明现阶段对 `FedNova` 来说，“合适的学习率”比“单纯拉长轮次”更关键。

### 5.3 增大本地训练强度也有收益

在 `lr = 0.02` 下，把 `local_epochs` 从 `1` 提高到 `2`，最终准确率从 `83.45%` 提升到 `88.05%`。

这说明：

- `FedNova` 的归一化聚合确实有能力承接更强的本地训练
- 在一定程度上，它比普通加权平均更能抵抗客户端本地步数带来的偏移

不过从单次实验结果看，`e2` 虽然已经非常接近最优，但仍略低于 `lr=0.05, e1`。

## 6. 当前阶段性结论

基于这轮已完成实验，可以得到比较清晰的阶段性判断：

1. 之前 `FedNova` 在 Non-IID 下的低性能，主要与参数过于保守有关。
2. 在当前实现中，`FedNova` 更适合较积极的学习率设置。
3. `learning_rate = 0.05`、`rounds = 20`、`local_epochs = 1` 是当前最优工作点。
4. `learning_rate = 0.02` 配合更长轮次或更大的本地 epoch 也能达到较强结果，但综合效率和最终性能，仍略逊于最优配置。

## 7. 推荐配置

如果下一步要进入正式对比实验，当前建议直接使用：

- [fednova_mnist_noniid_best.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/fednova_mnist_noniid_best.yaml)

对应参数为：

- `algorithm = fednova`
- `rounds = 20`
- `learning_rate = 0.05`
- `local_epochs = 1`
- `fraction_fit = 1.0`
- `alpha = 0.1`

## 8. 与现有基线的关系

这轮调参后，`FedNova` 在 `MNIST + Non-IID(alpha=0.1)` 下的表现已经明显好于它自己早期的保守设置：

- 早期基线最终准确率：`48.70%`
- 当前最优最终准确率：`88.20%`

这说明之前文档里“FedNova 有潜力但稳定性不足”的判断是合理的，但那个判断当时更多反映的是“参数未调开”，而不是算法上限。

## 9. 后续建议

为了让这组结果更适合直接进入论文正文，建议下一步做两件事：

1. 用当前最优 `FedNova` 配置做多随机种子复验  
   建议至少：
   - `seed = 42`
   - `seed = 123`
   - `seed = 2026`

2. 用统一的公共数据设置，重新比较：
   - `FedAvg`
   - `FedProx`
   - `FedNova`

其中：

- `FedAvg` 继续使用已经调好的公共底座
- `FedProx` 使用当前最优或次优 `mu`
- `FedNova` 使用本轮推荐配置

这样你就可以把“算法本身差异”与“参数是否合理”分开，论文说服力会强很多。
