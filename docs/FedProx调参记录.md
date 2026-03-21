# FedProx 调参记录

## 1. 调参目的

本轮调参的目标是验证：在已经改良过的公共训练参数下，`FedProx` 是否能够比 `FedAvg` 在强 Non-IID 场景中表现得更稳定或更优。

这一步非常关键，因为在前一轮基线实验中，`FedProx` 几乎与 `FedAvg` 重合，无法支撑“FedProx 对异构数据更友好”这一论点。为了避免把“公共参数太差”误判成“FedProx 没效果”，本轮调参采用了新的、更合理的公共参数底座。

## 2. 公共参数底座

本轮调参统一采用前一轮 `FedAvg + IID` 调参中效果较好的公共设置：

- 数据集：`MNIST`
- 数据分布：Dirichlet Non-IID
- `alpha = 0.1`
- `num_clients = 10`
- `rounds = 20`
- `local_epochs = 1`
- `batch_size = 64`
- `learning_rate = 0.05`
- `fraction_fit = 1.0`
- `seed = 42`

在这组公共参数下，参考组 `FedAvg + Non-IID` 已经明显优于最早那轮保守设置，因此更适合作为 `FedProx` 的调参底座。

## 3. 调参范围

参考 `FedProx` 官方仓库给出的常见 `mu` 建议范围，本轮优先测试以下值：

- `mu = 0.001`
- `mu = 0.01`
- `mu = 0.05`
- `mu = 0.1`
- `mu = 0.5`

其中 `mu = 0.5` 作为更强正则的扩展候选，用于观察较大近端约束是否能够进一步缓解客户端漂移。

## 4. 对照组与实验配置

### 4.1 对照组

- `FedAvg + Non-IID`
  - 配置文件：[tune_fedavg_noniid_r20_lr005_f10.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fedavg_noniid_r20_lr005_f10.yaml)
  - 结果目录：[20260318_151916_tune_fedavg_noniid_r20_lr005_f10](/Users/admin/Desktop/fl-adaptive-dp/outputs/20260318_151916_tune_fedavg_noniid_r20_lr005_f10)

### 4.2 FedProx 调参配置

- [tune_fedprox_noniid_mu0001.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fedprox_noniid_mu0001.yaml)
- [tune_fedprox_noniid_mu001.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fedprox_noniid_mu001.yaml)
- [tune_fedprox_noniid_mu005.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fedprox_noniid_mu005.yaml)
- [tune_fedprox_noniid_mu01.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fedprox_noniid_mu01.yaml)
- [tune_fedprox_noniid_mu05.yaml](/Users/admin/Desktop/fl-adaptive-dp/configs/experiments/tune_fedprox_noniid_mu05.yaml)

## 5. 当前已完成结果

### 5.1 FedAvg 参考组

| 方法 | 最佳轮次 | 最佳准确率 | 最终轮次 | 最终准确率 |
| --- | ---: | ---: | ---: | ---: |
| FedAvg | 18 | 87.95% | 20 | 84.65% |

### 5.2 FedProx 调参结果

| 方法 | `mu` | 最佳轮次 | 最佳准确率 | 当前最终轮次 | 当前最终准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FedProx | 0.001 | 18 | 87.95% | 20 | 84.70% |
| FedProx | 0.01 | 18 | 87.95% | 20 | 84.70% |
| FedProx | 0.05 | 18 | 87.60% | 20 | 85.85% |
| FedProx | 0.1 | 18 | 87.80% | 20 | 86.55% |
| FedProx | 0.5 | 20 | 88.05% | 20 | 88.05% |

说明：

- 以上结果均已完整跑完 20 轮。
- 从当前这组单次实验结果看，较大的 `mu` 并没有直接导致训练失效，反而在本次实验中带来了更高的最终准确率。

## 6. 结果分析

### 6.1 已经可以明确得到的结论

#### 结论 1：小 `mu` 值几乎等价于 FedAvg

从当前结果看：

- `mu = 0.001`
- `mu = 0.01`

与参考组 `FedAvg` 的最佳准确率和最终准确率几乎完全重合。

这说明在当前训练设置下，这两个 `mu` 值对本地训练轨迹的影响非常有限，近端项过弱，没有显著改变模型更新方向。

#### 结论 2：中等到较大的 `mu` 值开始体现出稳定性优势

当前最值得关注的是：

- `mu = 0.05`
- `mu = 0.1`
- `mu = 0.5`

它的当前结果显示：

- `mu = 0.05` 的最终准确率 `85.85%` 高于 `FedAvg` 的 `84.65%`
- `mu = 0.1` 的最终准确率 `86.55%`，进一步提升
- `mu = 0.5` 的最终准确率和最佳准确率都达到 `88.05%`，是当前最优结果

这意味着：

- 较强的近端约束没有明显提升早期峰值
- 但确实改善了训练后期的性能保持能力
- 在当前公共参数底座下，`FedProx` 的优势更多体现在“减缓后期回落”而不是“显著抬高中期峰值”

这一点很符合 `FedProx` 的设计初衷：它未必让模型峰值更高，但可能减少局部更新漂移带来的震荡。

#### 结论 3：当前这组实验里，`mu = 0.5` 反而是最优候选

从完整结果看：

- `mu = 0.001` 和 `mu = 0.01` 几乎等价于 `FedAvg`
- `mu = 0.05` 和 `mu = 0.1` 已经表现出一定稳定性收益
- `mu = 0.5` 在本次单次实验中取得了最高的最终准确率 `88.05%`

这说明：

- 之前担心“大 `mu` 会显著压制优化”的结论，在当前这组完整结果下并不成立
- 至少在 `MNIST + alpha=0.1 + rounds=20 + lr=0.05` 这组设置下，较大的 `mu` 是值得继续追踪的方向
- 但由于这仍然只是单随机种子结果，不能直接写成“`mu = 0.5` 普遍最优”的最终论文结论

因此，现阶段更合理的做法不是回避大 `mu`，而是把 `0.05 / 0.1 / 0.5` 都列为后续复验候选。

## 7. 当前阶段性判断

如果只基于当前已经完成的结果，比较合理的判断是：

1. `mu = 0.001` 和 `mu = 0.01` 太小，基本没有体现出 `FedProx` 的优势。
2. `mu = 0.05` 和 `mu = 0.1` 已经能带来比 `FedAvg` 更好的最终结果。
3. `mu = 0.5` 是当前这轮单次实验中的最优候选值。

因此，如果下一步需要先选一个候选值继续做正式实验，建议优先考虑：

- `mu = 0.5`

如果希望保守一点，也可以保留：

- `mu = 0.1`

作为“更稳妥的折中候选值”。

## 8. 论文写作建议

这轮调参目前更适合写成“参数敏感性分析的第一轮结果”，而不是“FedProx 已经显著优于 FedAvg”的最终证据。

建议论文中这样表述：

- 在改良后的公共参数设置下，FedProx 的性能对近端系数 `mu` 较为敏感。
- 当 `mu` 较小时，FedProx 与 FedAvg 的表现接近；
- 当 `mu` 增大到一定程度时，模型后期性能保持能力有所改善；
- 初步结果表明，较强的近端约束有可能缓解 Non-IID 场景下的客户端漂移问题，但这一现象仍需多随机种子进一步验证。

## 9. 后续建议

为了把这轮调参真正变成论文可引用的结论，建议接下来做两件事：

1. 对 `mu = 0.1` 和 `mu = 0.5` 做多随机种子复验  
   建议至少：
   - `seed = 42`
   - `seed = 123`
   - `seed = 2026`

2. 用当前候选 `FedProx` 参数，重新和 `FedAvg / FedNova` 做一次统一对比  
   统一使用：
   - `rounds = 20`
   - `learning_rate = 0.05`
   - `fraction_fit = 1.0`

这样就能把“参数调优”与“方法对比”两步明确分开，论文结构也会更清晰。
