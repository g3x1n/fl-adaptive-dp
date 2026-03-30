# Adaptive DP + Non-IID 三方法联合调参记录

## 1. 本次目标

本轮工作的目标是先在 `MNIST + Dirichlet Non-IID(alpha=0.1)` 条件下，将项目中的自适应差分隐私机制分别接入：

- `FedAvg`
- `FedProx`
- `FedNova`

并在统一实验底座下做一轮小规模调参，判断：

1. 当前哪种方法最适合作为后续 `adaptive DP` 主线实验底座
2. 当前自适应噪声调度的参数范围是否合理

## 2. 运行环境

- `conda` 环境：`py310`
- Python 版本：`3.10.20`
- 运行命令统一使用：

```bash
conda run -n py310 python scripts/run_experiment.py --config <config_path>
```

## 3. 本轮先修复的工程问题

在正式跑实验前，项目存在两个会直接阻塞实验的问题：

### 3.1 缺失 `src.data` 模块

训练入口 `scripts/run_experiment.py` 启动后会在 `src.fl.runner` 中导入 `src.data`，但仓库里该模块缺失，导致所有实验无法运行。

本轮已补齐以下数据模块：

- `src/data/__init__.py`
- `src/data/datasets.py`
- `src/data/partition.py`
- `src/data/statistics.py`

补充内容包括：

- `MNIST / CIFAR-10` 数据集加载
- IID / Dirichlet 划分
- 客户端子集与 `DataLoader` 构建
- 客户端标签分布统计

### 3.2 服务器聚合时的设备不一致

在 Apple Silicon 的 `mps` 设备下，客户端更新是 `cpu` 张量，而服务器全局模型参数在 `mps` 上，`apply_model_update()` 会因为设备不一致报错。

本轮已修复：

- `src/compression/topk.py`

修复方式：

- 在应用聚合更新前，将更新张量对齐到全局参数所在设备与数据类型

### 3.3 已验证的测试

本轮额外执行并通过：

- `pytest tests/test_data_partition.py tests/test_privacy.py -q`
- `pytest tests/test_compression.py tests/test_fedavg.py tests/test_fednova.py -q`

## 4. 统一实验设置

除被调参数外，三种方法统一采用以下设置：

- 数据集：`MNIST`
- 数据划分：`Dirichlet Non-IID`
- `alpha = 0.1`
- `num_clients = 10`
- `rounds = 20`
- `local_epochs = 1`
- `batch_size = 64`
- `eval_batch_size = 256`
- `fraction_fit = 1.0`
- `max_train_samples = 10000`
- `max_test_samples = 2000`
- 压缩：关闭，即 `compression.mode = none`
- `delta = 1e-5`
- `clip_norm = 1.0`
- `accountant = gaussian`
- 自适应噪声调度：`round_based`

算法专属设置：

- `FedAvg`: `lr = 0.05`
- `FedProx`: `lr = 0.05, mu = 0.5`
- `FedNova`: `lr = 0.05`

## 5. 调参过程

### 5.1 第一轮探测：激进调度 A

为了先判断当前 adaptive DP 的默认推荐范围是否可直接用于正式实验，先以 `FedAvg` 作为探测算法测试一组较激进的调度：

- `min_noise_multiplier = 0.2`
- `max_noise_multiplier = 0.8`
- `schedule_warmup_rounds = 1`

配置文件：

- `configs/experiments/tune_adp_fedavg_noniid_a.yaml`

输出目录：

- `outputs/20260326_104321_tune_adp_fedavg_noniid_a`

结果：

- 最佳准确率：`45.20%`，出现在第 `3` 轮
- 最终准确率：`14.20%`

结论：

- 这组调度在当前 `Non-IID` 条件下过于激进
- 随着噪声从 `0.2` 逐步升到 `0.8`，模型后半程明显失稳
- 这组参数不适合作为三种方法的正式对比底座

### 5.2 第二轮正式对比：保守调度 C

根据第一轮结果，将自适应噪声范围整体下调，改为更保守的调度：

- `min_noise_multiplier = 0.02`
- `max_noise_multiplier = 0.2`
- `schedule_warmup_rounds = 2`

然后在三种方法上统一使用该调度进行正式对比。

对应配置文件：

- `configs/experiments/tune_adp_fedavg_noniid_c.yaml`
- `configs/experiments/tune_adp_fedprox_noniid_c.yaml`
- `configs/experiments/tune_adp_fednova_noniid_c.yaml`

对应输出目录：

- `outputs/20260326_104533_tune_adp_fedavg_noniid_c`
- `outputs/20260326_104631_tune_adp_fedprox_noniid_c`
- `outputs/20260326_104740_tune_adp_fednova_noniid_c`

## 6. 结果汇总

### 6.1 激进调度 A 的探测结果

| 方法 | 调度 | 最佳轮次 | 最佳准确率 | 最终准确率 | 最终损失 | 最终 epsilon |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FedAvg | A | 3 | 45.20% | 14.20% | 22.4884 | 129.1613 |

### 6.2 保守调度 C 的正式对比结果

| 方法 | 最佳轮次 | 最佳准确率 | 最终准确率 | 最终损失 | 最终 epsilon | 最终噪声 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| FedAvg | 10 | 73.90% | 70.90% | 0.8688 | 900.6098 | 0.2000 |
| FedProx | 10 | 74.00% | 65.80% | 1.0010 | 900.6098 | 0.2000 |
| FedNova | 15 | 76.20% | 71.70% | 0.8113 | 900.6098 | 0.2000 |

## 7. 结果分析

### 7.1 当前最优方法：`FedNova + adaptive DP`

在保守调度 `C` 下，`FedNova` 是当前表现最好的组合：

- 最佳准确率最高：`76.20%`
- 最终准确率最高：`71.70%`
- 最终损失最低：`0.8113`

这说明在当前实现和当前参数底座下，`FedNova` 对 adaptive DP 注入的扰动更稳，后期也比 `FedAvg` 和 `FedProx` 更能维持性能。

### 7.2 `FedAvg` 次优，但稳定性尚可

`FedAvg + adaptive DP` 的最终准确率达到 `70.90%`，与 `FedNova` 接近，但峰值略低，后期震荡也更明显一些。

这说明：

- `FedAvg` 不是不能与 adaptive DP 结合
- 但在强 Non-IID 条件下，它的上限和稳定性略逊于 `FedNova`

### 7.3 `FedProx` 在当前 adaptive DP 设置下不占优

`FedProx` 在无 DP 情况下本来是有竞争力的，但在本轮 adaptive DP 对比中：

- 中前期能达到 `74.00%`
- 后期回落更明显
- 最终只保留到 `65.80%`

这表明当前 `mu = 0.5` 与当前 adaptive 噪声调度的组合还不够理想。问题不一定出在 `FedProx` 本身，也可能是：

- `FedProx` 对当前噪声递增节奏更敏感
- 或者需要重新调 `mu`

因此，现阶段不建议直接把 `FedProx + adaptive DP` 作为主线方案。

### 7.4 一个非常重要的现象：当前 `epsilon` 很高

虽然 `C` 方案的模型可用性明显优于 `A`，但当前实验输出的最终 `epsilon` 约为 `900.61`，数值非常大。

这说明：

- 当前使用的轻量会计方式会在低噪声区间下累计出较大的预算
- 当前 `min_noise = 0.02, max_noise = 0.2` 更偏向“保模型性能”，而不是“强隐私”

因此，这轮实验更适合回答：

- 三种方法在同一 adaptive DP 机制下谁更抗噪声

而不适合直接作为论文中“强隐私预算下的最终结论”。

## 8. 当前结论

基于本轮结果，当前阶段可得出以下结论：

1. 自适应噪声从 `0.2 -> 0.8` 的激进调度不适合当前 `MNIST + Non-IID(alpha=0.1)` 任务，会显著破坏模型后期训练。
2. 将调度收窄到 `0.02 -> 0.2` 后，三种方法都能稳定完成训练并得到可比较结果。
3. 在统一 adaptive DP 口径下，当前表现最好的是 `FedNova + adaptive DP`。
4. `FedAvg + adaptive DP` 是可以接受的备选方案。
5. `FedProx + adaptive DP` 在当前参数组合下不如前两者稳定，后续若要继续保留，建议单独调 `mu`。

## 9. 对后续实验的建议

### 9.1 作为后续主线底座的建议

当前建议优先选择：

- `FedNova + adaptive DP`

作为后续固定 DP / adaptive DP 主对比实验的候选底座。

### 9.2 下一步最值得做的 3 件事

1. 以 `FedNova` 为主线，再做一轮更细的 adaptive DP 调参  
   重点可考察：
   - `min_noise_multiplier`
   - `max_noise_multiplier`
   - `schedule_warmup_rounds`

2. 重新压低最终 `epsilon`  
   当前可尝试：
   - 增大噪声下界
   - 提高整体噪声范围
   - 或调整会计口径的展示方式

3. 将本轮最优 `FedNova + adaptive DP` 与无 DP / fixed DP 做正式对比  
   这是后续论文最关键的一步。

## 10. 本轮涉及的关键文件

新增配置：

- `configs/experiments/tune_adp_fedavg_noniid_a.yaml`
- `configs/experiments/tune_adp_fedavg_noniid_b.yaml`
- `configs/experiments/tune_adp_fedavg_noniid_c.yaml`
- `configs/experiments/tune_adp_fedprox_noniid_a.yaml`
- `configs/experiments/tune_adp_fedprox_noniid_b.yaml`
- `configs/experiments/tune_adp_fedprox_noniid_c.yaml`
- `configs/experiments/tune_adp_fednova_noniid_a.yaml`
- `configs/experiments/tune_adp_fednova_noniid_b.yaml`
- `configs/experiments/tune_adp_fednova_noniid_c.yaml`

补充代码：

- `src/data/__init__.py`
- `src/data/datasets.py`
- `src/data/partition.py`
- `src/data/statistics.py`

修复代码：

- `src/compression/topk.py`

## 11. 为什么之前结果一般：诊断结论

在继续分析曲线后，可以更明确地判断：

### 11.1 主要问题不是“轮次太少”，而是“调度策略有缺陷”

原因有两个：

1. 三种方法在中前期都已经达到各自峰值  
   例如：
   - `FedAvg + adaptive DP(C)` 在第 `10` 轮达到 `73.90%`
   - `FedProx + adaptive DP(C)` 在第 `10` 轮达到 `74.00%`
   - `FedNova + adaptive DP(C)` 在第 `15` 轮达到 `76.20%`

2. 后期准确率下滑与噪声持续上升同步出现  
   原来的 `round_based` 调度会不看模型状态，强行把噪声从 `0.02` 一路推到 `0.2`。这会导致：
   - 中前期训练逐渐变好
   - 后期明明模型还没必要加强噪声，噪声却继续上升
   - 最终把性能拉低

因此，问题更接近：

- adaptive DP 调度逻辑设计不合理

而不是：

- 聚合轮次本身不足

### 11.2 当前 `metric_based` 也不够好

项目里原本还提供了 `metric_based` 调度，但它依赖 `update_norm`。本轮日志显示，`avg_update_norm` 在训练过程中几乎单调上升，因此如果直接用它控制噪声，噪声也会很快逼近上界。

这意味着当前旧版 adaptive 机制存在一个根本问题：

- 它并没有真正根据“模型是否还在有效变好”来调节噪声

## 12. 本轮新增的算法改进：`performance_plateau` 调度

为验证问题是否真的出在调度逻辑，本轮新增了一种更贴合训练状态的自适应策略：

- `noise_schedule = performance_plateau`

核心思想：

1. 模型还在明显提升时，保持低噪声
2. 进入平台期后，再小步增加噪声
3. 如果性能明显回落，则把噪声降回去，优先恢复训练稳定性

### 12.1 新增配置

配置文件：

- `configs/experiments/tune_adp_fednova_noniid_plateau.yaml`

输出目录：

- `outputs/20260326_112524_tune_adp_fednova_noniid_plateau`

核心参数：

- `noise_schedule = performance_plateau`
- `schedule_metric = test_accuracy`
- `min_noise_multiplier = 0.02`
- `max_noise_multiplier = 0.12`
- `schedule_warmup_rounds = 3`
- `adaptive_step = 0.01`
- `plateau_patience = 2`
- `schedule_metric_tolerance = 0.003`
- `schedule_drop_tolerance = 0.02`

### 12.2 新结果

在 `FedNova` 上，新调度取得了明显更好的结果：

| 方法 | 调度 | 最佳轮次 | 最佳准确率 | 最终准确率 | 最终损失 | 最终 epsilon |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FedNova | `round_based(C)` | 15 | 76.20% | 71.70% | 0.8113 | 900.6098 |
| FedNova | `performance_plateau` | 19 | 83.10% | 81.90% | 0.5673 | 2594.7074 |

从结果看：

- 最佳准确率提升了 `6.9` 个百分点
- 最终准确率提升了 `10.2` 个百分点
- 最终损失明显下降
- 后期不再出现原来那种明显的“先变好、再被噪声打坏”的现象

### 12.3 这说明什么

这轮改进给出的结论非常明确：

1. 当前结果一般，主要原因是旧版 adaptive 调度策略不合理。
2. `FedNova` 本身并不是主要瓶颈，它在更合理的调度下可以明显提升。
3. 简单增加聚合轮次，不会从根本上解决旧调度带来的后期退化问题。
4. 如果后续继续优化，优先级应该是：
   - 先优化 adaptive 调度
   - 再考虑轮次、`clip`、噪声范围等参数

### 12.4 仍然存在的限制

新的 `performance_plateau` 虽然把精度拉上来了，但当前噪声几乎一直保持在 `0.02`，因此最终 `epsilon` 反而更高。

这意味着：

- 从“模型效能”角度，这次改进是成功的
- 从“强隐私预算”角度，它还不能直接作为论文终版方案

更准确地说，这轮改进回答的是：

- 之前精度一般，主要是不是算法设计问题？

答案是：

- 是，主要是旧版 adaptive 调度设计的问题

但它还没有完全回答：

- 如何在更低 `epsilon` 下仍然保持较高精度？

这个问题还需要继续调参和设计。

## 13. 基于方案 B 的进一步方法改进

在确认“旧版 adaptive 调度本身存在缺陷”之后，本轮继续沿方案 B 往前推进，不再只做全局噪声调度，而是开始显式利用客户端异质性。

### 13.1 轻量版方案 B：`client-aware clipping`

核心思想：

- 使用客户端上一轮的更新范数作为漂移强度信号
- 将其与全体客户端历史更新范数的中位数做比较
- 若某客户端漂移更大，则下一轮更强裁剪
- 若某客户端漂移较小，则下一轮适度放宽裁剪

新增实现：

- `src/privacy/client_adaptive.py`
- `src/fl/server.py`

新增配置：

- `configs/experiments/tune_adp_fednova_noniid_clientclip.yaml`

输出目录：

- `outputs/20260326_113600_tune_adp_fednova_noniid_clientclip`

结果：

| 方法 | 调度/机制 | 最佳准确率 | 最终准确率 | 最终损失 | 最终 epsilon |
| --- | --- | ---: | ---: | ---: | ---: |
| FedNova | `round_based(C)` | 76.20% | 71.70% | 0.8113 | 900.6098 |
| FedNova | `round_based(C) + client-aware clipping` | 76.90% | 72.95% | 0.7859 | 900.6098 |

结论：

- 单独加入客户端感知裁剪后，性能有小幅提升
- 说明“统一 clip 对所有客户端一刀切”确实不是最优方案
- 但它还不足以单独解决“隐私预算和后期噪声调度”这两个更大的问题

### 13.2 方法 v2：`client-aware clipping + performance_budget`

为了让方法更完整，本轮进一步把方案 B 与预算感知的全局调度结合起来，形成一个更像正式方法的版本：

- 客户端层：`client-aware clipping`
- 全局层：`performance_budget` 噪声调度

`performance_budget` 的思路是：

1. 若模型还在明显提升，则尽量维持较低噪声
2. 若进入平台期，则小步增加噪声
3. 若累计 `epsilon` 明显快于目标预算轨迹，则主动提高噪声
4. 若预算压力缓解，则允许回到更低噪声

新增配置：

- `configs/experiments/tune_adp_fednova_noniid_clientclip_budget.yaml`

输出目录：

- `outputs/20260326_122710_tune_adp_fednova_noniid_clientclip_budget`

关键参数：

- `noise_schedule = performance_budget`
- `target_epsilon = 1200`
- `min_noise_multiplier = 0.04`
- `max_noise_multiplier = 0.14`
- `client_aware_clipping = true`
- `min_clip_norm = 0.6`
- `max_clip_norm = 1.4`

### 13.3 v2 方法结果

| 方法 | 调度/机制 | 最佳准确率 | 最终准确率 | 最终损失 | 最终 epsilon |
| --- | --- | ---: | ---: | ---: | ---: |
| FedNova | `round_based(C)` | 76.20% | 71.70% | 0.8113 | 900.6098 |
| FedNova | `round_based(C) + client-aware clipping` | 76.90% | 72.95% | 0.7859 | 900.6098 |
| FedNova | `performance_plateau` | 83.10% | 81.90% | 0.5673 | 2594.7074 |
| FedNova | `client-aware clipping + performance_budget` | 82.35% | 80.85% | 0.5850 | 1284.3802 |

### 13.4 对 v2 的判断

这一版结果很关键，因为它说明：

1. 方案 B 是有效方向  
   客户端异质性感知确实能带来收益。

2. 单纯客户端感知裁剪还不够  
   它只能小幅改进，不足以从根本上解决旧 adaptive 机制的问题。

3. `client-aware clipping + performance_budget` 更像真正的“方法改进版”  
   它同时考虑了：
   - 客户端异质性
   - 全局训练状态
   - 隐私预算消耗速度

4. 这一版比 `performance_plateau` 略低一点点精度，但隐私预算明显收敛  
   - `performance_plateau`：最终 `81.90%`，`epsilon = 2594.71`
   - v2：最终 `80.85%`，`epsilon = 1284.38`

也就是说，v2 不是当前“最高精度”的版本，但它更平衡，更像论文里可以继续打磨的正式方法雏形。

### 13.5 当前阶段最值得保留的方向

如果以“论文方法雏形”来选，而不是只看单次最高准确率，我当前更推荐保留：

- `FedNova + client-aware clipping + performance_budget`

原因是它同时回答了你最核心的三个问题：

1. 如何利用 Non-IID 异质性信息  
2. 如何避免旧版 adaptive DP 的后期退化  
3. 如何避免一直维持低噪声导致预算完全失控

因此，后续若继续做正式方法，我建议以这个 v2 版本为主线继续精修，而不是回到旧的 `round_based`。

## 14. 针对“隐私预算仍然偏高”的进一步优化

虽然 v2 已经比 `performance_plateau` 好很多，但最终 `epsilon = 1284.38` 仍然偏高。因此，本轮继续针对“高风险客户端预算过快积累”这个问题做进一步改进。

### 14.1 新增改进：高风险客户端额外加噪

在上一版 v2 中，客户端感知部分只作用于 `clip_norm`，还没有直接作用于噪声。

这一版新增：

- `client-aware noise`

核心思想：

1. 若客户端上一轮更新范数更大，则认为其漂移风险更高
2. 若客户端本地样本更少，则其有效采样率更高，也更容易积累隐私预算
3. 对这类高风险客户端，额外提高噪声倍率
4. 对普通客户端，不额外加噪，不再“一刀切”提高所有人的噪声

这样做的目标是：

- 优先压低最容易拉高最大 `epsilon` 的那部分客户端
- 尽量少伤整体模型精度

新增实现：

- `src/privacy/client_adaptive.py`

新增配置：

- `configs/experiments/tune_adp_fednova_noniid_clientclip_budget_noise.yaml`

输出目录：

- `outputs/20260326_124708_tune_adp_fednova_noniid_clientclip_budget_noise`

### 14.2 新结果

| 方法 | 调度/机制 | 最佳准确率 | 最终准确率 | 最终损失 | 最终 epsilon |
| --- | --- | ---: | ---: | ---: | ---: |
| FedNova | `round_based(C)` | 76.20% | 71.70% | 0.8113 | 900.6098 |
| FedNova | `round_based(C) + client-aware clipping` | 76.90% | 72.95% | 0.7859 | 900.6098 |
| FedNova | `performance_plateau` | 83.10% | 81.90% | 0.5673 | 2594.7074 |
| FedNova | `client-aware clipping + performance_budget` | 82.35% | 80.85% | 0.5850 | 1284.3802 |
| FedNova | `client-aware clipping + performance_budget + client-aware noise` | 80.80% | 80.80% | 0.5803 | 657.2709 |

### 14.3 对这一版的判断

这一版非常关键，因为它第一次把“预算”真正压到了一个更像论文可接受范围的区间，同时没有把精度打崩。

具体看：

- 与 `performance_plateau` 相比  
  - 最终准确率只下降了约 `1.1` 个百分点
  - 但 `epsilon` 从 `2594.71` 降到 `657.27`

- 与 v2 相比  
  - 最终准确率从 `80.85%` 基本维持到 `80.80%`
  - `epsilon` 从 `1284.38` 进一步降到 `657.27`

这说明新增的 `client-aware noise` 是有效的：

- 它没有像“全局统一提高噪声”那样明显破坏模型
- 却成功把最坏客户端的预算积累压了下来

### 14.4 当前最优候选方法

如果现在要在“精度、隐私预算、方法完整性”三者之间选一个最平衡的版本，我当前最推荐的是：

- `FedNova + client-aware clipping + performance_budget + client-aware noise`

原因是它同时具备：

1. 利用 Non-IID 异构信息  
   通过客户端感知裁剪与客户端感知噪声体现“异构感知”。

2. 保持较高模型精度  
   最终准确率仍在 `80%+`。

3. 显著降低隐私预算  
   在目前几版方法里，已经把 `epsilon` 压到了目前最合理的水平。

### 14.5 当前阶段的建议

到这一步，我不建议再回到旧方法继续大范围盲调。更值得做的是以这版方法为主线，继续做两件事：

1. 固定当前方法结构  
   即：
   - `FedNova`
   - `performance_budget`
   - `client-aware clipping`
   - `client-aware noise`

2. 在这个结构上做小范围精修  
   重点参数可以只保留：
   - `min_noise_multiplier`
   - `client_noise_beta`
   - `client_clipping_beta`
   - `target_epsilon`

也就是说，后续工作重点应从“重新找方法”切换为“在已经合理的方法结构上做正式实验”。

## 15. 以更强无 DP 基线重新评估掉点

### 15.1 背景

前面的 DP 结果都建立在一个相对保守的底座上：

- `max_train_samples = 10000`
- `max_test_samples = 2000`
- `rounds = 20`
- `local_epochs = 1`
- 无学习率衰减

后来我们把无 DP 基线升级为：

- 全量 `MNIST`
- `rounds = 100`
- `local_epochs = 2`
- 指数学习率衰减

对应配置：

- `configs/experiments/fednova_mnist_full_nodp_strong.yaml`

对应输出：

- `outputs/20260326_125832_fednova_mnist_full_nodp_strong`

这组更强底座的最终准确率达到：

- `98.50%`

这意味着，之前在弱底座上观察到的 DP 掉点，可能低估了“真正强基线下的精度损失”。

### 15.2 本轮重新评估的做法

为了在同一底座下公平比较，我额外补了两组配置：

- 固定 DP：
  - `configs/experiments/fednova_mnist_full_fixed_dp_eval.yaml`
- 自适应 DP：
  - `configs/experiments/fednova_mnist_full_adaptive_dp_eval.yaml`

两者都与强无 DP 基线保持一致：

- 全量 `MNIST`
- `FedNova`
- `rounds = 100`
- `local_epochs = 2`
- 相同学习率衰减
- 相同 `Dirichlet alpha = 0.1`

说明：

- 这两组长程实验已启动，但目前先记录到第 `12` 轮的同窗口结果。
- 因为完整 `100` 轮运行时间较长，这里先给出中期结论，不把它误写成最终值。

对应输出：

- `outputs/20260329_192459_fednova_mnist_full_fixed_dp_eval`
- `outputs/20260329_193247_fednova_mnist_full_adaptive_dp_eval`

### 15.3 第 5 / 10 / 12 轮的同窗口对比

| 轮次 | 无 DP 强基线 | 固定 DP | 自适应 DP |
| --- | ---: | ---: | ---: |
| 5 | 95.51% | 70.58% | 80.77% |
| 10 | 97.06% | 85.36% | 88.15% |
| 12 | 97.06% | 89.14% | 89.88% |

对应的累计 `epsilon`：

| 轮次 | 固定 DP epsilon | 自适应 DP epsilon |
| --- | ---: | ---: |
| 5 | 75.52 | 54.67 |
| 10 | 151.04 | 103.36 |
| 12 | 181.25 | 123.50 |

### 15.4 结果解释

这组结果非常重要，因为它把“DP 到底掉了多少点”放回到了一个更合理的高性能底座上。

在第 `12` 轮这个可直接对齐的窗口下：

- 无 DP 强基线：`97.06%`
- 固定 DP：`89.14%`
- 自适应 DP：`89.88%`

也就是说：

- 固定 DP 相对强基线掉了约 `7.92` 个百分点
- 自适应 DP 相对强基线掉了约 `7.18` 个百分点

同时，自适应 DP 的累计 `epsilon` 还更低：

- 固定 DP：`181.25`
- 自适应 DP：`123.50`

这说明两件事：

1. 在更强底座下，DP 带来的精度代价会被更清楚地暴露出来。  
   之前弱底座下的结果，部分掩盖了这个问题。

2. 我们当前的自适应方法是有效的，但提升幅度还不算“质变”。  
   它相对固定 DP 更稳、更省预算，但还没有把强基线下的掉点压到足够小。

### 15.5 当前判断

到这一步，可以得到一个更清楚的结论：

- 之前低精度的主因，确实是底座太弱，不是 `FedNova` 本身不行。
- 但当底座抬高以后，DP 方法的真实代价也更明显了。
- 当前自适应 DP 已经优于固定 DP，但距离“接近无 DP 基线”还有明显差距。

因此，下一阶段的改进重点不应该再是“证明 FedNova 能不能训起来”，而应该是：

- 如何在强基线上进一步缩小 `97.06% -> 89.88%` 这类中期掉点
- 如何在不显著增加 `epsilon` 的前提下继续保精度

## 16. 算法改进：风险感知聚合

### 16.1 改进动机

前面的分析说明，当前 adaptive DP 已经优于固定 DP，但在强基线上仍然存在明显掉点。

一个潜在原因是：

- 服务器聚合时仍然主要按样本数加权
- 但在 DP 场景下，不同客户端的更新“可靠性”并不一样
- 高漂移客户端、强噪声客户端的更新更容易把全局方向拉偏

这说明仅仅在客户端侧做：

- `client-aware clipping`
- `client-aware noise`

还不够，服务器侧也应该感知“哪些客户端更新更不稳定”。

### 16.2 方法设计

我新增了一层轻量的 `risk-aware aggregation`，核心思路是：

- 先保留 `FedNova` 原有的归一化聚合框架
- 不改变主算法主体
- 只在客户端样本权重之外，再乘一个“可靠性系数”

可靠性系数同时参考：

- 当前轮客户端更新范数相对中位数的偏离程度
- 当前轮客户端噪声倍率相对中位数的偏离程度

直觉上：

- 更新范数异常大、噪声异常强的客户端，适度降权
- 稳定客户端保持原有权重附近
- 不直接丢弃任何客户端

新增实现：

- `src/privacy/client_adaptive.py`
- `src/fl/server.py`

新增配置：

- `configs/experiments/fednova_mnist_full_adaptive_dp_riskagg_long_eval.yaml`

### 16.3 公平对比结果

为了公平比较，我使用与上一版 adaptive 完全一致的底座：

- 全量 `MNIST`
- `FedNova`
- `rounds = 100`
- `local_epochs = 2`
- 相同学习率衰减
- 相同 `Dirichlet alpha = 0.1`

只看第 `12` 轮窗口：

| 方法 | 第 12 轮准确率 | 第 12 轮 epsilon | 第 12 轮全局噪声 |
| --- | ---: | ---: | ---: |
| 无 DP 强基线 | 97.06% | 0.00 | 0.00 |
| 固定 DP | 89.14% | 181.25 | 0.12 |
| adaptive DP | 89.88% | 123.50 | 0.10 |
| adaptive DP + risk-aware aggregation | 90.40% | 123.50 | 0.09 |

对应输出：

- `outputs/20260329_204458_fednova_mnist_full_adaptive_dp_riskagg_long_eval`

### 16.4 结果判断

这个改进不是颠覆式提升，但它是一个“真实有效、方向正确”的改进：

- 相比上一版 adaptive，准确率从 `89.88%` 提升到 `90.40%`
- 提升约 `0.52` 个百分点
- `epsilon` 基本不变，仍为 `123.50`

这说明：

1. 服务器侧确实存在可优化空间。  
   仅靠客户端侧裁剪和噪声控制，还没有把 DP 噪声传播问题处理完。

2. 风险感知聚合是一个合理方向。  
   它没有额外扩大隐私预算，却带来了稳定的收益。

3. 但它还不是最终答案。  
   即使加上这一层，强基线下的第 `12` 轮掉点仍然约为 `6.66` 个百分点。

### 16.5 当前最值得继续的方向

到目前为止，我认为最有希望的主线结构已经比较清楚：

- `FedNova`
- `performance_budget`
- `client-aware clipping`
- `client-aware noise`
- `risk-aware aggregation`

这比早期单纯“调噪声曲线”的版本更完整，也更像一个正式的方法框架。
