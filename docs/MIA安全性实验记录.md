# MIA安全性实验记录

## 1. 实验目的

验证在 `MNIST + Non-IID(alpha=0.1)` 场景下：

- 原始 `FedNova`
- `FedNova + adaptive DP`

两者面对黑盒成员推理攻击（Membership Inference Attack, MIA）时的安全性差异。

## 2. 威胁模型

本次采用的是一个黑盒 MIA：

- 攻击者只能访问最终全局模型
- 攻击者不知道客户端本地数据细节
- 攻击者对样本输入进行查询，获得模型输出
- 攻击者基于样本的真实标签置信度与单样本损失进行成员判别

成员样本与非成员样本定义：

- 成员样本：联邦训练中实际使用过的训练集样本
- 非成员样本：测试集样本

攻击指标：

- `AUC(loss)`
- `AUC(confidence)`
- 最优阈值攻击准确率

说明：

- 若 `AUC` 接近 `0.5`，说明攻击接近随机猜测
- 若显著高于 `0.5`，说明模型更容易泄露成员信息

## 3. 实验设置

公共底座：

- 数据集：`MNIST`
- 划分方式：`Dirichlet Non-IID`
- `alpha = 0.1`
- 客户端数：`10`
- 训练轮数：`12`
- `local_epochs = 2`
- `batch_size = 64`
- 学习率：`0.05`
- 指数学习率衰减

攻击采样：

- 成员样本数：`2000`
- 非成员样本数：`2000`

## 4. 对比方法

### 4.1 原始 FedNova

配置：

- `configs/experiments/fednova_mnist_full_nodp_mia_eval.yaml`

输出目录：

- `outputs/20260329_212241_fednova_mnist_full_nodp_mia_eval_mia`

### 4.2 我们的方法

配置：

- `configs/experiments/fednova_mnist_full_adaptive_dp_riskagg_eval.yaml`

输出目录：

- `outputs/20260329_212902_fednova_mnist_full_adaptive_dp_riskagg_eval_mia`

说明：

- 该方法包含：
  - `performance_budget`
  - `client-aware clipping`
  - `client-aware noise`
  - `risk-aware aggregation`

攻击脚本：

- `scripts/run_mia_experiment.py`

## 5. 实验结果

| 方法 | 最终准确率 | 最终 epsilon | AUC(loss) | AUC(confidence) | 最优攻击准确率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FedNova | 97.06% | 0.00 | 0.4927 | 0.4927 | 0.5048 |
| FedNova + adaptive DP | 91.15% | 131.20 | 0.4999 | 0.4999 | 0.5153 |

补充统计：

原始 `FedNova`

- 成员平均置信度：`0.951365`
- 非成员平均置信度：`0.949293`
- 成员平均 loss-score：`-0.084398`
- 非成员平均 loss-score：`-0.099325`

`adaptive DP`

- 成员平均置信度：`0.876334`
- 非成员平均置信度：`0.875141`
- 成员平均 loss-score：`-0.261935`
- 非成员平均 loss-score：`-0.289939`

## 6. 结果解读

这次实验最重要的结论是：

1. 两个模型在当前黑盒 MIA 下都接近随机猜测  
   两组 `AUC` 都非常接近 `0.5`，说明攻击者几乎不能有效区分成员与非成员。

2. 我们的方法没有表现出更强的成员泄露风险  
   `adaptive DP` 的 `AUC` 约为 `0.4999`，与随机水平几乎一致。

3. 原始 `FedNova` 也没有在当前设置下表现出明显泄露  
   这说明当前模型的泛化较好，成员与非成员在输出分布上的差异本来就不大。

4. 从“安全性是否足够”的角度看，当前结果是正面的  
   至少在这个黑盒成员推理攻击下，我们的方法没有暴露出明显隐私泄露问题。

## 7. 需要保留的谨慎说明

虽然结果是正面的，但论文里建议保留以下限定：

- 这里只验证了黑盒 MIA
- 攻击对象是最终全局模型
- 还没有测试更强攻击者，例如：
  - 基于影子模型的 MIA
  - 基于中间轮次更新的攻击
  - 白盒攻击

因此，更严谨的写法应该是：

- “在当前黑盒成员推理攻击设置下，`adaptive DP` 方法未表现出显著的成员信息泄露风险。”

而不是直接写成：

- “该方法完全抵御成员推理攻击。”

## 8. 当前结论

如果用一句话概括这次安全性实验：

- 原始 `FedNova` 和我们的 `adaptive DP` 在当前黑盒 MIA 下都接近随机猜测，而我们的方法在保持较好精度的同时，没有表现出额外的成员泄露风险。
