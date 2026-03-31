# experiments scripts

本目录提供按实验组织的批量运行脚本：

- `run_exp_a_iid.sh`
- `run_exp_a_noniid_alpha_0_1.sh`
- `run_exp_b_alpha_sensitivity.sh`
- `run_exp_c_whitebox.sh`
- `run_exp_d_privacy_tradeoff.sh`
- `run_exp_e_final_summary.sh`

默认假设你已经激活了正确的 Python / Conda 环境。

批量脚本默认会先跑 `MNIST`，再跑 `CIFAR-10`。

如果要指定解释器，可使用：

```bash
PYTHON_BIN=python scripts/experiments/run_exp_a_iid.sh
```
