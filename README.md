# CertiScale · 衡证

> **可证明的推理时计算规模化** —— 唯一在「随时可停 + 错误率可证 + 用到验证器 + 共享算力预算」四点上同时成立的开源推理认证系统。
>
> Certifiable Test-Time Compute Scaling · Anytime-Valid Verification × Conformal Calibration × Best-Arm Budget Allocation
>
> **作者：晨星** · GitHub [@CJX0712](https://github.com/CJX0712) · MIT License

---

## 一句话说清它解决什么

大模型靠「多采样几条推理链再投票」提分，但**什么时候停**一直是个玄学：停早了不准，停晚了烧钱，而且**看数据决定何时停**这件事本身会让经典统计推断失效（反复偷看 p 值会把 5% 的错误率 inflated 到 20%+）。

当前世界最强解是 **CITE**（arXiv:2605.05873），它能做到「数据驱动的任意停止规则下，假认证率仍 ≤ ε」。但它有三个盲区：

| 盲区 | 后果 | CertiScale 的解法 |
|------|------|-------------------|
| 只看**众数**，不看**正确性** | 模型一致地答错时，CITE 照样自信认证 | **VCE** 通道：把任意黑盒验证器转成严格有效的 e-process |
| 单问题视角，无**共享预算** | 简单题浪费算力，难题饿死 | **AVBA**：按对数财富增长率分配采样预算 |
| 只输出一个标签 | 无覆盖率保证的预测集 |  conformal 校准层 |

---

## 三大原则（执行总纲）

| 原则 | 本仓库如何落地 |
|------|----------------|
| **零重写** | 每个原语都标注来源论文，不自己发明。CITE 逐条复现为 Tier-0 基线 |
| **集大成** | 站在 Ramdas (e-process)、Vovk & Wang (p→e)、Vovk et al. (conformal)、CITE (mode testing) 之上做**组合** |
| **闭环交付** | `python verify.py` 一键自检 13/13 全绿；benchmark 输出 JSON 可复现 |

**本仓库的原创性不在原语，而在组合。** 原语全是已知最优；组合方式（VCE × CITE 的 union bound、AVBA 的财富梯度）是新的。

---

## 架构

```
                      共享采样预算 B
                            │
                    ┌───────▼────────┐
                    │  AVBA 分配器    │  ← 按 log-wealth 增长率决定下一个采样谁
                    └───────┬────────┘
             ┌──────────────┼──────────────┐
             ▼              ▼              ▼
        ┌─────────┐    ┌─────────┐    ┌─────────┐
        │ 问题 1  │    │ 问题 2  │    │ 问题 K  │
        └────┬────┘    └────┬────┘    └────┬────┘
             │              │              │
        ┌────▼──────────────▼──────────────▼────┐
        │         CertiScale 认证引擎            │
        │  ┌──────────────┐  ┌────────────────┐ │
        │  │ CITE 通道     │  │ VCE 通道        │ │
        │  │ ·pairwise e  │  │ ·conformal p    │ │
        │  │ ·LCB on p_r  │  │ ·Vovk-Wang e    │ │
        │  │ ·unseen bound│  │ ·batch 聚合      │ │
        │  └──────┬───────┘  └────────┬───────┘ │
        │         └──── AND ──────────┘         │
        └──────────────────┬────────────────────┘
                           ▼
                  认证 / 继续采样 / 放弃
```

---

## 数学核心（为什么它是对的，而不是碰巧对）

### 停止规则

```
τ = inf{ t :  E_t ≥ 3/ε_cite  且  L_t > U_t      ← CITE 通道（r 是唯一众数）
          且  W_t ≥ 1/ε_vce   }                  ← VCE 通道（r 是正确答案）
```

### 有效性论证（关键）

两个通道检验的是**不同的零假设**：

- CITE：`H₀ᵐᵒᵈᵉ : p_r ≤ sup_{a≠r} p_a`（r 不是唯一众数）
- VCE：`H₀ᶜᵒʳʳᵉᶜᵗ :` r 不是正确答案

我们认证的是合取命题「r 是唯一众数**且** r 正确」，所以联合零假设是两者的**并集**。在任意零世界中至少一个通道是**真零**，于是

> **P(τ < ∞) ≤ max(ε_cite, ε_vce)**

—— 注意是 **max 而非 sum**。合取结构买到这个，这也是我们把两个预算都设为 ε 的原因。

### VCE 为什么是真 e-process（不是跑动平均）

跑动**均值**在每个固定 t 是 e-variable，但**不是**上鞅，Ville 不等式不适用，偷看仍会破坏保证。所以用**乘积** `W_t = Π e_i`：

- 采样轨迹 i.i.d.、校准集在打分前冻结 ⟹ 每个 `e_i` 只是 `X_i` 的函数
- ⟹ `E[e_i | F_{i-1}] = E[e_i] ≤ 1`（条件 e-variable）
- ⟹ `W_t` 是非负上鞅，Ville 在**任意**数据依赖停止时间下成立
- 全部在 log 空间计算，不会溢出

### 分批（batching）：VCE 唯一的调参旋钮

批内**聚合**降方差（∝ 1/batch），批间**不相交**保上鞅性质。这是把弱验证器变可用的关键。

> ⚠️ **陷阱（实践中最常踩）**：校准集必须用**相同 batch** 构建。用单分数校准、却用批均值打分，等于拿两个不同统计量互比，conformal 的超均匀性会静默失效。本仓库 `verifier_calibration(..., batch=...)` 强制对齐，测试 I7 守卫这一点。

---

## 实验结果

`bench/run_bench.py --seeds 5 --problems 40 --vce-batch 8`（ε=0.05，30% 陷阱题 = 众数是错答案）

| Regime | CITE FDR | **CertiScale FDR** | FDR 降幅 | Power (CITE→CS) | 样本成本 |
|--------|---------:|-------------------:|---------:|-----------------|---------:|
| moderate | 0.300 ± 0.000 | **0.014 ± 0.008** | **21.4×** | 1.000 → 1.000 | +9.9% |
| diffuse_tail | 0.299 ± 0.003 | **0.007 ± 0.007** | **42.7×** | 0.986 → 0.993 | −0.6% |
| near_tie | 0.301 ± 0.007 | **0.007 ± 0.007** | **43.0×** | 0.879 → 0.893 | +6.3% |

**结论**：FDR 降 21–43 倍，功效**不降反升**，样本成本 +6%~+10%（diffuse 场景下甚至持平）。

> 诚实的 framing：VCE 是 **AND** 条件，它**不会**让认证更快——它让认证**更对**。任何声称"我们的方法更快"的说法在这里都是错的。真实权衡是：**用约 6% 的额外算力，把认证答案的错误发现率从 30% 砍到 1% 以下。**

复现：

```bash
python bench/run_bench.py --seeds 5 --problems 40 --vce-batch 8
```

---

## 快速开始

```bash
pip install -r requirements.txt
python verify.py                 # 13 道数学不变量门禁，全绿
pytest tests/ -q                 # 16 项完整不变量套件
python bench/run_bench.py --seeds 3 --problems 20   # 快速跑一遍实验
```

最小用法：

```python
import numpy as np
from certiscale.cite import CITECertifier, CiteConfig
from certiscale.flagship import CertiScaleConfig, CertiScaleEngine
from certiscale.verifier import ConformalEVerifier

# 1) 纯 CITE：认证「42」是唯一众数，随时可停
cert = CITECertifier(CiteConfig(eps=0.05), target="42")
for x in sampled_answers:          # x 是一条推理链的最终答案
    if cert.step(x):
        break                      # 已认证，且假认证率 ≤ 5%

# 2) CertiScale：众数 + 正确性 双重认证
v = ConformalEVerifier(name="unit_test", null_calibration=wrong_scores, kappa=0.5)
eng = CertiScaleEngine(CertiScaleConfig(eps=0.05, vce_batch=8), target="42", verifiers=[v])
for label, score in trajectories:
    if eng.step(label, {"unit_test": [score]} if label == "42" else None):
        break                      # 众数正确且唯一，错误率仍 ≤ 5%
```

---

## 可验证不变量（`python verify.py`）

| ID | 不变量 | 类型 |
|----|--------|------|
| I1 | pairwise e-process 满足 Ville 不等式 | Monte-Carlo |
| I3 | `U_t` 单调递减且足够紧 | 解析 |
| I4 | LCB 有效性（CITE Prop. 3.4） | Monte-Carlo |
| I5 | Vovk-Wang 校准器 `E[e] ≤ 1` | Monte-Carlo |
| I6 | conformal p 值超均匀 | Monte-Carlo |
| I7 | 验证器 e 值：零假设 ≤1、备择 >1 | Monte-Carlo |
| I8 | 验证器融合（**无需独立性**）保持有效 | Monte-Carlo |
| I9 | CITE 假认证率 ≤ ε（含顶部平局） | Monte-Carlo |
| I10 | CITE 功效：显著众数被认证 | Monte-Carlo |
| I11 | CertiScale 拒绝**错误的**众数 | Monte-Carlo |
| I13 | 分配器遵守预算、不重采已完成项 | 解析 |
| I14 | 固定种子逐位确定性 | 解析 |
| I15 | 无 NaN / 无溢出 | 解析 |

```
✅ I1 … I15   13/13 gates green
```

---

## 文献来源（零重写清单）

| 组件 | 来源 |
|------|------|
| 众数认证的 e-process 框架 | CITE, Ota/Iwase/Ichihara/Komiyama/Imaizumi (2026), [arXiv:2605.05873](https://arxiv.org/abs/2605.05873) |
| pairwise 下注型 e-process | Ramdas, Ruf, Larsson & Koolen (2023), [arXiv:2209.11119](https://arxiv.org/abs/2209.11119) |
| p→e 校准 `e_κ(p)=κp^{κ-1}` | Vovk & Wang (2021), *Annals of Statistics* 49(3) |
| conformal p 值 | Vovk, Gammerman & Shafer (2005), *Algorithmic Learning in a Random World* |
| 序贯减半 / 预算化纯探索 | Karnin, Koren & Somekh (2013); Audibert & Bubeck (2010) |
| e 值合并规则 | Vovk & Wang (2021) §4 |

**本仓库贡献（novel）**：VCE（`verifier.py`）、AVBA（`allocator.py`）、CITE×VCE 的 max-误差组合（`flagship.py`）。

---

## 目录

```
src/certiscale/
  eprocess.py    e 值 / e-process 原语（log 空间，Vovk-Wang 校准器）
  cite.py        CITE 完整复现（Tier-0 SOTA 基线）
  conformal.py   分割 conformal p 值
  verifier.py    VCE：验证器 → 合法 e 值（本仓库原创）
  allocator.py   AVBA：跨问题预算分配（本仓库原创）
  flagship.py    CertiScale：CITE × VCE × AVBA（本仓库原创）
  simulator.py   可复现合成问题生成器（仅用于 benchmark）
tests/           16 项不变量（pytest）
bench/           实验脚本 + JSON 结果
docs/            THEORY / INVARIANTS / BENCHMARK
verify.py        零依赖一键自检
```

---

## 局限（诚实清单）

- VCE 的力量完全取决于验证器质量。验证器与真值无关时（sep=0），VCE 退化为随机噪声，系统靠 CITE 通道兜底，仍保证错误率但无额外收益。
- 批量 `vce_batch` 越大越稳，但每批需攒够支持该答案的轨迹；低 `p_mode` 场景下可能迟迟凑不满一批。
- 合成 benchmark 用高斯分数模拟验证器；真实验证器常有重尾与系统性偏差，conformal 层保证的是**交换性下**的有效性，分布漂移需重新校准。
- 单问题认证在极难 regime（gap < 0.05）下可能需要数百次采样，此时 AVBA 的预算分配比认证规则本身更影响吞吐。

---

## 作者

**晨星** · [@CJX0712](https://github.com/CJX0712)

MIT License · 零重写 · 集大成 · 闭环交付
