# CertiScale 不变量清单

> `python verify.py` 一键自检 · `pytest tests/ -q` 完整套件
> 作者：晨星 (CJX0712)

---

## 设计原则

每条不变量都是**数学命题**，不是回归快照。绿灯意味着 README 与 `docs/THEORY.md` 承诺的性质**在本机、本 commit 上真的成立**。

**Monte-Carlo 型门禁的松弛**：统计保证只能被**估计**。要求观测违反率不超过名义水平加 **4 个标准误**：

$$\text{slack} = 4\sqrt{\frac{\varepsilon(1-\varepsilon)}{R}}$$

$R$ 为重复次数。$R$ 越大松弛越小 —— 这是刻意的，避免用少量样本"碰巧通过"。

---

## 门禁清单

| ID | 不变量 | 数学依据 | 类型 | 松弛 |
|----|--------|----------|------|------|
| **I1** | pairwise e-process 满足 Ville 不等式：零假设下 $P(\sup_t E_t \ge 1/\alpha) \le \alpha$ | Ville / CITE Cor. 3.2 | MC, 400 路径 | 4 se |
| **I2** | $E_t$ 关于 $N_r$ 单调递增、关于 $N_a$ 单调递减 | 式 (2) 直接推论 | 解析 | — |
| **I3** | $U_t$ 单调递减，且质量 $>5U_t$ 的类别必被观测 | CITE Def. 3.5 | 解析 + MC | — |
| **I4** | LCB 有效性：$P(\exists t : L_t > p_r) \le \varepsilon/3$ | CITE Prop. 3.4 | MC, 300 路径 | 4 se |
| **I5** | Vovk-Wang 校准器 $\mathbb{E}[e_\kappa(p)] \le 1$（$\kappa = 0.25/0.5/0.75$） | Vovk & Wang 2021 | MC, 5000 | 4 se |
| **I6** | conformal p 值超均匀：$P(p \le u) \le u$（$u$ = 0.05/0.1/0.25/0.5） | VGS 2005 | MC, 5000 | 4 se |
| **I7** | 验证器 e 值：零假设下均值 $\le 1$，备择下 $> 1$ | §3.2 证明 | MC, 4000 | 4 se |
| **I8** | 验证器融合**无需独立性**仍有效（刻意用完全相关的分数） | 凸组合保 e-variable | MC, 3000 | 4 se |
| **I9** | CITE 假认证率 $\le \varepsilon$（含**顶部平局**最难零假设） | CITE Thm 4.1 | MC, 300×2 | 4 se |
| **I10** | CITE 功效：显著众数（$p_r = 0.70$）被认证，$\tau < 200$ | CITE Thm 4.3 | MC, 60 | — |
| **I11** | **CertiScale 拒绝错误的众数** | §4 max-误差论证 | MC, 200 | 4 se |
| **I12** | VCE 不显著拖慢认证（$\tau_{\text{VCE}} \le 1.35\,\tau_0$） | §4 代价分析 | MC, 40×2 | — |
| **I13** | 分配器遵守预算、从不重采已认证项 | AVBA 不变量 | 解析, 3 策略 | — |
| **I14** | 固定种子**逐位**确定性（模拟器 + 分配器） | 可复现性 | 解析 | — |
| **I15** | 长程运行无 NaN / 无溢出（允许饱和到 $+\infty$） | 数值卫生 | 解析 | — |
| **I16** | 模拟器可复现且分布合法（$\sum p = 1$） | 可复现性 | 解析 | — |

---

## 实测输出

```
CertiScale invariant gates
============================================================================
✅ I1  pairwise e-process: Ville under H0      rate=0.0400 <= 0.1       0.3s
✅ I3  unseen bound U_t monotone & tight       U_1e6=1.52e-05           0.0s
✅ I4  LCB validity (CITE Prop. 3.4)           rate=0.0033 <= 0.0462    0.7s
✅ I5  Vovk-Wang calibrator: E[e] <= 1         worst mean-4se=0.9866    0.0s
✅ I6  conformal p-value super-uniform         all u ok                 0.0s
✅ I7  verifier e-value: E[e]<=1 null, >1 alt  null=0.972 alt=4.90      0.0s
✅ I8  bank fusion valid w/o independence      mean=0.901               0.0s
✅ I9  CITE false certification <= eps         strict=0.0000 tie=0.0067 78.6s
✅ I10 CITE power: dominant mode certified     rate=1.000 tau=37.9      2.3s
✅ I11 CertiScale rejects a WRONG mode         rate=0.0000 <= 0.1116   21.9s
✅ I13 allocator: budget & done-masking        3 strategies ok          0.0s
✅ I14 bit-level determinism (fixed seed)      simulator + allocator    0.0s
✅ I15 numerical hygiene: no NaN/overflow      E_t=inf U_t=0.4233       0.0s
============================================================================
13/13 gates green
total 104.0s
```

`pytest tests/ -q` → **16 passed**（含 I2、I12、I16 三条 verify.py 未覆盖的）。

---

## 踩过的坑（真实发生，已修复）

这些不是虚构的注意事项，是本次开发中**实际触发的失败**：

| 现象 | 根因 | 修复 |
|------|------|------|
| `math domain error` in `unseen_bound` | $u = 1$ 时 $\log(1-u) = \log 0$ | 特判 $u \ge 1 \Rightarrow -\infty$ |
| $U_t$ 恒返回 1.0 | 可行集是**上集**，单调方向判反 | 先查 $\mathrm{viol}(lo) \le 0$，再查 $\mathrm{viol}(hi) > 0$ |
| e 值变 `nan` | $(1+\lambda)^N$ 溢出双精度 | 全程 log 空间 + `log1p` + `logsumexp` |
| I10 `rate=0.000` | 测试里 target `"a"` 不在生成的标签表 `c0..c4` 中，计数恒为 0 | 修正 target 并加 `assert target in labels` |
| I9 假认证率 0.2467（远超 ε） | 测试注释写"平局"但代码是 $p_r = 0.45 > 0.275$ —— **根本不是零假设** | 改用真正的 $H_0$：$p_r = 0.35 < 0.45$，并补测顶部平局 |
| I6 边界失败 | 5000 个 p 值**共享一个校准集**，相互相关 | 每 100 个轮换校准集 |
| **FDR 只降到 0.233** | 校准用**单分数**、打分用**批均值**，统计量不一致 | `batch` 参数强制对齐 → FDR 降到 0.007 |

最后一条是本轮**最有价值的发现**：统计量一致性比任何超参数调优都重要。

---

作者：晨星 (CJX0712) · MIT License
