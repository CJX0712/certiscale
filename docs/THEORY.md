# CertiScale 理论文档

> 本文档给出完整的数学设定、每条公式的来源，以及**为什么组合之后保证仍然成立**。
> 作者：晨星 (CJX0712)

---

## 1. 问题形式化

设 $\mathcal{A}$ 为可数（可能无限）的答案类别集合，$P$ 为 $\mathcal{A}$ 上的未知类别分布，记 $p_a := P(X=a)$。$p_{(1)} \ge p_{(2)} \ge \cdots$ 为其降序重排。观测到 i.i.d. 序列 $X_1, X_2, \ldots \sim P$，自然滤流 $\mathcal{F}_t := \sigma(X_1,\ldots,X_t)$。

记

$$N_t(a) := \sum_{i=1}^{t} \mathbf{1}\{X_i = a\}, \qquad \hat p_t(a) := \frac{N_t(a)}{t}, \qquad \mathcal{A}_t := \{a : N_t(a) \ge 1\}$$

$\mathcal{A}$ 可能是可数无限的**且未知** —— 这正是 LLM 答案空间的本质特征，也是 CITE 相较经典方法的关键区别（Remark 2.4）。

**固定目标唯一众数检验**（CITE Def. 2.2）：对先验固定（与数据独立）的目标 $r$，

$$H_{0,r} := \left\{P : p_r \le \sup_{a \ne r} p_a\right\}, \qquad H_{1,r} := \left\{P : p_r > \sup_{a \ne r} p_a\right\}$$

**序贯认证程序**（Def. 2.3）：$\mathcal{F}_t$-可测的决策序列 $(D_t)$，停止时间 $\tau := \inf\{t : D_t = 1\}$。程序在水平 $\varepsilon$ 上具有 **anytime-valid I 类错误控制**，若

$$\sup_{P \in H_{0,r}} P_P(\tau < \infty) \le \varepsilon \tag{1}$$

注意 $(1)$ 对**任意**数据依赖的停止规则同时成立 —— 这是它与固定样本量检验的根本区别。

---

## 2. CITE：三个随机过程（复现）

### 2.1 (i) pairwise 下注型 e-process

对固定目标 $r$ 与竞争者 $a \ne r$，定义中心化指示量 $Z_i^{(r,a)} := \mathbf{1}\{X_i=r\} - \mathbf{1}\{X_i=a\} \in \{-1,0,+1\}$，下注参数 $\lambda \in (0,1)$：

$$E_t^{(r,a)}(\lambda) := \prod_{i=1}^{t}\left(1 + \lambda Z_i^{(r,a)}\right) = (1+\lambda)^{N_t(r)}(1-\lambda)^{N_t(a)} \tag{2}$$

在 $H_0: p_r \le p_a$ 下这是非负上鞅（NSM），由 **Ville 不等式**：

$$P_{H_0}\left(\sup_t E_t \ge 1/\alpha\right) \le \alpha$$

最优 $\lambda^* = (p_r - p_a)/(p_r + p_a)$ 未知，故在**数据无关**的网格 $\Lambda_{\mathrm{pw}}$ 上混合（凸组合保 NSM）：

$$E_t := \sum_{\lambda \in \Lambda_{\mathrm{pw}}} w_\lambda \, E_t^{(r,\hat a_t)}(\lambda), \qquad \hat a_t := \operatorname*{arg\,max}_{a \in \mathcal{A}_t \setminus \{r\}} N_t(a) \tag{3}$$

> **为什么 $\hat a_t$ 数据依赖却仍然有效**：被 Ville 界定的是**固定竞争者**过程 $E_t^{(r,a^\star)}$，而 $\hat a_t$ 的存在性论证（CITE Thm 4.1 证明）表明「runner-up 单调性」使得 $\hat a_t$ 触发 ⟹ 真零见证 $a^\star$ 也触发。

### 2.2 (ii) $p_r$ 的时间一致下置信界（LCB）

处理**隐藏竞争者**（质量 $> p_r$ 但从未出现过的类别）。对每个候选 $q \in (0,1]$，检验 " $p_r \le q$ "，下注于 $\mathbf{1}\{X_i = r\} - q$：

$$M_t(q) := \sum_{\lambda \in \Lambda_r(q)} v_\lambda \prod_{i=1}^{t}\bigl(1 + \lambda(\mathbf{1}\{X_i=r\} - q)\bigr), \quad \Lambda_r(q) := \{\lambda \in \Lambda_r : \lambda < 1/q\} \tag{4}$$

**闭式推导**（本仓库实现，避免 $O(t)$ 重算）：$N_t(r)$ 次命中贡献 $(1+\lambda(1-q))$，其余 $t - N_t(r)$ 次贡献 $(1-\lambda q)$，故

$$M_t(q) = \sum_{\lambda} v_\lambda \bigl(1+\lambda(1-q)\bigr)^{N_t(r)}\bigl(1-\lambda q\bigr)^{t-N_t(r)}$$

$q \mapsto M_t(q)$ 非增（CITE Lemma C.3），故超水平集是下集，定义良好：

$$L_t(r) := \sup\Bigl(\{0\} \cup \{q \in (0, \hat p_t(r)] : M_t(q) \ge 3/\varepsilon\}\Bigr) \tag{5}$$

满足 $P(\exists t : L_t(r) > p_r) \le \varepsilon/3$（Prop. 3.4）。

### 2.3 (iii) 未见类别上界

质量 $u$ 的类别在 $t$ 次 i.i.d. 抽样后仍未出现的概率为 $(1-u)^t$。加**概率加权 union bound**：

$$U_t := \min\left\{u \in (0,1] : u^{-1}(1-u)^t \le \varepsilon/3\right\} \tag{6}$$

满足 $P(\exists a, \exists t : N_t(a)=0 \text{ 且 } p_a > U_t) \le \varepsilon/3$（Prop. 3.6）。

**数值要点**：$u^{-1}(1-u)^t$ 在 $u$ 上单调递减，可行集是上集；用二分求解，且 $u \to 1$ 时取 $\log 0 = -\infty$ 需特判（否则 `math domain error`）。

### 2.4 CITE 停止规则

$$\tau := \inf\{t \ge 1 : E_t \ge 3/\varepsilon \ \text{且}\ L_t > U_t\} \tag{7}$$

误差三分（pairwise / LCB / unseen）各 $\varepsilon/3$，union bound 得 $\le \varepsilon$。

**停止时间速率**（Thm 4.3），$\delta := p_r - p_{(2)}$：

$$\mathbb{E}[\tau] = O\!\left(\frac{\log(1/\varepsilon) + \log(1/\delta)}{\delta^2} + \frac{\log(1/\varepsilon) + \log(1/p_r)}{p_r}\right)$$

**与 $|\mathcal{A}|$ 无关** —— 这是 CITE 的核心理论突破，并配了匹配的下界（Thm 4.4）。

---

## 3. VCE：把任意验证器变成合法 e-process（本仓库原创）

### 3.1 动机

CITE 认证的是**众数**，不是**正确性**。一个一致地答错的模型会被同样快地认证。而实际系统几乎总有廉价辅助验证器（单元测试、符号检查器、检索重叠分、小奖励模型）。难点在于：验证器分数**既不是 p 值也不是 e 值**，直接指数化或归一化到 $[0,1]$ 会摧毁任何保证。

### 3.2 两步转换

**第一步：分割 conformal p 值。** 设零校准集 $\{c_1,\ldots,c_n\}$（历史上**错误答案**的分数），

$$p = \frac{1 + \#\{i : c_i \ge s\}}{n + 1}$$

在可交换性下 $P(p \le u) \le u$ 对所有 $u$ 成立 —— 这是**无分布**的，不需要对分数分布做任何假设。

**第二步：Vovk-Wang p→e 校准器**，$\kappa \in (0,1)$：

$$e_\kappa(p) = \kappa \, p^{\kappa - 1} \tag{8}$$

**有效性证明**：$e_\kappa$ 在 $(0,1]$ 上非负、非增，且

$$\int_0^1 e_\kappa(u)\,du = \kappa \int_0^1 u^{\kappa-1} du = \bigl[u^\kappa\bigr]_0^1 = 1$$

超均匀的 $p$ 随机占优于 $U[0,1]$，非增函数保序，故 $\mathbb{E}[e_\kappa(p)] \le \mathbb{E}[e_\kappa(U)] = 1$。这正是 e-variable 的定义。$\blacksquare$

### 3.3 乘积而非均值（关键）

跑动**均值** $\frac{1}{t}\sum_{i \le t} e_i$ 在每个固定 $t$ 是 e-variable，但**不是上鞅**，Ville 不适用，偷看仍会破坏保证。

改用**乘积** $W_t := \prod_{i \le t} e_i$。因为轨迹 i.i.d. 且校准集在打分前已冻结，每个 $e_i$ 只是 $X_i$ 的函数，故

$$\mathbb{E}[e_i \mid \mathcal{F}_{i-1}] = \mathbb{E}[e_i] \le 1$$

即 $e_i$ 是**条件 e-variable**，从而 $W_t$ 是 NSM，Ville 在任意数据依赖停止时间下成立。

全部在 log 空间计算（$\log W_t = \sum \log e_i$），避免溢出；超过 $e^{700}$ 时饱和为 $+\infty$，而所有下游用途都是与有限阈值比较，语义不变。

### 3.4 多验证器融合

**算术平均**（凸组合）**无条件**保 e-variable —— 不需要独立性。这对真实场景至关重要：NLI 模型与检索分数高度相关，乘积规则会静默失效。

`VerifierBank` 默认 `rule="mean"`；`rule="product"` 仅在能论证条件独立性时使用，并在审计中记录所用规则。

### 3.5 分批（batching）

每 $b$ 条支持目标的轨迹聚合成**一个**统计量后再校准。批内平均把 conformal 统计量的方差压低约 $1/b$，批间不相交故乘积仍是上鞅。

> ⚠️ 校准集必须用**相同的 $b$** 构建。用单分数校准却用批均值打分，等于比较两个不同统计量，超均匀性静默失效 —— 这是该构造在实践中最常犯的错。本仓库 `verifier_calibration(..., batch=b)` 与 `CertiScaleConfig(vce_batch=b)` 强制对齐，由不变量 I7 守卫。

---

## 4. CertiScale 组合：误差是 max 而非 sum（本仓库原创）

停止规则要求**两个通道同时触发**：

$$\tau := \inf\{t : E_t \ge 3/\varepsilon_{\mathrm{cite}} \ \wedge\ L_t > U_t \ \wedge\ W_t \ge 1/\varepsilon_{\mathrm{vce}}\} \tag{9}$$

**论证**。联合零假设是并集 $H_0 = H_0^{\mathrm{mode}} \cup H_0^{\mathrm{correct}}$：

- **零世界 A**：$r$ 不是唯一众数 ⟹ CITE 通道为真零，$P(\text{CITE 触发}) \le \varepsilon_{\mathrm{cite}}$，故 $P(\tau<\infty) \le \varepsilon_{\mathrm{cite}}$
- **零世界 B**：$r$ 是众数但**错误** ⟹ VCE 通道为真零，$P(\text{VCE 触发}) \le \varepsilon_{\mathrm{vce}}$，故 $P(\tau<\infty) \le \varepsilon_{\mathrm{vce}}$

$$\boxed{\sup_{P \in H_0} P_P(\tau < \infty) \le \max(\varepsilon_{\mathrm{cite}}, \varepsilon_{\mathrm{vce}})}$$

合取结构买到 **max 而非 sum**，故两预算均设为 $\varepsilon$ 即可。**代价**：VCE 使规则更严，不会认证更快 —— 它让认证**更对**。实验显示 FDR 降 21–43 倍，样本成本 +6%~+10%。

---

## 5. AVBA：跨问题预算分配（本仓库原创）

CITE 是单问题的。生产系统面对的是 $K$ 个问题共享预算 $B$。均匀轮询在 3 次采样就收敛的简单题上浪费算力，同时饿死需要 200 次的难题。

**形式化**：这是 **anytime-valid 停止下的预算分配**，是多臂最优臂识别与序贯检验的结合。已有最优解（Successive Halving, Karnin et al. 2013；UCB-E, Audibert & Bubeck 2010；Gittins index, 1979）**不能直接迁移**：它们的奖励是有界臂均值，而这里是**无界尺度的上鞅财富**，且停止规则数据依赖。

AVBA 保留 UCB 式的**乐观原则**，但用 CITE 真正赖以认证的 **log-财富增长率** 驱动：

$$g_k := \frac{\log W_k(t_k) - \log W_k(t_k - w)}{w}, \qquad \text{score}_k = g_k + c \cdot \mathbf{1}\{t_k < w\}\left(1 - \frac{t_k}{w}\right)$$

每轮选 score 最高的 $b$ 个问题采样；已认证的问题释放预算。三种策略同接口可横向对比：`uniform` / `ucb` / `wga`。

---

## 6. 数值实现要点

| 问题 | 现象 | 解法 |
|------|------|------|
| $(1+\lambda)^{N}$ 溢出 | `RuntimeWarning: overflow`，e 值变 `nan` | 全程 log 空间 + `log1p`，`logsumexp` 混合 |
| $U_t$ 在 $u \to 1$ | `math domain error`（$\log 0$） | 特判 $u \ge 1 \Rightarrow -\infty$ |
| $U_t$ 单调性判反 | 恒返回 1.0 | 可行集是**上集**：先查 $\mathrm{viol}(lo)\le 0$，再查 $\mathrm{viol}(hi)>0$ |
| e 值无限增长 | 下游比较失效 | $e^{700}$ 以上饱和为 $+\infty$（语义正确的哨兵） |
| 校准/打分统计量不一致 | conformal 超均匀性静默失效 | `batch` 参数强制对齐 + 不变量 I7 |

---

## 7. 符号表

| 符号 | 含义 |
|------|------|
| $\mathcal{A}$ | 可数（可能无限）答案类别集 |
| $p_a, p_{(k)}$ | 类别概率、降序重排 |
| $N_t(a), \hat p_t(a)$ | 计数、经验频率 |
| $E_t^{(r,a)}(\lambda)$ | pairwise e-process |
| $M_t(q), L_t$ | $p_r \le q$ 的混合 e-process、下置信界 |
| $U_t$ | 未见类别质量上界 |
| $W_t$ | VCE 通道财富（e 值乘积） |
| $\tau$ | 停止时间 |
| $\varepsilon$ | anytime-valid I 类错误预算 |
| $\delta$ | 众数间隙 $p_r - p_{(2)}$ |
| $\kappa$ | Vovk-Wang 校准器指数 |

---

作者：晨星 (CJX0712) · MIT License
