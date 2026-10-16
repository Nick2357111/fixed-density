# Fixed Density

**Fixed Density v1** —— 一套面向 osu!mania 4K 的**谱面难度计算算法**（不是 pp 系统，而是输出与社区 dan / 难度标尺对齐的“难度值”）。

给定一张谱面，它输出两个数：

| 输出 | 含义 | 标尺 |
| --- | --- | --- |
| `rc_value` | 米（rice / 单键流）难度 | 原始 value（与 dan 标尺同尺，如 `14.06`） |
| `ln_value` | LN（长条）参考难度 | 变换后标尺 `g_ln(value)`（与 tosu 插件显示一致） |

算法**只依赖物件的「时间戳 + 长度」**，不需要 BPM、节拍线、SV 或任何其它元数据，因此任何能输出
「逐轨 `(t, l)` 元组列表」的前端（osu! / Malody / 自研解析器）都能直接复用。

> 本仓库是 `main` 分支：算法本体 + benchmark + 调参 + 分析报告。
> 另有两个分支：`tosu`（算法移植成的 tosu 实时插件）、`qqbot`（调用该插件的 QQ 机器人）。

---

## 1. 快速开始

```powershell
python -m pip install -r requirements.txt

# ① 从谱面文件到 value（完整工作流：解析 → 计算 → 输出）
python -m fixed_density.value "examples/Saikoro - far in the blue sky... (1nar) [42].osu"

# ② 从「逐轨 (t, l) 元组列表」开始直到 value（不经过文件）
python -c "from fixed_density import evaluate; \
cols=[[(0,0),(500,0),(1000,0)],[(250,0),(750,1200)],[],[(100,0)]]; \
print(evaluate(cols, od=8.0))"
```

```text
examples/... [42].osu
    rc_value = 14.0602   ln_value = 0.1816   n = 41.948021   k = 0.008425
```

`evaluate(columns, od, rate, model)` 就是**第 ② 条入口**：入参 `columns[k]` 是第 k 轨的
`[(开始时间ms, 长度ms), ...]`（单键长度 0，按时间升序），返回 `dict(rc_value, ln_value, n, k, chord)`。
`--rate 1.5` 即 DT、`0.75` 即 HT（时间戳按 `1/rate` 缩放，与插件一致）。

---

## 2. 目录结构

```
fixed_density/
├─ core.py            算法核心：参数注册表 + 物件赋值序列算法 + LN + 映射曲线（仅依赖 numpy）
├─ value.py           端到端求值入口：.osu 或 (t,l) 列表 → rc_value / ln_value（CLI）
├─ reader.py          .osu → 逐轨 (t, l) 列表
├─ benchmark.py       benchmark 运行 / 曲线拟合 / 统计 / 单文件 HTML 报告（CLI）
├─ ln_benchmark.py    LN 参考难度 benchmark 报告（CLI）
├─ tune.py            全量参数调优：Sobol' 初始设计 + 高斯过程 EI 贝叶斯优化（CLI）
├─ extract.py         benchmark 预处理：.osu → 物件 (t, l) JSON（CLI）
└─ params_final.json  冻结的最终模型：算法参数 + 映射曲线 + LN 标定（与 tosu 插件逐字一致）

benchmark/            基准集①（RC）  data.csv + course/jack/speed/stamina/tech/ln 的 .osu
benchmark-ln/         基准集②（LN）  data.csv（含 subPattern 标注）+ 散点诊断图
benchmark_extracted/  benchmark/ 的预处理产物（逐轨 (t,l) JSON，默认数据源，免 .osu 解析）
examples/             5 张示例谱面
report/               使用最终参数生成的分析报告（3 份 HTML）
```

数据流：`benchmark/**.osu` --`extract.py`--> `benchmark_extracted/**.json` --`benchmark.py`--> `report/*.html`

---

## 3. 算法流程（Fixed Density v1）

下面按**实际参与计算**的顺序讲。符号：`K` 键数（4K 时 K=4）、`t` 物件开始时间（ms）、`ℓ` 长条长度（ms）。

### 3.1 输入与预处理

1. 解析 `.osu` → 逐轨 `[(t, ℓ), ...]`（`[HitObjects]` 的 `x` 按 `floor(x·K/512)` 分轨，
   `type` 含 bit 7 即长条、`ℓ = endTime − t`）。
2. 变速：`rate ≠ 1` 时所有 `t`、`ℓ` 乘 `1/rate`（DT/HT/lazer 任意正速统一处理）。

### 3.2 RC 通道：物件赋值序列

**① 单轨分组与注入贡献 `contribution`**

逐轨把**相同时间戳**的物件合并成一组 `(t, size, prev_dt, next_dt)`（`prev/next_dt` 为与同轨前/后一组的时间差）。

- 间距倒数：`prev_inv = 1000/prev_dt`、`next_inv = 1000/next_dt`（无邻件记 0）。
- 单件组的基值：`base = M(prev_inv, next_inv)`

  其中 `M` 是**合并算子**：参数 `soft_beta = β > 0` 时为 softmax 加权平均（处处光滑、无分段）

  ```
  w = ½·(1 + tanh(½β(a−b))),      M(a, b) = w·a + (1−w)·b
  ```

  `β→∞` 收敛到 `max(a,b)`、`β→0` 收敛到算术平均。代码同时保留 `β=0` 时退回 **Lp 范数**
  `(|a|ᵖ+|b|ᵖ)^(1/p)` 的分支（最终模型 `β = 0.706887 > 0`，Lp 分支不参与）。
- 同刻堆叠（`size > 1`）：按“超限替代”语义累加，惩罚常量 `OVERLAP_PENALTY = 20000`

  ```
  base = (size−2)·M(P, P) + M(prev_inv, P) + M(P, next_inv),    P = 20000
  ```
- **配对轨邻域额外贡献**：取同手另一轨（4K 下 1↔2、3↔4）中与 `t` 的距离 `x` 落在 `[−4a, 4a]` 内的键，
  逐键按对数域上的高斯核加权求和（`a = 1000/base`，即该键当前的等效时间尺度）

  ```
  extra = Σ size_m · exp( −½·( log_base(x/a) + extra_b )² )
  contribution = base · (1 + extra_rate · extra)
  ```
- **单轨局部熵乘子**（贡献层）：`contribution *= exp(γ_lane · irr_lane(t))`，
  `irr` 为该起音邻域的**局部间隔多样性**（窗 16 个间隔、步长 4，对 `log2(间隔)` 做窗口归一化后
  用高斯核估计的微分熵 / KDE 交叉熵，档间线性插值回逐键）。γ_lane = `irr_gain_lane`。

**② 衰减赋值序列**

对每轨，沿时间递推（相邻物件间隔 `dt`）：

```
value ← value · 0.5^(dt / H) + contribution        # H = half_life_ms
```

即把原 1ms 步长的 IIR 时间序列改成「按物件间隔递推」，数学等价但无需分配百万级数组。

**③ 两级合并**

- **第一级（配对双轨）**：把 (1,2)、(3,4) 两条序列在**时间戳并集**上采样（衰减递推保持一致），
  逐时刻用 `M` 合并；合并后该时刻的 `count` = 两轨该时刻物件数之和。
- **第二级（各配对序列之间）**：把每个配对序列视为「赋值处发生突变的时间衰减函数」，
  在彼此时间戳上采样后逐时刻 `M` 合并，`count` 相加。
  这样**同一时间戳的“相同值”个数恒等于该时刻的物件数**（保证后续分位数统计口径正确）。
- **全谱局部熵乘子**（最终合并层）：`v *= exp(γ_global · irr_global(t))`，γ_global = `irr_gain_global`。

**④ 求 n**

把合并序列按 `count` 展开为逐物件赋值列表，求 `n` 使

```
mean( min(1, n / a_i) ) = quantile
```

采用「排序 + `1/a` 前缀和 + 二分」，每步 `O(log N)`；二分上界初值取 `a` 的 `quantile` 分位数。

**⑤ 反解 value**

```
chord = 物件总数 / 去重起音数        # 和弦度（平均同时按键数）
g_pred = ln(n) − C·chord − E·ln(N)   # N = 物件总数；最终模型 C = 0
rc_value = g⁻¹(g_pred)
```

`g` 是**连续映射曲线**：以 7 个衔接点 `(v0ᵢ, sᵢ)` 与 8 段斜率 `σⱼ` 参数化，由 `softplus` 转折基构成

```
g(v) = c + Σ_j σ_j · ψ_j(v),     ψ 由 φ_i(v) = s_i·softplus((v−v0_i)/s_i) 组合得到
```

v→±∞ 时 `g` 具有**线性渐近线**、全程 C∞ 光滑；`σ_j > 0` 保证**严格单调**（故可二分求逆 `g⁻¹`）。
斜率上下界 `[0.06, 0.25]` 抑制某段过平/过陡。低 value 段另设四个转折点，使低段不再靠单一直线外推。

### 3.3 LN 通道：`ln_value`

LN 的困难藏在「物件头时间戳之外」，由以下结构量刻画（全谱 LN 事件上的均值）：

| 量 | 物理含义 |
| --- | --- |
| `S_rel` | **抓准**：同轨相邻**松手**（`t+ℓ`）间距倒数 |
| `S_jack` | **控手**：松手 → 同轨下一次按下的间距倒数 |
| `S_cross` | **跨轨**：按住期间同手另一轨动作数超出其本地基线的速率 |
| `S_hold` | **保持**：保持时长 / 该轨物件平均间隔（无量纲，跨多少个本地节拍） |

三个折扣/门（都与“判定”有关）：

- 长度权重 `u = smoothstep(ℓ; lo, hi)`，`lo = 20 + 0.4·(W_head+W_tail)`、`hi = 1.3·lo`，
  其中 `W_head = (64−3·OD)/2`、`W_tail = 1.5·W_head`——**过短的长条按单键处理**；
- 长度因子 `h = ℓ/(ℓ+40)`（作用在 `S_rel`/`S_jack` 上，恢复“有效长条内越长越难”的单调性）；
- 松手吸收折扣 `w = 1 − 0.25·ā`（`ā` 为全谱松手“被邻近起音吸收”的程度均值）。

再乘「长条占比」的饱和曲线（Hill + 线性 toe，`F(0)=0`、`F(1)=1`、单调不减）：

```
ρ = 加权长条数 / 物件总数
k = F(ρ) · (γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross + γ_hold·S_hold)
ln_value = a · ln(1 + b·k·n)          # n 即 RC 通道求得的同一个 n
```

`ρ = 0` 时 `k = 0 ⇒ ln_value = 0`，天然满足“几乎没有 LN 的谱面显示 0”。`ln(1+·)` 的**含下界**形式
保证 `k·n → 0` 时输出也 → 0。

### 3.4 被参数取消的分支（保留但不参与）

最终参数下若干分支被显式“取消”，代码**完整保留**以便后续实验（这也是调参脚本仍在搜它们的原因）：

| 分支 | 取消原因 |
| --- | --- |
| RC 的 Lp 范数合并、`merge_power` / `merge2_power` / `contrib_power` | `soft_beta > 0`，改用 softmax 合并 |
| 跨轨间距混合 `cross_lane_mix` | = 0，贡献间距只取同轨 |
| 全谱熵@贡献层 `irr_local_gain`、轨道对熵@配对合并层 `irr_gain_pair` | = 0 |
| LN 通道（松手流/占用，`ln_gain` 及其子参数） | `ln_gain = 0`，LN 难度改由 §3.3 的 `ln_value` 承载 |
| 拟合模型 irr 项 `C` | = 0（`--no-irr`，irr 信息由 §3.2 的局部熵增益层承担） |
| LN 倍率 `γ_cross` | = 0 |

### 3.5 参数注册表

所有可调参数统一注册在 `core.py` 的 `ParamSpec` 里，因此：

- `python -m fixed_density.benchmark --list-algos` 可列出全部参数与默认值；
- 任意参数都能用 `--set NAME=VALUE` 或 `--params-json <file>` 覆盖，报告参数表自动适配；
- 新增算法只需写一个 `func(columns, params) -> n` 并用 `@register_algorithm` 注册，
  benchmark / 拟合 / 统计 / 报告 / 调参全流程自动复用。

---

## 4. 最终模型与复现

**算法参数与映射曲线全部冻结在 [`params_final.json`](fixed_density/params_final.json)**，与 `tosu` 分支插件内
`js/fixedDensity/index.js` 的常量**逐字一致**。关键取值（RC）：

```
half_life_ms = 306.625      quantile = 0.91162709
extra_rate = 2.0524656      extra_b = 2.3379955     extra_log_base = 4.012384
soft_beta = 0.706887        irr_gain_lane = 0.1     irr_gain_global = 0.06
ln_mult_rel = 0.1  ln_mult_jack = 0.4  ln_mult_hold = 0.2  (γ_cross = 0)
ln_mult_exp = 6.0  ln_mult_rho_c = 0.33  ln_mult_rho_lin = 0.1  ln_absorb_gain = 0.25
LN 标定：a = 5.44200375179058   b = 0.0960408821250538
映射曲线：8 段斜率 σ = [0.25, 0.06, 0.25, 0.06, 0.2307, 0.06, 0.0737, 0.1786]
         衔接点 = [(0.3,1.6), (2.45,1.6), (3.85,0.9), (4.5,2.0), (7.1,1.6), (12.05,0.9), (12.65,0.9)]
         C = 0,  E = −0.032905,  D = −0.028511,  R²(log) = 0.9754
```

复现报告（`report/` 下的 3 份 HTML 即由以下命令生成）：

```powershell
# RC 报告（默认排除 ln 目录，见 §6.3）
python -m fixed_density.benchmark --no-irr --params-json fixed_density/params_final.json

# LN 报告（γ 与 LN 标定 a、b 均固定为最终模型取值）
python -m fixed_density.ln_benchmark --no-irr --params-json fixed_density/params_final.json `
  --fix-gamma --gamma-rel 0.1 --gamma-jack 0.4 --gamma-cross 0 --gamma-hold 0.2 `
  --gamma-exp 6 --rho-c 0.33 --rho-lin 0.1 `
  --ln-a 5.44200375179058 --ln-b 0.0960408821250538
```

> 用最终参数重跑时，`benchmark.py` **自行拟合出的映射曲线与上表逐位相同**
> （σ = 0.2500/0.0600/0.2500/0.0600/0.2307/0.0600/0.0737/0.1786，E = −0.032905，D = −0.028512），
> 说明「拟合流程 + 冻结算法参数」与插件内硬编码的模型完全等价。

---

## 5. 调参

`tune.py` 是**全量调参**主场（拟合调参与变量调参都在这里）：

```powershell
python -m fixed_density.tune --dry-run                 # 只打印参数空间与预算
python -m fixed_density.tune --no-irr --max-evals 200  # 默认：Sobol' 16 初始点 + GP-EI 至多 200 次
python -m fixed_density.tune --resume                  # 从 tune_log.jsonl 断点续跑
python -m fixed_density.tune --space mapped --workers 1
```

- **目标**：与 `benchmark.py` 完全同构（算 n → 分层标定集拟合 g → 反算 value′ → 统计 dvalue → RMSE），
  可用 `--space original|mapped`、`--filter-min/--filter-max`、`--exclude-folder` 切换口径。
- **搜索**：`tune_params` 的 15 维参数空间（半衰期、分位数、额外贡献三参数、softmax 系数、三个熵增益、
  LN 通道 6 参数）在单位超立方体上用 Sobol' 低差异序列做初始设计，再用高斯过程（Matern-ARD + 白噪声）
  的 Expected Improvement 提议新点；结果写 JSONL 并缓存，重复点零成本跳过。
- **辅助脚本**：`ln_benchmark.py` 内含 LN 倍率权重 `(γ_rel, γ_jack, γ_cross, γ_hold)` 的网格搜索
  （`--fix-gamma` 可改用给定值）；`extract.py` 负责把 `.osu` 预处理成免解析的 JSON。

---

## 6. 精度现状与 benchmark 的不完备性

### 6.1 RC 通道（排除 ln 目录，N = 643）

原始 value 口径：**RMSE = 0.444**，均值偏差 +0.024，|d| ≤ 0.2 占 **37.3%**、≤ 0.5 占 76.8%、> 1 仅 3.9%。

| folder | N | exp−pred 均值 | RMSE |
| --- | --- | --- | --- |
| course（段位单曲，留出） | 34 | +0.384 | 0.533 |
| jack | 221 | +0.037 | 0.411 |
| speed | 147 | −0.006 | 0.444 |
| stamina | 130 | −0.026 | 0.433 |
| tech | 111 | −0.012 | 0.489 |

映射 value 口径（拟合所在空间）：RMSE = 0.238，均值偏差 +0.011，**误判（|d|>1）为 0**。

### 6.2 LN 通道（N = 102 LN 谱，含 64 张段位单曲）

原始 value 口径 **RMSE = 0.916**（非合成谱 38 张：0.599；段位单曲 64 张：1.060）；
变换后口径 RMSE = 0.766。相较直接用 RC 口径评价 LN 谱（RMSE 3.398、偏差 −3.0），LN 通道把误差压掉 3.7 倍。

### 6.3 为什么“整体 RMSE”要分开看

`benchmark/data.csv` 里 `ln` 目录是**纯 LN 谱**，它们的难度由 `ln_value` 承载，
其 `rc_value` 本就应该很小。若把 ln 目录一并塞进 RC 口径统计，RMSE 会从 **0.444 虚高到 1.323**；
这就是 `benchmark.py` 默认 `--exclude-folders ln` 的原因（`--exclude-folders ""` 可包含全部）。

### 6.4 benchmark 的不完备性（存在误差的根本原因）

当前算法**在数据充足时仍可能做涉及核心算法内容的进一步修改**，因为这些误差主要来自标注端：

1. **标注粒度**：大量难度值来自段位（dan）标定，段位本身是**整数分档**；即使模型完全正确，
   量化误差也会贡献约 `1/√12 ≈ 0.29` 的 RMSE 下限，紧公差（如 |d| ≤ 0.2）先天受限。
2. **构造性耦合**：数据集内 OD 与 value 的整体相关性（约 +0.28）主要来自 course 的构造方式；
   在 8+ 难度段二者几乎无关，因此**不能**把 OD 当作线性特征——强行加入会损害泛化。
3. **样本分布不均**：低 value 段（1~6）几乎只有段位单曲这一种样本类型；
   `ln` 目录里 102 张有 64 张是马拉松段位单曲，真实竞技 LN 谱偏少。
4. **pattern 覆盖不均**：`benchmark-ln/` 与 `benchmark/` 的 subPattern 分布并不均匀，
   控手/抓准桶（release + inverse）的误差明显高于密度桶（LN 口径 1.216 vs 0.864）。
5. **评测集与调参集不完全重叠**：调参阶段的集合与本基准集的构成存在差异，
   因此本报告更像是**独立复核**，而非训练误差。
6. **chart 级系统性偏差**：多组参数加权合并无法抵消 pattern 级系统性偏差，
   说明残差里有相当一部分来自**算法结构**（如 LN 结构量是否完备、合并算子的形态），
   而不是超参数选择——这正是“未来数据充足时可能改核心算法”的方向。

---

## 7. 与其它分支的关系

| 分支 | 内容 |
| --- | --- |
| `main`（本分支） | 算法本体（Python）+ 两份 benchmark + 调参 + 分析报告 |
| `tosu` | 把本算法移植成的 tosu 实时插件（`ManiaMapAnalyser`，修改自 Leo_Black 的同名插件），**最终参数即硬编码于此** |
| `qqbot` | 调用该插件的 QQ 机器人（发送 `/rework` 返回分析卡片图片） |

`main` 的 `params_final.json` 与 `tosu` 插件内的常量一一对应：
用 `python -m fixed_density.value <map.osu>` 得到的 `rc_value / ln_value`
与插件卡片上的两个星级徽章一致（同参数、同映射曲线、同 LN 标定）。

---

## 8. 致谢

- 插件原型：[Leo_Black 的 ManiaMapAnalyser](https://github.com/)（`tosu` 分支在此基础上修改）。
- 基准数据中的段位单曲来自 Signicial 等 dan 系列；谱面版权归原作者所有，此处仅用于难度算法研究。
