# Fixed Density

**Fixed Density** 是面向 osu!mania 4K 的谱面难度评估算法，其输出以社区长期使用的 dan / 难度标尺为参照。

对每张谱面，算法给出两个拟合值：

| 输出 | 含义 | 标尺 |
| --- | --- | --- |
| `rc_value` | RC 参考难度 | 原始 value（与 dan 标尺同尺，如 `14.06`） |
| `ln_value` | LN 参考难度 | 变换后标尺 `g_ln(value)`（与 tosu 插件显示的 ln 值同尺度） |

算法仅依赖物件的起始时间与长度，不读取 BPM、节拍线、SV 等元数据；凡是能够提供逐轨 `(t, l)` 序列的
前端（osu!、Malody 或自研解析器）均可直接复用。

> 仓库包含三个相互独立的分支：`main` 为算法实现、基准评测、参数标定与结果报告；
> `tosu` 为本算法的 tosu 实时插件（修改自 Leo_Black 的同名插件）；
> `qqbot` 为调用该插件并生成分析图片的 QQ 机器人。

---

## 1. 快速开始

```powershell
python -m pip install -r requirements.txt

# ① 由谱面文件求值
python -m fixed_density.value "examples/Saikoro - far in the blue sky... (1nar) [42].osu"

# ② 由逐轨 (t, l) 序列求值（不读取文件）
python -c "from fixed_density import evaluate; \
cols=[[(0,0),(500,0),(1000,0)],[(250,0),(750,1200)],[],[(100,0)]]; \
print(evaluate(cols, od=8.0))"
```

```text
examples/... [42].osu
    rc_value = 14.0602   ln_value = 0.1816   n = 41.948021   k = 0.008425
```

`evaluate(columns, od, rate, model)` 对应上述第二种调用方式：`columns[k]` 为第 k 轨的
`[(开始时间ms, 长度ms), ...]`（单键长度为 0，按时间升序），返回
`dict(rc_value, ln_value, n, k, chord)`。`--rate 1.5` 表示 DT、`0.75` 表示 HT，
时间戳按 `1/rate` 缩放，与插件采用相同的处理方式。

---

## 2. 目录结构

```
fixed_density/
├─ core.py            算法核心：参数注册表、物件赋值序列算法、LN 通道与映射曲线（仅依赖 numpy）
├─ value.py           求值接口：.osu 或 (t,l) 序列 → rc_value / ln_value（CLI）
├─ reader.py          .osu → 逐轨 (t, l) 序列
├─ benchmark.py       基准运行、曲线拟合、统计与单文件 HTML 报告（CLI）
├─ ln_benchmark.py    LN 参考难度评测报告（CLI）
├─ tune.py            参数标定：Sobol' 初始设计与高斯过程 EI 贝叶斯优化（CLI）
├─ extract.py         基准预处理：.osu → 物件 (t, l) JSON（CLI）
└─ params_final.json  最终模型：算法参数、映射曲线与 LN 标定（取自 tosu 插件）

benchmark/            基准数据集（RC）：data.csv 与 course/jack/speed/stamina/tech/ln 谱面
benchmark-ln/         基准数据集（LN）：data.csv（含 subPattern 标注）与散点诊断图
benchmark_extracted/  benchmark/ 的预处理产物（逐轨 (t,l) JSON，默认数据源，可免去 .osu 解析）
examples/             示例谱面
report/               基于最终模型的评测报告（3 份 HTML）
```

数据流：`benchmark/**.osu` →（`extract.py`）→ `benchmark_extracted/**.json` →（`benchmark.py`）→ `report/*.html`

---

## 3. 算法流程（Fixed Density v1）

以下按实际计算顺序说明。符号约定：`K` 为键数（4K 时 K=4），`t` 为物件起始时间（ms），`ℓ` 为长条长度（ms）。

### 3.1 输入与预处理

1. 解析 `.osu`，得到逐轨 `[(t, ℓ), ...]`：`[HitObjects]` 的 `x` 按 `floor(x·K/512)` 分轨，
   `type` 含 bit 7 视为长条，`ℓ = endTime − t`。
2. 变速处理：`rate ≠ 1` 时，所有 `t` 与 `ℓ` 乘以 `1/rate`（DT、HT 及 lazer 任意正速统一处理）。

### 3.2 RC 通道：物件赋值序列

**① 单轨分组与注入贡献 `contribution`**

逐轨将相同时间戳的物件合并为一组 `(t, size, prev_dt, next_dt)`，其中 `prev/next_dt` 为与同轨
前/后一组的时间差。

- 间距倒数：`prev_inv = 1000/prev_dt`、`next_inv = 1000/next_dt`（无相邻组时记为 0）。
- 单件组的基值：`base = M(prev_inv, next_inv)`。

  其中 `M` 为**合并算子**。当 `soft_beta = β > 0` 时采用 softmax 加权平均（处处光滑、无分段）：

  ```
  w = ½·(1 + tanh(½β(a−b))),      M(a, b) = w·a + (1−w)·b
  ```

  `β→∞` 时收敛于 `max(a,b)`，`β→0` 时收敛于算术平均；`β=0` 时退化为 **Lp 范数**
  `(|a|ᵖ+|b|ᵖ)^(1/p)`。
- 同刻堆叠（`size > 1`）：按超限替代语义累加，惩罚常量为 `OVERLAP_PENALTY = 20000`：

  ```
  base = (size−2)·M(P, P) + M(prev_inv, P) + M(P, next_inv),    P = 20000
  ```
- **配对轨邻域额外贡献**：取同手另一轨（4K 下 1↔2、3↔4）中与 `t` 的距离 `x` 落在 `[−4a, 4a]`
  范围内的键，按对数域高斯核逐键加权求和，其中 `a = 1000/base`：

  ```
  extra = Σ size_m · exp( −½·( log_base(x/a) + extra_b )² )
  contribution = base · (1 + extra_rate · extra)
  ```
- **单轨局部熵乘子**（贡献层）：`contribution *= exp(γ_lane · irr_lane(t))`。`irr` 表示该起音邻域的
  局部间隔多样性，计算方式为对 `log2(间隔)` 做窗口归一化后的高斯核微分熵（KDE 交叉熵），
  窗口 16 个间隔、步长 4，档间线性插值回逐键；`γ_lane = irr_gain_lane`。

**② 衰减赋值序列**

对每轨沿时间递推（相邻物件间隔 `dt`）：

```
value ← value · 0.5^(dt / H) + contribution        # H = half_life_ms
```

该递推与以 1 ms 为步长的 IIR 滤波数学等价，但无需分配与谱面时长成正比的内存。

**③ 两级合并**

- **第一级（配对双轨）**：将 (1,2)、(3,4) 两条序列在时间戳并集上采样（保持同一衰减递推），
  逐时刻用 `M` 合并；合并后该时刻的 `count` 为两轨在该时刻的物件数之和。
- **第二级（配对序列之间）**：将每个配对序列视为“赋值处发生突变的时间衰减函数”，
  在彼此时间戳上采样后逐时刻用 `M` 合并，`count` 相加。
  由此，同一时间戳上“相同值”的个数恒等于该时刻的物件数，保证后续分位数统计的口径正确。
- **全谱局部熵乘子**（最终合并层）：`v *= exp(γ_global · irr_global(t))`，`γ_global = irr_gain_global`。

**④ 求 n**

将合并序列按 `count` 展开为逐物件赋值列表，求 `n` 满足：

```
mean( min(1, n / a_i) ) = quantile
```

采用“排序 + `1/a` 前缀和 + 二分”，单步复杂度 `O(log N)`；二分上界初值取 `a` 的 `quantile` 分位数。

**⑤ 反解 value**

```
chord = 物件总数 / 去重起音数        # 和弦度（平均同时按键数）
g_pred = ln(n) − C·chord − E·ln(N)   # N = 物件总数；最终模型 C = 0
rc_value = g⁻¹(g_pred)
```

`g` 为**连续映射曲线**，由 7 个衔接点 `(v0ᵢ, sᵢ)` 与 8 段斜率 `σⱼ` 参数化，基于 `softplus` 转折基：

```
g(v) = c + Σ_j σ_j · ψ_j(v),     ψ 由 φ_i(v) = s_i·softplus((v−v0_i)/s_i) 组合得到
```

当 `v→±∞` 时 `g` 具有线性渐近线，全程 C∞ 光滑；`σ_j > 0` 保证严格单调，故可用二分法求逆 `g⁻¹`。
斜率上下界 `[0.06, 0.25]` 用于抑制某些分段过平或过陡。低 value 段另设四个转折点，
使该区间不再依赖单一直线外推。

### 3.3 LN 通道：`ln_value`

LN 的难度信息主要位于物件起始时间之外，由以下结构量描述（取全谱 LN 事件上的均值）：

| 量 | 含义 |
| --- | --- |
| `S_rel` | **抓准**：同轨相邻松手（`t+ℓ`）间距的倒数 |
| `S_jack` | **控手**：松手至同轨下一次按下的间距倒数 |
| `S_cross` | **跨轨**：按住期间同手另一轨动作数超出其本地基线的速率 |
| `S_hold` | **保持**：保持时长 / 该轨物件平均间隔（无量纲，表示跨越多少个本地节拍） |

计算中还引入三个与判定相关的修正：

- 长度权重 `u = smoothstep(ℓ; lo, hi)`，`lo = 20 + 0.4·(W_head+W_tail)`、`hi = 1.3·lo`，
  其中 `W_head = (64−3·OD)/2`、`W_tail = 1.5·W_head`；过短的长条按单键处理；
- 长度因子 `h = ℓ/(ℓ+40)`，作用于 `S_rel` 与 `S_jack`，以恢复“有效长条内越长越难”的单调性；
- 松手吸收折扣 `w = 1 − 0.25·ā`，其中 `ā` 为全谱松手“被邻近起音吸收”程度的均值。

最后乘以长条占比的饱和曲线（Hill 型叠加线性 toe，满足 `F(0)=0`、`F(1)=1`、单调不减）：

```
ρ = 加权长条数 / 物件总数
k = F(ρ) · (γ_rel·S_rel + γ_jack·S_jack + γ_cross·S_cross + γ_hold·S_hold)
ln_value = a · ln(1 + b·k·n)          # n 即 RC 通道求得的同一个 n
```

`ρ = 0` 时 `k = 0`，从而 `ln_value = 0`，即“几乎没有 LN 的谱面显示 0”。`ln(1+·)` 的含下界形式
保证 `k·n → 0` 时输出同样趋于 0。

### 3.4 参数注册表

全部可调参数统一注册于 `core.py` 的 `ParamSpec`：

- `python -m fixed_density.benchmark --list-algos` 可列出所有参数及默认值；
- 任意参数均可通过 `--set NAME=VALUE` 或 `--params-json <file>` 覆盖，报告中的参数表自动适配；
- 新增算法只需实现 `func(columns, params) -> n` 并用 `@register_algorithm` 注册，
  基准运行、曲线拟合、统计、报告与参数标定流程即可复用。

---

## 4. 最终模型与复现方式

算法参数与映射曲线均冻结于 [`params_final.json`](fixed_density/params_final.json)，取自 `tosu` 分支插件内
`js/fixedDensity/index.js` 的同名常量。主要取值（RC）如下：

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

生成 `report/` 下三份报告的命令如下：

```powershell
# RC 报告（默认排除 ln 目录，见 §6.3）
python -m fixed_density.benchmark --no-irr --params-json fixed_density/params_final.json

# LN 报告（γ 与 LN 标定 a、b 均固定为最终模型取值）
python -m fixed_density.ln_benchmark --no-irr --params-json fixed_density/params_final.json `
  --fix-gamma --gamma-rel 0.1 --gamma-jack 0.4 --gamma-cross 0 --gamma-hold 0.2 `
  --gamma-exp 6 --rho-c 0.33 --rho-lin 0.1 `
  --ln-a 5.44200375179058 --ln-b 0.0960408821250538
```

---

## 5. 参数标定

`tune.py` 提供参数标定流程，涵盖映射曲线的拟合与算法超参数的搜索：

```powershell
python -m fixed_density.tune --dry-run                 # 仅打印参数空间与预算
python -m fixed_density.tune --no-irr --max-evals 200  # 默认：Sobol' 16 个初始点 + GP-EI 至多 200 次
python -m fixed_density.tune --resume                  # 从 tune_log.jsonl 断点续跑
python -m fixed_density.tune --space mapped --workers 1
```

- **优化目标**：与 `benchmark.py` 采用相同的评价流程（计算 n → 分层标定集拟合 g → 反算 value′ →
  统计 dvalue → RMSE），可通过 `--space original|mapped`、`--filter-min/--filter-max`、
  `--exclude-folder` 切换评价口径。
- **搜索方式**：在 15 维参数空间（半衰期、分位数、额外贡献三参数、softmax 系数、三个熵增益、
  LN 通道 6 参数）上，先用 Sobol' 低差异序列生成初始设计，再用高斯过程（Matern-ARD + 白噪声）的
  Expected Improvement 提议新点；结果写入 JSONL 并缓存，重复点不重复计算。
- **配套工具**：`ln_benchmark.py` 包含 LN 倍率权重 `(γ_rel, γ_jack, γ_cross, γ_hold)` 的网格搜索
  （`--fix-gamma` 可改用给定值）；`extract.py` 负责将 `.osu` 预处理为可快速读取的 JSON。

---

## 6. 评测结果与局限

### 6.1 RC 通道（排除 ln 目录，N = 643）

原始 value 尺度下 **RMSE = 0.444**，均值偏差 +0.024；|d| ≤ 0.2 占 **37.3%**，≤ 0.5 占 76.8%，
大于 1 的仅占 3.9%。

| folder | N | 偏差均值（exp−pred） | RMSE |
| --- | --- | --- | --- |
| course（段位单曲，留出） | 34 | +0.384 | 0.533 |
| jack | 221 | +0.037 | 0.411 |
| speed | 147 | −0.006 | 0.444 |
| stamina | 130 | −0.026 | 0.433 |
| tech | 111 | −0.012 | 0.489 |

映射后 value 尺度（拟合空间）下 RMSE = 0.238，均值偏差 +0.011，未出现 |d| > 1 的样本。

### 6.2 LN 通道（N = 102 张 LN 谱，含 64 张段位单曲）

原始 value 尺度下 **RMSE = 0.916**（非合成谱 38 张：0.599；段位单曲 64 张：1.060）；
变换后尺度下 RMSE = 0.766。若改用 RC 通道评价同一批 LN 谱，RMSE 为 3.398、偏差 −3.0；
LN 通道将 RMSE 降至约其 1/3.7。

### 6.3 两类谱面需分别评价

`benchmark/data.csv` 中的 `ln` 目录为纯 LN 谱，其难度由 `ln_value` 承担，`rc_value` 本应处于较低水平。
若将 ln 目录一并计入 RC 统计，RMSE 会由 0.444 升至 1.323。因此 `benchmark.py` 默认使用
`--exclude-folders ln`；如需包含全部谱面，可传 `--exclude-folders ""`。

### 6.4 基准数据的局限

现有误差主要来自基准标注端，下列因素限制了评测精度的上限：

1. **标注粒度**：大量难度值来自段位（dan）标定，而段位为整数分档。即使模型完全正确，
   量化误差仍会贡献约 `1/√12 ≈ 0.29` 的 RMSE 下限，|d| ≤ 0.2 一类的紧公差因此受到固有限制。
2. **构造性耦合**：数据集内 OD 与 value 的整体相关性（约 +0.28）主要来自 course 的构造方式；
   在 8+ 难度段二者几乎无关。因此不宜将 OD 作为线性特征，强行加入会损害泛化。
3. **样本分布不均**：低 value 段（1~6）几乎只有段位单曲一类样本；`ln` 目录的 102 张中有 64 张
   为马拉松段位单曲，真实竞技 LN 谱偏少。
4. **pattern 覆盖不均**：`benchmark-ln/` 与 `benchmark/` 的 subPattern 分布并不均匀，
   控手/抓准桶（release + inverse）的误差明显高于密度桶（LN 尺度 1.216 对 0.864）。
5. **评测集与标定集不完全重叠**：参数标定所用集合与本基准集的构成存在差异，
   因此本报告更接近于独立复核，而非训练误差。
6. **谱面级系统性偏差**：多组参数加权合并无法抵消 pattern 级的系统性偏差，
   说明残差中有相当部分来自算法结构（如 LN 结构量是否完备、合并算子的形态），
   而非超参数选择。

上述问题多属数据层面。随着标注数据在数量与分布上的完善，核心算法中的特征项与合并算子仍有
进一步调整的空间。

---

## 7. 相关分支

| 分支 | 内容 |
| --- | --- |
| `main`（本分支） | 算法实现、基准数据集、参数标定与评测报告 |
| `tosu` | 本算法的 tosu 实时插件移植（修改自 Leo_Black 的同名插件），最终模型参数硬编码于此 |
| `qqbot` | 调用该插件的 QQ 机器人，`/rework` 指令返回分析卡片图片 |

两个分支使用同一套模型参数：`main` 的 `params_final.json` 与 `tosu` 插件内的常量一一对应
（同参数、同映射曲线、同 LN 标定）。

---

## 8. 致谢

- 插件原型：[Leo_Black 的 ManiaMapAnalyser](https://github.com/LeoBlackMT/osumania_map_analyser)
  （`tosu` 分支在此基础上修改）。
- 基准数据主要来自 [Leo_Black 的 VSRG Dan Estimation Benchmark](https://github.com/LeoBlackMT/VSRG-DanEstimation-Benchmark)，谨此致谢。
- 基准数据中的段位单曲来自 Signicial 等 dan 系列；谱面版权归原作者所有，此处仅用于难度算法研究。
