# ManiaMapAnalyser · Fixed Density v1（修改版）

> **本插件修改自 Leo_Black 的同名插件 [ManiaMapAnalyser](https://github.com/LeoBlackMT/osumania_map_analyser)（v2.0.0）。**
> 原插件是给 **tosu** 用的 osu!mania 实时信息面板（4K/6K/7K），功能包括难度估算、
> Pattern/MSD 分析、图表与暂停检测等。本仓库在原插件基础上**新增了 Fixed Density v1 难度计算与显示**，
> 其余功能与上游一致。
>
> 上游作者：**Leo_Black**（`metadata.txt` 中的 Author / authorLinks 未改动）。
> 本修改版的分支说明与算法本体见同一仓库的 `main` 分支。

---

## 1. 这个分支做了什么

在原插件里加了一个**自包含**的难度模块 `js/fixedDensity/`，并把它接到显示管线中：
当插件的 **Top-left Capsule Text（`srText`）设为 `Fixed Density v1`** 时，
卡片左上角不再显示其它算法的星数，而是显示 **Fixed Density 的 rc_value**，
并在其右侧并排显示 **ln_value**（LN 参考难度），两者下方各带一行段位标签
（rc：`rf4` … `α+`；ln：`ln4` … `ln15-`）。

`rc_value` 与 `ln_value` 的定义、算法流程与精度评估见本仓库 `main` 分支的 README；
`main` 分支的 `fixed_density/params_final.json` 与本文档第 4 节列出的常量**逐字一致**，
因此插件与 Python 实现的输出相同（同参数、同映射曲线、同 LN 标定）。

---

## 2. 与上游的差异（修改内容清单）

### 2.1 新增文件

| 文件 | 说明 |
| --- | --- |
| `js/fixedDensity/index.js` | **Fixed Density v1 算法本体**。DOM-free / state-free，Node 与浏览器一致。导出 `calculateFixedDensity(source, rate)` → `{ rcValue, lnValue }`、`fixedDensityRcLabel(value)`、`fixedDensityLnLabel(value)`。冻结参数与映射曲线以常量形式硬编码在文件顶部。 |

### 2.2 改动文件

| 文件 | 改动 |
| --- | --- |
| `js/pipeline/runAnalysisPipeline.js` | 引入 `calculateFixedDensity`；`options` 新增 `withFixedDensity`；新增第 7b 步：复用同一 `parsed` 实例计算并输出 `fixedDensityValue` / `fixedDensityLnValue` / `fixedDensityError`（软失败，不进 `errors[]`）；按 `options.speedRate` 缩放时间戳（DT/HT/lazer 任意正速）。 |
| `js/app/analysis.js` | `needComputed.fixedDensity`（仅当 `srText === "Fixed Density v1"` 时为真）；快照缓存键升级为 `star-v3`（旧的 star 快照缺 `fixedDensity` 字段须失效重算）；调用管线时传 `withFixedDensity`；渲染时调用 `showFixedDensityValue` / `showFixedDensityLnValue`；`resetReworkDisplay` 中调用 `hideFixedDensityLnValue`。 |
| `js/app/display.js` | 新增 `showFixedDensityValue(value)`、`showFixedDensityLnValue(value)`、`hideFixedDensityLnValue()`：写入主星等元素 `#rework-star` 与 LN 元素 `#fixed-density-ln`，并按 rc/ln 段位标签着色（含 high-contrast / 单位徽章等原有样式逻辑）。 |
| `js/app/appContext.js` | 新增 4 个 DOM 句柄：`fixedDensityLnEl`、`fixedDensityLnColEl`、`fixedDensityRcLabelEl`、`fixedDensityLnLabelEl`。 |
| `js/parser/settingsParser.js` | 把设置里的 `"fixed density v1"` / `"fixeddensityv1"` 归一化为规范名 `"Fixed Density v1"`（与 `config.js` 的选项一致）。 |
| `config.js` | `srText` 选项数组新增 `"Fixed Density v1"`。 |
| `settings.json` | `srText` 默认选项列表包含 `"Fixed Density v1"`。 |
| `index.html` | 星等区域新增 4 个元素：`#fixed-density-rc-label`、`#fixed-density-ln-col`、`#fixed-density-ln-label`、`#fixed-density-ln`（默认 `hidden`）。 |
| `styles/star.css` | 新增 Fixed Density 双框（rc / ln 并排）与段位文字说明的样式。 |
| `metadata.txt` | `Notes` 追加一行，标明本目录是含 Fixed Density v1 的修改版。 |

> 除以上文件外，其余目录（`js/estimator/`、`js/ett/`、`js/interlude/`、`js/patterns/`、`js/app/` 其它文件、
> `presets/`、`styles/` 其它文件等）保持上游原样。

---

## 3. 安装与使用

1. 把**本目录整体**放到 tosu 的插件目录下（默认 `tosu/plugins/`，与上游插件的安装方式相同；
   tosu 读取 `metadata.txt` 识别插件）。
2. 启动 tosu，在插件设置里把 **Top-left Capsule Text** 设为 `Fixed Density v1`
   （等价于设置 `settings.json` 的 `srText` 为该值）。
3. 打开任意 4K 谱面：左上角显示 rc_value，右侧显示 ln_value，下方两行是段位标签。

其它相关设置沿用上游：`speedRate`（变速，DT=1.5 / HT=0.75）、
`enableAnalyzeLN`、`Card Body Content = Pattern` 等。

---

## 4. 算法参数位置

固定参数与映射曲线硬编码在 [`js/fixedDensity/index.js`](js/fixedDensity/index.js) 顶部
（`HALF_LIFE_MS`、`QUANTILE`、`EXTRA_*`、`SOFT_BETA`、`IRR_GAIN_*`、`MAP_KNOTS`、`MAP_COEF`、
`LN_A`、`LN_B`、`LN_MULT_*` 等）。要点：

```js
HALF_LIFE_MS = 306.625   QUANTILE = 0.91162709        // RC 衰减半衰期 / n 的分位数
EXTRA_RATE = 2.0524656   EXTRA_B = 2.3379955   EXTRA_LOG_BASE = 4.012384
SOFT_BETA = 0.706887                                   // 合并算子（softmax）
IRR_GAIN_LANE = 0.1      IRR_GAIN_GLOBAL = 0.06        // 局部熵增益
MAP_COEF[0..8] 映射曲线（c + 8 段斜率）；MAP_COEF[9]=E(chord)、MAP_COEF[10]=D(ln N)
LN_A = 5.44200375179058  LN_B = 0.0960408821250538     // ln_value = a·ln(1+b·k·n)
LN_MULT_REL = 0.1   LN_MULT_JACK = 0.4   LN_MULT_HOLD = 0.2   （γ_cross = 0）
LN_MULT_EXP = 6.0   LN_MULT_RHO_C = 0.33  LN_MULT_RHO_LIN = 0.1   LN_ABSORB_GAIN = 0.25
```

改这些常量即可改变插件输出；要与 `main` 分支保持一致，请同步修改
`main` 分支的 `fixed_density/params_final.json`。

---

## 5. 许可与致谢

- 上游插件：**ManiaMapAnalyser by Leo_Black** — <https://github.com/LeoBlackMT/osumania_map_analyser>
  （本分支的所有非 Fixed Density 代码版权归原作者，遵循上游许可）。
- Fixed Density v1 算法与 `js/fixedDensity/` 为本修改版新增。
- 谱面数据版权归各原作者所有，此处仅用于难度算法研究。
