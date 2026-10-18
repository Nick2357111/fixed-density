# qqbot · Rework 谱面分析机器人

在 QQ 里发送 `/rework` 指令，即可得到一张 **[ManiaMapAnalyser by Leo_Black]（tosu 插件）风格的分析卡片图片**：
星级徽章（Fixed Density RC / LN）、难度标签、LN% / Keys、五条 pattern 进度条、HB/LN/RC/Mix 胶囊等，
与插件实时窗口的显示完全一致（同一套前端代码 + 同一套设置）。

---

## 1. 功能

| 用法 | 说明 |
| --- | --- |
| `/rework <谱面ID>` | 分析单张谱面，返回一张卡片图片 |
| `/rework bid<谱面ID>` / `b<ID>` | 同上（显式指定谱面 ID；`b`、`bid` 与数字之间可省略空格，写 `bid 123` 亦可） |
| `/rework sid<谱包ID>` / `s<ID>` | 分析整个谱包，**以聊天记录（合并转发）形式发送多张图片** |
| `/rework <osu! 链接>` | 支持 `osu.ppy.sh/beatmapsets/...`、`/beatmaps/...`、`/b/...`、`/s/...` |
| 回复一条谱面文件并发送 `/rework` | 支持 `.osu` / `.osz` / `.mc` / `.mcz`，以及**任意 `.zip` 压缩包**（可嵌套，自动扫描包内所有目录）；多张谱面会以聊天记录形式发送 |
| 回复机器人图片发送 `/report <rc> [ln]` | 提交实际体感难度；谱包（合并转发）用 `/report <编号> <rc> [ln]`，编号从 1 开始 |

**容器读取**：`.osz` / `.mcz` / `.zip` 统一按 zip 处理，递归遍历**包内所有目录**，收集其中的
`.osu`、`.mc`，并对嵌套的 `.osz` / `.mcz` / `.zip` 继续展开（最多 4 层，含条目数与解压体积上限）。
同一包内重复的谱面会按内容去重。

**Malody 支持**：`.mc` / `.mcz` 会先在本地按 **模式 + 轨道数** 校验，再转换为一份临时的 `.osu`
（**laz 形式：物件时间戳保留浮点毫秒**）后交给插件分析。

**容错行为**：读取谱面时若 **轨道数不是 4** 或 **不是 osu!mania 谱面**，
该张谱面的卡片会被**错误提示图片替换**（例如「该谱面为 7K，本工具仅支持 4K mania 谱面。」），
其余谱面照常输出。Malody 谱面同理（非 Key 模式 / 非 4K 会给出对应提示）；
单张渲染失败也只会替换那一张，不会中断整包。

---

## 2. 目录结构

```
qqbot/
├─ run.py                  # 入口（启动机器人 / 本地渲染自测）
├─ refresh_db.py           # 数据库刷新入口（重新渲染全部已缓存谱面）
├─ config.json             # 你的配置（首次运行自动由 config.example.json 复制生成）
├─ config.example.json     # 配置模板
├─ requirements.txt
├─ temp/                   # 渲染产物（output.keep_files 为 true 时落地）
├─ data/                   # 谱面库
│  ├─ charts/              # <哈希>.osu / <哈希>.png / <哈希>.bg / <哈希>.json
│  ├─ feedback.json        # 用户反馈
│  └─ sent_messages.json   # 已发送消息 → 谱面哈希索引
└─ reworkbot/
   ├─ config.py            # 配置加载
   ├─ beatmap.py           # .osu 解析与 4K mania 校验
   ├─ archive.py           # 统一压缩包扫描（.osz/.mcz/.zip，含嵌套）
   ├─ malody.py            # .mc 解析与转换为 .osu（laz 浮点时间戳）
   ├─ cache.py             # 哈希谱面库（命中直接复用图片）
   ├─ feedback.py          # 用户反馈 + 发送索引
   ├─ osu_source.py        # 镜像站下载 / 链接解析 / 本地读取
   ├─ onebot.py            # OneBot v11 客户端（收发消息、合并转发）
   ├─ renderer.py          # 本地静态服务 + 无头浏览器截图（核心）
   ├─ errorcard.py         # 错误提示图片
   └─ service.py           # 指令解析与业务编排
```

---

## 3. 环境准备

需要 **Python ≥ 3.10**（本机为 3.13）与一个可用的 Chromium 内核浏览器。

```powershell
cd "g:\modding list\4kcalc\qqbot"
python -m pip install -r requirements.txt
```

浏览器二选一：

- **推荐**：使用系统已安装的 Edge —— 无需额外下载。`config.json` 中 `render.browser_channel` 保持 `"msedge"` 即可。
- 或安装 Playwright 自带 Chromium，并把它设为空字符串：

```powershell
python -m playwright install chromium
# 然后把 config.json 的 render.browser_channel 改为 ""
```

---

## 4. 绑定 QQ 机器人（步骤）

本项目通过 **OneBot v11 协议**与 QQ 端通信，机器人作为**正向 WebSocket 客户端**连接。
下面以 Windows 上最常用的 **NapCatQQ** 为例（Lagrange.Core / go-cqhttp 等同理，只需拿到 WS 地址与 Token）。

### 4.1 准备一个 QQ 账号

建议使用一个**小号**作为机器人账号（新注册的账号建议先挂机养号几天，降低风控概率）。

### 4.2 安装并登录 NapCatQQ

**Windows 推荐「一键版」（自带独立 QQ，不干扰你正在用的 QQ）**

1. 到 [Releases](https://github.com/NapNeko/NapCatQQ/releases/latest) 下载 `NapCat.Shell.Windows.OneKey.zip` 并解压。
2. 运行 `NapCatInstaller.exe`：它会自动下载 NapCat 当前适配的 QQ 版本（如 NapCat v4.18.38 对应 QQ 9.9.33）并生成 `NapCat.XXXX.Shell` 目录。
   > 注意版本必须匹配：NapCat 各版本只适配特定 QQ 版本（例如本机原有的 QQ 9.9.19 就不适用 v4.18.38）。
   > 一键版自带适配的 QQ，因此不必担心；若坚持用 `NapCat.Shell.zip` 复用已装 QQ，需自行确认版本对应。
3. 进入生成的 `NapCat.XXXX.Shell` 目录，双击 `napcat.bat` 启动，按控制台提示**用机器人 QQ 扫码登录**。登录成功即完成「绑定 QQ 账号」。
4. 控制台会打印 WebUI 地址（默认 <http://127.0.0.1:6099/webui>）与 Token，浏览器打开后进入
   **网络配置 → 添加 → WebSocket 服务器（正向 WS）**，填写：
   - 主机：`127.0.0.1`
   - 端口：`3001`（可自定义，需与 `config.json` 的 `onebot.ws_url` 一致）
   - Token：填成与 `qqbot/config.json` 里 `onebot.access_token` **完全相同的字符串**
   - 消息格式：`array`（必须，本项目按消息段数组解析）
5. 保存并启用，此时 NapCat 已在 `ws://127.0.0.1:3001` 监听。

> 若 NapCat 与机器人不在同一台机器，请把 `127.0.0.1` 换成实际可访问的内网 IP，并在防火墙放行该端口。

### 4.3 填写配置

编辑 `qqbot/config.json`：

```jsonc
"onebot": {
  "ws_url": "ws://127.0.0.1:3001",   // ← 4.2 里的正向 WS 地址
  "access_token": "你设置的Token",     // ← 4.2 里的 Token
  "command_prefix": "/rework"
}
```

### 4.4 启动

```powershell
cd "g:\modding list\4kcalc\qqbot"
python run.py
```

看到 `已连接到 OneBot。` 与 `机器人已就绪，等待指令。` 即绑定成功。
在 QQ 里 **私聊机器人** 或 **在群里 @机器人** 发送 `/rework 129891` 试一下。

**排查问题时打开更详细的日志**：

```powershell
python run.py --log-level DEBUG              # 控制台输出更详细
python run.py --log-file .\logs\qqbot.log    # 同时写入滚动日志文件（5MB × 3）
```

默认就以 `INFO` 逐条打印完整链路，渲染进度一目了然：

```
[#1] 收到指令：'/rework 129891'（来自用户 20002）
[#1] ID=129891 未指定类型，先按谱面 ID 探测…
    下载 .osu：https://osu.direct/api/osu/129891
    .osu 下载完成：49.7 KB
[#1] [1/1] xi - FREEDOM DiVE - FOUR DIMENSIONS // Nakagawa-Kanon
[#1] [1/1] 校验通过：4K mania，背景图 无
    页面已加载（0.4s），等待插件分析…
    等待分析… 页面状态：Loading beatmap file (initial load)...
    等待分析… 页面状态：xi - FREEDOM DiVE - FOUR DIMENSIONS // ...
    分析完成：星等=12.63（用时 2.6s）
    等待动画结束（1500 ms）…
    卡片就绪：谱面=...  星等=12.63
    正在截图：卡片 475×575（倍率 2）…
    渲染完成：输出 322.3 KB，用时 4.7s
[#1] [1/1] 完成，用时 5.5s（累计 5.6s）
[#1] 发送单张消息（分析卡片）
[#1] 处理完成，总用时 5.6s
```

若日志停在「等待分析…」，说明卡在插件分析阶段，可据此判断是页面报错还是渲染超时；
`render.settle_ms` 调大可解决动画未结束就截图的问题。

> 群聊使用前，请确认 `config.access.allow_group` 为 `true`；若只想在指定群生效，
> 把群号填入 `access.group_allowlist`（留空表示不限制）。

---

## 5. 配置说明（config.json）

| 字段 | 说明 |
| --- | --- |
| `onebot.ws_url` | OneBot 正向 WebSocket 地址 |
| `onebot.access_token` | 与 NapCat 中一致的 Token，留空表示不鉴权 |
| `onebot.command_prefix` | 指令前缀，默认 `/rework` |
| `render.plugin_dir` | 插件目录，默认 `../ManiaMapAnalyser.by.Leo_Black` |
| `render.host` / `render.port` | 本地静态服务地址，默认 `127.0.0.1:24750`。**不要与 tosu 的 24050 相同** |
| `render.browser_channel` | `"msedge"` 用系统 Edge；`""` 用 Playwright 自带 Chromium |
| `render.device_scale_factor` | 截图倍率，2 = 二倍图（默认） |
| `render.settle_ms` | 等动画结束再截图的时间，画面异常可调大 |
| `osu.beatmap_url` / `osu.beatmapset_url` | 谱面镜像下载地址模板，`{id}` 会被替换 |
| `osu.max_beatmaps_per_pack` | 单次最多处理多少张谱面 |
| `osu.include_background` | 是否使用谱面包内的背景图（卡片封面垫图 + 取色） |
| `output.keep_files` | 是否把 PNG 落地到 `temp/` 便于排查 |
| `output.send_as_forward` | 多张谱面时是否用合并转发 |
| `storage.data_dir` | 谱面库目录，默认 `data`（相对配置文件） |
| `storage.enable_cache` | 是否启用哈希缓存；关掉则每次都重新渲染 |
| `storage.max_sent_records` | 发送索引保留条数（供 `/report` 定位，默认 5000） |
| `access.*` | 群聊/私聊开关、群白名单、用户黑名单 |

**谱面镜像**：默认使用 `osu.direct`。若该镜像不可用，可换成其他镜像，例如

```jsonc
"beatmap_url":    "https://catboy.best/osu/{id}",
"beatmapset_url": "https://catboy.best/d/{id}"
```

> `/rework <ID>` 需要机器人所在机器能访问外网镜像。回复文件的方式则完全离线可用。

---

## 6. 工作原理（为什么能离线复现插件卡片）

`ManiaMapAnalyser` 是 tosu 的前端插件：页面加载后会直接向
`http://<socketHost>/files/beatmap/file` 取 .osu 原文并分析、渲染，
tosu 的 WebSocket 断连只会在后台重试。因此本项目：

1. 用本地 HTTP 服务把插件目录当作站点根托管；
2. 拦截 `/files/beatmap/file` 返回目标谱面文本、`/files/beatmap/background` 返回谱面背景图；
3. 拦截 `/settings.json`，**在内存中**注入与截图一致的插件设置（**不会修改磁盘上的 settings.json**）；
4. 用无头浏览器打开 `index.html`，等卡片渲染完成后对 `.main-card` 截图。

注入的设置项（对应你给出的插件设置截图）：

| 插件设置 | 值 |
| --- | --- |
| Card Body Content | `Pattern` |
| Top-left Capsule Text | `Fixed Density v1` |
| Top-right Content | `Difficulty` |
| Map Tag Capsule / osu!Lazer 主题 / osu! 字体 / 浮动三角 / 封面背景 | 开 |
| Custom Background Color | `#000000`（关闭纯色，改由封面取色） |
| Rainbow Bars / Reverse Card Extension / Extended Estimation Range | 关 |
| Metadata Marquee / Numeric Difficulty / LN Star Rating / Show 6K Constant Rating | 开 |
| Card Visibility / Card Opacity / Content Background Blur / Card Radius | `Always` / `95%` / `4px` / `Medium` |
| Pause Detection / Vibro Detection | 开 |
| SV Detection | 关 |
| Improve Sunny LN Estimation / Analyze LN Parts | 开 |
| Pause Detection Threshold | `500` |
| Estimator Algorithm / Global Etterna Version / Companella Etterna Version | `Mixed` / `0.72.3` / `0.74.0` |

另有两项离屏渲染专用改动：`Enable Update Check` 与 `Anonymous Usage Statistics` 关闭，避免联网阻塞。

想改显示效果，直接改 [renderer.py](file:///g:/modding%20list/4kcalc/qqbot/reworkbot/renderer.py) 里的 `PRESET_OVERRIDES` 即可。

### 6.1 Malody（.mc / .mcz）转换

插件只认 `.osu`，所以 Malody 谱面先在本地转换成临时 `.osu` 再交给插件。转换规则
（依据真实 `.mcz` 逆向 + 与 rmstZ / rconv 参考实现交叉验证）：

| Malody | osu! |
| --- | --- |
| `meta.mode`（0 = Key） / `meta.mode_ext.column` | `Mode: 3` / `CircleSize` |
| `beat: [小节, snap索引, snap大小]` | 拍位 `a + b/c`，再按 `time[].bpm` 逐段积分为毫秒 |
| `note[].endbeat` | `type 128`（长按）+ `endTime` |
| `effect[].scroll > 0` | 绿线（`uninherited=0`，`beatLength = -100/scroll`） |
| 无 `column` 且 `type==1` 的条目 | BGM 事件，其 `offset` → `AudioLeadIn` |
| `column` | `x = round((column*2+1)/(key*2) * 512)` |

几点说明：

- **时间戳为 lazer 形式**：物件与 TimingPoint 的时间都保留浮点毫秒、不取整，
  避免转换引入系统性偏差。
- **OD 取 8**：Malody 没有 OD 概念，转换时固定写 `OverallDifficulty:8`（osu! 默认值）。
- 转换结果会作为谱面文本存进 `data/charts/<哈希>.osu`，可直接打开核对。
- `.mcz` 内的背景图会用于卡片封面垫图与取色。
- 旧版 Malody 的数组式音符格式暂不支持，会给出明确提示而不是产出错误结果。

### 6.2 谱面库（哈希缓存）

每次查询都会把**最终用于分析的那份 `.osu` 文本**做 SHA-256 作为键，落盘到 `data/charts/`：

| 文件 | 内容 |
| --- | --- |
| `<哈希>.osu` | 谱面文本（Malody 则是转换后的结果） |
| `<哈希>.png` | 生成的分析卡片 |
| `<哈希>.bg` | 可选背景图（刷新时用来还原封面） |
| `<哈希>.json` | 名称、来源、创建/更新时间 |

再次查询到同一谱面时（包括来自不同谱包、改了文件名的同一份内容）**直接复用已有图片，不再渲染**，
日志会显示 `命中历史结果（<哈希前12位>…）`。

想强制重算：删除 `data/charts/` 下对应文件，或直接跑 `refresh_db.py`。

### 6.3 用户反馈（/report）

- 回复机器人发出的图片，发送 `/report <rc> [ln]`。
- 谱包发的是**合并转发**，需要 `/report <编号> <rc> [ln]`，编号从 1 开始（表头不计入）。
- 只填一个数值时按 **RC** 处理（LN 留空）。
- 同一用户对同一谱面只保留一条记录，重复提交会覆盖并回执「已更新」。
- 数据存于 `data/feedback.json`，键为 `<用户QQ>|<谱面哈希>`；错误提示图片没有谱面哈希，无法反馈。

---

## 7. 本地自测（不启动机器人）

```powershell
# 单张谱面
python run.py --render "..\benchmark\ln\Xyris - Eviternity [LN Eviternity].osu" --out .\temp\out.png

# 谱包 / 任意 zip（自动递归嵌套容器）
python run.py --render .\temp\pack.osz --out .\temp

# 非 4K / 非 mania 谱面 -> 输出错误提示图片，进程退出码 2
python run.py --render .\sevenK.osu --out .\temp
```

**刷新整个谱面库**（改过插件设置或升级插件后，统一重算所有缓存图片）：

```powershell
python refresh_db.py --dry-run     # 先看看有哪些条目
python refresh_db.py               # 重新渲染并覆盖全部 <哈希>.png
```

刷新会逐条打印 `[i/N] 谱面名 已刷新（大小，耗时）`，失败的条目保留原图并计入失败数。

---

## 8. 常见问题

- **`无法启动无头浏览器`**：执行 `python -m playwright install chromium`，或把 `render.browser_channel` 设为 `"msedge"`。
- **端口被占用 / 启动报错**：修改 `render.port`（避开 tosu 的 24050）。
- **截图里是别的谱面**：确认 `render.port` 与 `onebot` 无关；本项目已把插件谱面来源指向本地服务，
  若你手工改过 `PRESET_OVERRIDES` 请勿删除 `wsEndpoint` 的注入。
- **`/rework <ID>` 报「镜像站返回的不是谱面文件」**：镜像限流或 ID 不存在，换镜像或改用回复文件。
- **谱包很大时很久**：谱面逐张串行渲染，可用 `osu.max_beatmaps_per_pack` 限制，或改用谱面 ID。
- **`.mcz` / `.osz` 里其实放的是 `.osu` 或嵌套压缩包**：现在统一按 zip 处理，会自动递归扫描所有目录。
- **改了插件设置后图片没变**：这是哈希缓存的正常行为，跑一次 `python refresh_db.py` 重算即可。
- **`/report` 提示找不到对应谱面**：发送索引只保留最近 `storage.max_sent_records` 条，且重启前发送的消息在索引清空后无法定位。
