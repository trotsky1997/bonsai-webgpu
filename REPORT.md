# W1A4 浏览器 v0.3.0：吞吐、加载与工具协议

设备：Apple M5 Pro、48 GiB、macOS ARM64；浏览器：Codex Chromium 155。原始 checkpoint 与 v0.2.0 相同。Rust/WASM + WebGPU 在浏览器中完成推理。

## 吞吐与首字延迟

三方轮换、预热后 7 轮、每轮生成 64 token。吞吐按 63 次 decode / decode 时间计算。清空 token 前缀，保留已分配 KV 容量；两个浏览器版本都先运行长提示词，将 KV 容量固定为 512 token。浏览器 prefill batch 8，原生参考逐 token prefill。TTFT 是请求到首个文本增量的墙钟时间，包括 prefill 的实现差异。

本轮 v0.2.0 对照来自 commit `d045b275a413f9ea592ba13bbb661f98a05a4cca`。同一轮测量的中位数，方括号为最小值至最大值。后台应用与 GPU 调度会影响绝对数值；不能把上一轮测量的绝对值拼接为本轮收益。

| 条件 | 浏览器 v0.2.0 | 浏览器 v0.3.0 | 原生 Rust MLX |
|---|---:|---:|---:|
| 短提示词 27 token / greedy，tok/s | 169.18 [159.88–175.29] | 218.08 [190.09–232.32] | 159.68 [156.57–166.95] |
| 短提示词 27 token / greedy，TTFT ms | 31.12 [29.45–37.53] | 26.94 [23.62–32.05] | 172.16 [161.67–179.86] |
| 长提示词 237 token / greedy，tok/s | 130.73 [124.08–133.01] | 157.65 [152.25–159.97] | 160.65 [157.23–162.76] |
| 长提示词 237 token / greedy，TTFT ms | 253.19 [246.11–261.56] | 224.67 [213.80–232.75] | 1397.11 [1392.74–1420.80] |
| 短提示词 / T=0.5, k=20, p=0.85，tok/s | 175.58 [164.37–178.24] | 223.61 [216.81–236.32] | 120.72 [120.23–129.70] |
| 短提示词 / T=0.5, k=20, p=0.85，TTFT ms | 34.20 [28.36–39.99] | 26.81 [24.20–33.30] | 170.48 [166.43–177.10] |

短提示词 greedy 提升 **28.9%**；长提示词 greedy 提升 **20.6%**；默认采样提升 **27.4%**。长提示词吞吐仍略低于当前原生 MLX 参考。默认采样还受益于浏览器 GPU top-k；原生参考会读回完整 logits 后在 CPU 排序。这是对当前两套实现的比较。

## 加载

新建 Worker；WASM/驱动 pipeline 缓存已预热。热加载使用已验证的 CacheStorage；缓存冷使用新建测试缓存并强制从本机 HTTP 下载。以下不代表互联网首访速度，也不包含 UI 下载或首次驱动编译。

| 加载条件 | v0.2.0 ms | v0.3.0 ms |
|---|---:|---:|
| 模型缓存命中，7 轮 | 184.66 [167.97–241.74] | 173.94 [169.43–195.22] |
| 模型缓存冷，3 轮，本机 HTTP | 579.72 [579.62–709.67] | 575.09 [565.76–589.06] |

本机热加载改善约 5.8%；本机缓存冷改善约 0.8%，网络和缓存波动较大。互联网 Pages 的测量单独记录，不能由本机数据推出。

加载路径改动：

- 原模型 FP16 scale 直接上传 GPU，统一转换为持久 FP32 buffer；去掉 WASM 中的逐 scale 转换和临时大数组。
- 生产 pipeline 集中编译和校验；六个诊断/参考 pipeline 在需要时才编译。模型 ready 等待 scale 转换完成。
- manifest 与权重一样按 SHA-256 寻址缓存，去掉每次热加载的 manifest 网络往返。导出站点时检查内嵌 manifest hash。
- 缓存命中预取深度 3，首次下载深度 6，读取与 GPU 初始化重叠。
- WASM、Worker 和应用壳使用版本化 URL；激活新应用壳保留已验证的大权重缓存。

## 计算内核

每个 W1 g128 scale 内的四个 A4 g32 点积手动展开。四个乘积用向量表达，但 FP32 累加仍依次执行 x、y、z、w，减少循环及地址计算并保留已有累加顺序。仍是完整 `[-7,7]` 的符号加幅值编码与三个 popcount，不丢激活值。W1 的对称 scale/bias 与共享激活绝对值总和沿用 v0.2.0。

四路独立累加后合并、attention 循环展开都曾改变整模型 logits；这些实验未进入发布版。逐次展开 FP16 scale 的版本吞吐退步，也未采用。没有引入新的近似量化或依赖。

## 质量与内存

v0.3.0 与 v0.2.0：18 段语料、1223 个评分 token 的 NLL、预测 token 逐项完全相同；3 组固定 greedy 生成完全相同。融合与参考内核的 40 token、6,066,760 个 logits 完全相同。40 组线性夹具、81,920 个输出误差为零；top-k 1/20/64、batch 1/8 的 40/320 token、KV 扩容、Unicode/特殊 token 分词边界检查通过。

浏览器与 MLX 的质量差异沿用 [v0.2.0 报告](REPORT-v020.md)：diagnostic NLL 2.99423777 vs MLX 3.00832716；holdout 3.34474746 vs MLX 3.37500826；本轮 3 组固定 greedy 仍只有 1 组与 MLX 逐 token 相同。这些是小规模诊断，浏览器实现仍为实验路径。

| 项目 | v0.2.0 | v0.3.0 |
|---|---:|---:|
| WASM 线性内存 | 50 MiB | 40.688 MiB |
| GPU 权重 payload | 256.759 MiB | 256.759 MiB |
| GPU KV，256 / 512 token 容量 | 56 / 112 MiB | 56 / 112 MiB |
| GPU scratch/readback payload | 2.173 MiB | 2.173 MiB |

WASM 线性内存减少 18.6%。GPU payload 不含驱动、pipeline、uniform、上传 scale 的临时 buffer、KV 扩容时同时存在的新旧 buffer；WASM 堆也不等于浏览器进程 RSS。不能把逻辑 buffer 与 MLX allocator 直接当作相同统计口径。

## 工具调用修复

原来的适配器只渲染文本消息，忽略请求的 `tools`，也未解析 `<tool_call>`。llama.cpp UI 虽然已提供本地 `get_datetime` 和 `get_info`，模型没有收到这些定义。

现在按 checkpoint 的 `chat_template.jinja` 注入 function schema、渲染 assistant tool_calls 和 tool_response，解析模型生成的 JSON，返回 OpenAI 流式 `delta.tool_calls` / `finish_reason: tool_calls`，由 UI 的 Agent 循环授权、执行、回传并继续生成。非流式响应也支持同样的 tool_calls。工具 JSON 无效、截断或调用未启用工具时返回错误，取消时不执行未完成调用。

已在实际 UI 中验证“今天是几号？”：模型主动调用 `get_datetime({})`，浏览器返回本地日期时间后，模型回答 10 月 10 日。中文流式、英文非流式两轮协议往返均通过，普通问候不调用工具。中文 greedy 样例日期和时分秒正确，但把 Asia/Shanghai 错解释成大阪；原始回答保留在 JSON 中，工具链跑通不等于模型对返回值的解释始终准确。70 个 XML 分块边界及字符流、错误/截断/取消夹具通过。没有通过关键字路由伪造调用，也没有把日期预先塞进 system prompt。

日期/时间和 runtime info 来自 UI 的浏览器本地工具。`/tools` 仍为空，表示没有服务器侧 shell/文件工具。JavaScript sandbox 默认关闭；工具授权沿用 UI 现有设置。支持自动工具选择和 `tool_choice: none`；强制选择暂未支持。小模型的调用准确率仍取决于问题与上下文，当前不是通用 Agent 能力评估。

## 复现与原始记录

- `benchmarks/optimization-v030.json`：三方轮换、固定生成与冷/热加载。
- `benchmarks/quality-v030.json`、`checks-v030.json`：质量及实际 GPU 检查。
- `benchmarks/tools-v030.json`、`tool-protocol.json`：真实模型工具往返及协议夹具。
- `benchmarks/v020/`：旧版源码、WASM 与 Worker；`native-reference/`：原生参考源码。

启动本机静态服务，打开 `/bench.html` 与 `/bench.html?variant=v020`，运行 `python3 benchmarks/optimize-v030.py`。原生参考的模型与 MLX 库目录通过脚本参数配置。协议夹具：`node benchmarks/tool-protocol.mjs`。Pages 只部署静态文件，本机测试 RPC 不部署。

发布 WASM SHA-256：`194efd3bab5785fd4c87a631030df56b89c856413210241db9c1422e3c645ea1`。
