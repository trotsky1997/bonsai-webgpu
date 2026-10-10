# Bonsai W1A4 · Rust/WASM + WebGPU

## v0.4.1 browser Agent

Native interactive cards, persistent virtual workspace tools and HTML artifacts, silent image OCR with result caching, exact n-gram lookup drafts, and user-key LangSearch search. Defaults: **128K context/KV limit, INT4 KV and sparse prefill enabled**. KV grows on demand. 128K YaRN and FlashPrefill-inspired sparse prefill remain experimental.

Token usage details includes speculative acceptance rate, average accepted prefix length, cumulative accepted tokens, and current/cumulative KV prefix hit rate. Counters follow the selected conversation branch and survive refresh; refreshing restores the fixed system prefix from IndexedDB, while conversation GPU KV starts empty.

The system prompt uses a minimal direct-answer mode and capability-specific Agent instructions. An explicit whole-clause request such as “No tools” or “不要调用工具” disables tools for that turn. See [prompt policy and measured examples](SYSTEM_PROMPT.md).

Fixed system-prefix KV is enabled by default: idle prewarming, two GPU snapshots and bounded IndexedDB persistence. Measured TTFT after losing the active KV: **3109.9 → 88.8 ms** median over seven pairs; one disk-restore sample was 173.4 ms. Outputs were identical. See [implementation, memory limits and measurements](SYSTEM_PREFIX_CACHE.md).

See [Agent/UI protocol and rebuild](INTERACTIVE_UI.md) and [KV layout, long context and sparse prefill](LONG_CONTEXT.md).


Bonsai-1.7B 的专用浏览器推理实现。28 层模型、W1A4 线性层、注意力和 KV 全部在浏览器执行。Rust 编译为 WASM，使用 wgpu 调用 WebGPU，计算内核为手写 WGSL。没有 ONNX、Python 推理进程或远程推理 API。

在线体验：[GitHub Pages](https://trotsky1997.github.io/bonsai-webgpu/) · [GitHub 源码](https://github.com/trotsky1997/bonsai-webgpu)

本机聊天：**http://127.0.0.1:18902/**

## 启动

```sh
node run.mjs
```

打开上面的地址。首次加载约 242 MB 的权重和约 3.2 MB 的二进制分词器，随后从浏览器缓存读取。网页首次成功加载并安装应用缓存后，可以断网重载并聊天。停止文件服务：

```sh
node stop.mjs
```

Node 只提供静态文件和本机测试结果收集；它没有推理实现。直接向静态服务的 `/v1/chat/completions` 发送 POST 会返回 405。聊天 UI 的请求在浏览器内被转换成 Worker 消息。

## 实现

- 专用 Qwen3/Bonsai 架构：hidden 2048、28 层、16 个 Q head、8 个 KV head、head dim 128、MLP 6144、151669 词表、YaRN factor 4（32K）；128K 实验模式使用 factor 16。
- W1 每 128 个权重一组。离线检查全部 197 个矩阵的 `bias = -scale/2`，保留原 FP16 scale、移除重复 bias，浏览器上传后由 GPU 将 scale 转为 FP32。
- A4 每 32 个激活一组，范围 `[-7,7]`，保存三个幅值位面和一个符号位面。每个 32 元素点积使用三个 `countOneBits`。
- QKV、gate/up 合并调度；残差加法与 RMSNorm、SiLU 与打包、注意力与打包分别融合。每个 decode 的主图从 508 次 dispatch 减少到 312 次（支持 10 个 storage binding 的适配器）。
- INT4 KV 按 token、g32 分组，保存 scale 和打包码；K 按组/时间排列，V 按时间/组排列。扩容用 GPU 位复制内核迁移。
- 所有张量和 KV 常驻 GPU。Q/K/V 共享激活打包；gate/up 共享激活打包。每个 token/chunk 集中提交计算，每个 decode 只读回 GPU 分区 top-k 候选。
- 默认 prefill batch 8，decode batch 1。批量 prefill 是同层批量执行的专用 W1A4 投影，并非通用浮点 GEMM。
- 默认上下文上限 128K（实验 YaRN），可在设置中选择 32K 原配置或 4K。INT4 KV 从 256 token 起按倍数扩容；短 attention 使用工作组内存，长 attention 分块计算并归约。GPU 缓冲区上限不足时提供缩短上下文重试。
- 紧凑二进制保存同一词表、merge 排序和所有预处理配置，Rust tokenizers 执行完整的原模型分词；原有 llama.cpp 聊天 UI 显示流式输出。
- 利用原模型 FP16 scale 与小整数乘积在 FP32 内可精确表示的性质，合并对称 scale/bias 项，保留参考累加顺序。
- 权重从 JS 缓存字节直接复制到 WebGPU mapping，避免大权重往返 WASM 堆；缓存命中预取 3 个、首次下载预取 6 个分层文件，GPU pipeline 和候选 bind group 复用。
- GPU 的 A4 `/7` 使用整数尾数计算保持正确 FP32 舍入；显式 FMA 控制乘积舍入和累加边界，避免浏览器隐式融合改变已有线性内核的顺序。
- SiLU 使用与 MLX 相同的稳定 sigmoid 分支形式；浮点归一化、exp、RoPE、注意力在不同后端仍可能产生差异。

## 已测结果与质量状态

v0.3.0 与 v0.2.0 同场七轮轮换：短提示词 greedy **169 → 218 tok/s（+28.9%）**，长提示词 **131 → 158 tok/s（+20.6%）**，默认采样 **176 → 224 tok/s（+27.4%）**。本机热加载中位数 **185 → 174 ms**；WASM 线性内存 **50 → 40.688 MiB**。优化前后 1223 个评分 token 的 NLL 和预测结果完全一致。

详见 [REPORT.md](REPORT.md) 和 `benchmarks/` 的原始 JSON。设备为 Apple M5 Pro、48 GiB 内存；浏览器为 Codex 内嵌 Chromium 155。

本项目代码为 MIT；模型及分词器为 Apache 2.0；llama.cpp UI 为 MIT，详见对应许可文件。

v0.3 FP32 KV 的整数打包和线性层测试通过；两套语料的平均 NLL 均未退化。3 组固定 greedy 生成中 1 组逐 token 相同，另外 2 组不同。v0.4 INT4 KV 短诊断 NLL 为 5.031，对照 FP32 为 4.904；详见 [v0.4 验收报告](RELEASE_0.4.md)。**浏览器后端保留为实验路径**，原生 MLX 页面仍位于 http://127.0.0.1:18901/。

默认采样为 temperature 0.5 / top_k 20 / top_p 0.85。支持 greedy、流式取消、stop strings、对话前缀复用及确定性 seed。当前 top_k 范围 1–64；repeat_penalty 要求 1，presence_penalty/frequency_penalty 要求 0；支持文本及 Qwen function calling。UI 调整到不支持的设置会返回明确错误。

WASM SIMD CPU 回退和其他模型架构尚未实现。当前发布版要求浏览器支持 WebGPU；localhost 为可用的安全上下文，远程静态托管需要 HTTPS。服务器启用了 COOP/COEP。

## 工具调用

复用 llama.cpp UI 的浏览器工具：`get_datetime` 读取本地日期时间，`get_info` 返回运行环境信息。模型看到 function schema 后生成 `<tool_call>`；适配器转换为标准 `tool_calls`，UI 授权并执行，结果回传模型继续回答。首次调用可能出现 **Allow once** 提示；工具菜单位于输入框旁的 `+ → Tools`。

所有推理和日期读取都在浏览器内完成；没有服务器侧文件或 shell 工具。JavaScript sandbox 默认关闭。新增浏览器虚拟工作区工具、原生交互卡片和 LangSearch；实际验收及模型失败案例见 [v0.4 验收报告](RELEASE_0.4.md)。工具调用质量受 1.7B 模型能力与上下文影响。

## 重建

已包含可直接部署的 WASM；导出静态站点运行 `node build-site.mjs`，输出 `work/site/`。GitHub Actions 在推送 main 时自动发布 Pages。站点支持项目子路径，推理无需跨源隔离或 SharedArrayBuffer。

重建 WASM 需要 Rust/rustup、`wasm32-unknown-unknown` target、wasm-pack：

```sh
rustup target add wasm32-unknown-unknown
./build.sh
```

源码：`src/lib.rs`、`src/*.wgsl`。`build.sh` 在 macOS 上为 Rust 自带链接器设置其 LLVM 库路径，避免本机 Rust 安装的动态库搜索路径问题。

重新转换权重只在离线准备阶段需要 Python + numpy：

```sh
python3 convert.py /path/to/Bonsai-1.7B-mlx-1bit ./model
```

`model/manifest.json` 记录每个分层文件的 SHA-256 和原始 checkpoint 的 SHA-256；首次下载时校验每个权重和分词器文件；校验成功后使用 SHA-256 寻址的 CacheStorage，缓存命中直接复用已验证内容。

## 复现对比

先打开 http://127.0.0.1:18902/bench.html，再运行：

```sh
python3 benchmarks/compare.py \
  --model-dir ../../work/model \
  --mlx-lib ../bonsai-mlx-chat/lib
```

`native-reference/` 提供原生参考源码，按其中说明重建 ARM64 二进制到 `benchmarks/bonsai-mlx-benchmark`，并提供 MLX SDK 动态库。参数允许指定其他模型目录、基线二进制、动态库目录、服务地址和轮数。脚本在同一提示词、清空 token 前缀缓存、相同生成长度下轮换两个后端；原生和浏览器的 prefill 实现不同，报告明确列出 batch 大小。

## 来源与许可

- 模型：[prism-ml/Bonsai-1.7B-mlx-1bit](https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit)，Apache 2.0；见 `MODEL_LICENSE`。权重只是重新排列并移除完全冗余的 bias，没有重新训练。
- UI：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/3d65c90d04d337e88f2b1f7f0061f40a5324e662/tools/ui)，固定 commit `3d65c90d04d337e88f2b1f7f0061f40a5324e662`，MIT；见 `LLAMA_CPP_LICENSE`。静态 UI 入口新增浏览器 Worker 适配脚本，服务使用本包的应用缓存。
- wgpu、wasm-bindgen、tokenizers 等依赖按各自许可使用，确切版本见 Cargo.lock。原生 MLX 基线的许可和动态库位于相邻的 `bonsai-mlx-chat` 包。
