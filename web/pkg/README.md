# Bonsai W1A4 · Rust/WASM + WebGPU

Bonsai-1.7B 的专用浏览器推理实现。28 层模型、W1A4 线性层、注意力和 KV 全部在浏览器执行。Rust 编译为 WASM，使用 wgpu 调用 WebGPU，计算内核为手写 WGSL。没有 ONNX、Python 推理进程或远程推理 API。

聊天页面：**http://127.0.0.1:18902/**

## 启动

```sh
node run.mjs
```

打开上面的地址。首次加载约 242 MB 的权重和约 11 MB 的分词器，随后从浏览器缓存读取。网页首次成功加载并安装应用缓存后，可以断网重载并聊天。停止文件服务：

```sh
node stop.mjs
```

Node 只提供静态文件和本机测试结果收集；它没有推理实现。直接向静态服务的 `/v1/chat/completions` 发送 POST 会返回 405。聊天 UI 的请求在浏览器内被转换成 Worker 消息。

## 实现

- 专用 Qwen3/Bonsai 架构：hidden 2048、28 层、16 个 Q head、8 个 KV head、head dim 128、MLP 6144、151669 词表、YaRN factor 4。
- W1 每 128 个权重一组。离线检查全部 197 个矩阵的 `bias = -scale/2`，保留原 FP16 scale、移除重复 bias，浏览器上传时把 scale 转为 FP32。
- A4 每 32 个激活一组，范围 `[-7,7]`，保存三个幅值位面和一个符号位面。每个 32 元素点积使用三个 `countOneBits`。
- 所有张量和 KV 常驻 GPU。Q/K/V 共享激活打包；gate/up 共享激活打包。每个 token/chunk 集中提交计算，每个 decode 只读回 GPU 分区 top-k 候选。
- 默认 prefill batch 8，decode batch 1。批量 prefill 是同层批量执行的专用 W1A4 投影，并非通用浮点 GEMM。
- 默认上下文上限 2048。FP32 KV 从 256 token 起按 256 扩容，保留历史前缀并复用缓存容量。
- Rust tokenizers 执行完整的原模型分词；原有 llama.cpp 聊天 UI 显示流式输出。
- GPU 的 A4 `/7` 使用整数尾数计算保持正确 FP32 舍入；显式 FMA 控制乘积舍入和累加边界，避免浏览器隐式融合改变已有线性内核的顺序。
- SiLU 使用与 MLX 相同的稳定 sigmoid 分支形式；浮点归一化、exp、RoPE、注意力在不同后端仍可能产生差异。

## 已测结果与质量状态

详见 [REPORT.md](REPORT.md) 和 `benchmarks/` 的原始 JSON。设备为 Apple M5 Pro、48 GiB 内存；浏览器为 Codex 内嵌 Chromium 155。

本实现的整数打包和线性层测试通过；两套语料的平均 NLL 均未退化。3 组固定 greedy 生成中 1 组逐 token 相同，另外 2 组不同。**浏览器后端保留为实验路径**，原生 MLX 页面仍位于 http://127.0.0.1:18901/。

默认采样为 temperature 0.5 / top_k 20 / top_p 0.85。支持 greedy、流式取消、stop strings、对话前缀复用及确定性 seed。当前 top_k 范围 1–64；repeat_penalty 要求 1，presence_penalty/frequency_penalty 要求 0；只支持文本。UI 调整到不支持的设置会返回明确错误。

WASM SIMD CPU 回退和其他模型架构尚未实现。当前发布版要求浏览器支持 WebGPU；localhost 为可用的安全上下文，远程静态托管需要 HTTPS。服务器启用了 COOP/COEP。

## 重建

需要 Rust/rustup、`wasm32-unknown-unknown` target、wasm-pack：

```sh
rustup target add wasm32-unknown-unknown
./build.sh
```

源码：`src/lib.rs`、`src/*.wgsl`。`build.sh` 在 macOS 上为 Rust 自带链接器设置其 LLVM 库路径，避免本机 Rust 安装的动态库搜索路径问题。

重新转换权重只在离线准备阶段需要 Python + numpy：

```sh
python3 convert.py /path/to/Bonsai-1.7B-mlx-1bit ./model
```

`model/manifest.json` 记录每个分层文件的 SHA-256 和原始 checkpoint 的 SHA-256；浏览器加载时校验每个权重文件。

## 复现对比

先打开 http://127.0.0.1:18902/bench.html，再运行：

```sh
python3 benchmarks/compare.py \
  --model-dir ../../work/model \
  --mlx-lib ../bonsai-mlx-chat/lib
```

附带的 `benchmarks/bonsai-mlx-benchmark` 是 ARM64 原生基线，复用已交付 MLX 包的动态库。参数允许指定其他模型目录、基线二进制、动态库目录、服务地址和轮数。脚本在同一提示词、清空 token 前缀缓存、相同生成长度下轮换两个后端；原生和浏览器的 prefill 实现不同，报告明确列出 batch 大小。

## 来源与许可

- 模型：[prism-ml/Bonsai-1.7B-mlx-1bit](https://huggingface.co/prism-ml/Bonsai-1.7B-mlx-1bit)，Apache 2.0；见 `MODEL_LICENSE`。权重只是重新排列并移除完全冗余的 bias，没有重新训练。
- UI：[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/3d65c90d04d337e88f2b1f7f0061f40a5324e662/tools/ui)，固定 commit `3d65c90d04d337e88f2b1f7f0061f40a5324e662`，MIT；见 `LLAMA_CPP_LICENSE`。静态 UI 入口新增浏览器 Worker 适配脚本，服务使用本包的应用缓存。
- wgpu、wasm-bindgen、tokenizers 等依赖按各自许可使用，确切版本见 Cargo.lock。原生 MLX 基线的许可和动态库位于相邻的 `bonsai-mlx-chat` 包。
