# W1A4 浏览器优化与 MLX 对比

设备：Apple M5 Pro、48 GiB、macOS ARM64；浏览器：Codex 内嵌 Chromium 155。全部计算由 Rust/WASM + WebGPU 在浏览器完成。没有 ONNX、Python 推理进程或远程推理 API。

## 吞吐与首字延迟

七轮预热后的三方轮换，清空 token 前缀缓存，每次生成 64 token；吞吐按 63 次 decode / decode 时间计算。相同 checkpoint，W1A4 g32、FP32 KV、上下文上限 2048。浏览器 prefill batch 8；当前 Rust MLX 参考逐 token prefill。TTFT 是请求到第一个文本增量的墙钟时间；因此首字优势也包含 prefill 调度差异。

数值是中位数，方括号为最小值至最大值。

| 条件 | 优化前浏览器 | 优化后浏览器 | 原生 Rust MLX |
|---|---:|---:|---:|
| 短提示词 27 token / greedy，tok/s | 138.14 [134.28–142.28] | 181.44 [179.02–191.73] | 175.04 [168.89–177.80] |
| 短提示词 27 token / greedy，TTFT ms | 41.19 [32.49–41.94] | 31.64 [27.51–41.17] | 162.28 [150.43–168.71] |
| 长提示词 237 token / greedy，tok/s | 113.26 [106.51–120.23] | 139.67 [123.14–145.52] | 169.32 [144.56–174.20] |
| 长提示词 237 token / greedy，TTFT ms | 289.24 [264.92–309.31] | 247.10 [239.62–363.69] | 1340.90 [1305.43–1444.80] |
| 短提示词 / T=0.5, top-k=20, top-p=0.85，tok/s | 133.47 [130.55–137.20] | 181.02 [176.65–185.26] | 127.03 [124.18–128.81] |
| 短提示词 / T=0.5, top-k=20, top-p=0.85，TTFT ms | 38.70 [34.54–43.56] | 29.53 [27.50–37.72] | 164.71 [161.14–166.62] |

短提示词 greedy 超过当前原生参考；长提示词 greedy 仍低于原生 MLX。默认采样的优势还来自 GPU top-k，原生参考会读回完整 logits 后在 CPU 排序。这些结论针对当前实现，不代表所有官方 MLX 推理器。

## 加载

新建 Worker，WASM/驱动 pipeline 缓存已预热。热加载使用 CacheStorage；冷加载使用新的 CacheStorage 名称并强制从本机静态 HTTP 服务下载。冷加载结果不代表互联网下载速度，也不包含首次浏览器启动、UI 下载或第一次驱动编译。

| 加载条件 | 优化前 ms | 优化后 ms |
|---|---:|---:|
| 模型缓存命中，7 轮 | 411.25 [400.45–549.35] | 178.69 [157.04–271.22] |
| 模型缓存冷，3 轮，本机 HTTP | 1142.72 [1093.18–1429.85] | 634.35 [590.35–637.31] |

权重约 242.4 MB；分词器从约 11 MB JSON 变为 3.19 MB 二进制。首次下载校验 SHA-256，成功后使用哈希寻址的缓存；热加载复用已验证字节。三个文件有界预取，读取与初始化重叠。权重从 JS 字节直接复制到 WebGPU mapping，避免大权重往返 WASM 堆。

## 优化内容

- 主图从 508 次 dispatch 减为 312 次：合并 QKV 和 gate/up 调度；融合残差与 RMSNorm、SiLU 与 A4 打包、注意力与 A4 打包。QKV 融合要求适配器支持 10 个 storage buffer binding；不满足时保留独立 Q/K/V 调度。
- K 缓存按 `[head, dim, position]` 排列；V 保留按时间排列。扩容通过 GPU 位复制迁移，保持数据精度。
- 激活绝对值总和只在打包时计算一次。原始 FP16 weight scale 乘小整数在 FP32 内精确表示，因此对称 scale/bias 可以合并，同时保留原有 FP32 组间累加顺序。这个保证依赖原模型的 FP16 scale 与整数范围，不是对任意 FP32 scale 的保证。
- 重用 pipeline handle、shape uniform 和候选 bind group。多 token prefill 共享整层执行。
- top-k>1 的每个线程缓存自己的 19 个候选，仅获胜线程更新局部最大值，去掉重复扫描已选 token 的循环；greedy 使用单独的简单路径。
- 保留 A4 `/7` 的正确 FP32 舍入、ties-away 量化，以及限制隐式乘加融合的运算边界。

## 质量

优化前后 18 段语料、1223 个评分 token：NLL 逐项相同，预测 token 全部相同。三组固定 greedy 生成也全部与优化前相同。

| 语料 | 评分 token | 原生 MLX NLL | 浏览器 NLL | 差值 |
|---|---:|---:|---:|---:|
| diagnostic | 390 | 3.00832716 | 2.99423777 | -0.01408938 |
| holdout | 833 | 3.37500826 | 3.34474746 | -0.03026080 |

这是小规模本地诊断，不是 MMLU 或通用能力结论。浏览器和 MLX 整模型仍不逐位相同：三组固定 greedy 中只有一组和 MLX 相同。浮点归一化、exp、RoPE 与注意力仍会放大少量差异。因此浏览器后端标为实验实现。

实际 GPU 检查：40 组线性层夹具、81920 个线性输出误差为零；top-k 1/20/64 与完整 logits 排序一致；batch 1/8 的 40/320 token 检查最终 logits 相同，包含 KV 扩容。融合与未融合路径另外比较了数百万个 logits，误差为零。二进制分词器在诊断语料及 Unicode、特殊 token 边界夹具上与 JSON 分词器一致。

## 内存

| 项目 | 优化前浏览器 | 优化后浏览器 |
|---|---:|---:|
| WASM 线性内存 | 163.625 MiB | 50 MiB |
| GPU 权重 payload | 256.759 MiB | 256.759 MiB |
| GPU KV，256 / 512 token 容量 | 56 / 112 MiB | 56 / 112 MiB |
| GPU scratch/readback payload | 2.173 MiB | 2.173 MiB |

原生 MLX 短提示词：active 中位数 364.04 MiB，allocator peak 中位数 368.39 MiB。
原生 MLX 长提示词：active 中位数 420.04 MiB，allocator peak 中位数 495.22 MiB。

GPU payload 不包含 driver/uniform、KV 扩容时旧缓冲与新缓冲同时存在的瞬时占用；WASM 线性内存也不等于整个浏览器进程 RSS。MLX allocator 与浏览器逻辑 buffer 的统计口径不同，不能据此宣称整体进程内存降低了同样比例。

## 复现与记录

- `benchmarks/optimization.json`：三方轮换与冷/热加载完整结果。
- `benchmarks/optimization-quality.json`：优化前后逐项质量回归。
- `benchmarks/checks-latest.json`：实际 GPU、分词器、batch 与 KV 扩容检查。
- `benchmarks/comparison-raw.json`：原始浏览器与 MLX 的质量及初始性能记录。
- `benchmarks/baseline/`：优化前的 WASM、Worker 与 WGSL 源码；`native-reference/`：原生参考源码。

本机启动后打开 `/bench.html` 与 `/bench.html?variant=baseline`，运行 `python3 benchmarks/optimize.py`。原生参考二进制按 `native-reference/README.md` 重建，MLX SDK 与原始 checkpoint 路径可通过参数指定。公共 GitHub Pages 只提供静态文件；这些 benchmark RPC 端点只存在于本机静态测试服务。

当前发布 WASM SHA-256：`351972d309a62f9da67c18d8f624f2be260174e526fc79903c6f86fea54cdf77`。
