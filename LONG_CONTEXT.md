# INT4 KV, 128K and experimental sparse prefill

W1A4 describes the quantized linear projections. Normalization, RoPE, attention probabilities and accumulators remain floating point. KV precision is a separate setting.

## KV layout

Default cache format: signed INT4 `[-7,7]`, group size 32, per-token scales for K and V, FP32 scale per group. Each group uses four packed words and one scale word: 20 bytes for 32 values. K is laid out by group/token for dot products; V by token/group for weighted reduction. RoPE is applied before K quantization. Attention decodes directly from the packed buffers, without a full floating-point cache.

For 28 layers, 8 KV heads and 128 dimensions:

| Context | INT4 g32 with scales | FP32 |
| --- | ---: | ---: |
| 32K | 1.09375 GiB | 7 GiB |
| 128K | 4.375 GiB | 28 GiB |

Allocation starts at 256 tokens (8.75 MiB total INT4 KV) and grows geometrically. Growth transposes K strides on GPU and copies V without re-quantization. Old and new buffers temporarily coexist; the figures above are steady allocated KV sizes, not total browser RAM or growth peaks. Weights and attention scratch are additional. Adapter storage-buffer limits are validated before loading.

## Long attention

The original shader stored every score in workgroup memory and capped context near 4K. The new path computes independent 128-key partial softmax results and merges them with max rescaling. Scratch stays bounded by the configured context and batch, rather than storing a dense L×L matrix. Short contexts retain the original fused attention/activation pack path. Decode uses full attention.

Runtime supports up to 131072 tokens. The checkpoint config itself specifies 32768 tokens, original length 8192, YaRN factor 4. Configuring 128K changes YaRN to factor 16 and its attention scale; it is an extrapolation experiment, not a claim of trained or validated 128K comprehension. Settings can return to the checkpoint's 32K configuration. FP32 KV remains available as a quality reference.

## FlashPrefill-inspired option

Official reference: [FlashPrefill repository](https://github.com/qhfan/FlashPrefill), commit `baa612047433a992a00d07dc178205eed065ae14`, and [paper](https://arxiv.org/abs/2603.06199).

The upstream implementation uses PyTorch/Triton. This code independently implements GPU key-block mean pooling, query-to-pooled-key energy scoring, max-relative thresholding, compact selected-block indices, and sparse partial attention in WGSL. It is explicitly an algorithm-inspired port, not a binary or performance-equivalent drop-in of upstream. It reuses this executor's 8-query microtiles; upstream uses 128-query tiles. Pooling is recomputed for each microtile, so its overhead can outweigh skipped attention. Upstream speedups do not apply to this browser implementation.

The option starts beyond 8K, retains the first two key blocks and the last four, uses full attention for the final two query blocks, and leaves all decode/draft verification dense. Default threshold alpha is 0.1. At the user's request, 128K and sparse prefill are enabled by default. It changes numerical/model outputs and remains experimental; settings retain a complete-attention option.

## Checks and limits of the evidence

`Engine.kv_check` uses synthetic data to verify real GPU RoPE/KV packing and compares split attention against a CPU FP64 softmax over the same quantized KV. Cases include 1, 127, 128, 129, 4097 and 131072 keys. A 128K synthetic one-head check is not a full 128K model evaluation or end-to-end throughput result.

Short-model NLL comparisons use the same 32K YaRN configuration for both precisions, separating cache quantization from position extrapolation. Quantization changes model quality and greedy outputs; exact parity is not an adoption condition. Long-text retrieval, RULER and representative agent-task evaluations are still needed before treating 128K or sparse attention as quality validated.
