# v0.4.1 local measurements

Machine: Apple M5 Pro, 48 GiB; Codex Chromium 155. Raw benchmark collection occurred before the final statistics instrumentation change described in RELEASE_0.4.md. `accepted` in these historical JSON rows counts reused candidate rows. The current runtime separately records `accepted` and `reused`.

- `kv-benchmark.json`: alternating FP32/INT4 KV runs at 32K YaRN factor 4; one warm-up pair (`round=-1`) and seven measured pairs, 64 tokens, greedy, drafts off. Contains identical-token bilingual diagnostic targets and one 10260-token dense/sparse prompt pair. NLL comes from per-token full logits.
- `ngram-benchmark.json`: INT4 KV, context 4096, minimum n-gram length 5 / maximum 8, draft length 4; temperature .5, top_k 20, top_p .85, seed 42; 64 tokens. One warm-up pair and seven alternating pairs for ordinary prose and repeated HTML. Token arrays permit exact equality checks.
- `gpu-checks.json`: real WebGPU packing/error-bound, attention, reference, batch/growth and draft-row equality checks. The 128K case checks synthetic one-head attention, not full-model 128K inference.
- `ocr-smoke.json`: fixture text and elapsed time for two ppocr WASM variants. OCR timings exclude model startup and browser image preprocessing.

Recompute summaries without model execution:

```sh
node benchmarks/v041/summary.mjs
```

For new runtime measurements, start the local server and open `/bench.html?variant=quality32&context=32768&kv_bits=32` or `/bench.html?variant=ngram4&context=4096&kv_bits=4`. The local benchmark bridge accepts JSON jobs at `/benchmark-command` and collects results at `/benchmark-status`. `web/bench.mjs` documents accepted operations; give each new job a unique `job_id` to avoid confusing prior results. Reset between paired generation runs and alternate order. The checked-in older `benchmarks/compare.py` compares browser/native MLX and is separate from these KV/draft comparisons.

Other browser tabs were active. Reported ranges are substantial; tiny median differences are not evidence of a speedup. Sparse prefill has one confounded timing pair and failed the requested answer. No full-model 128K comprehension result is present.
