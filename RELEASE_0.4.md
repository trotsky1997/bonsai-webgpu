# v0.4.1 implementation and acceptance record

Date: 2026-10-10. Browser inference remains local Rust/WASM + WebGPU. This release includes browser Agent tools, native interactive messages, OCR caching, exact n-gram drafts, INT4 KV and experimental long context. The static site uses same-browser storage; it has no Marina account or shared backend.

## Current defaults and statistics

- Context and KV upper bound: **131072 tokens**. Initial KV capacity: 256 tokens; geometric growth on demand. INT4 g32 KV at 128K occupies **4697620480 bytes (4.375 GiB)** across 28 layers, excluding weights, scratch and temporary growth copies. The runtime checks adapter buffer limits and offers smaller-context retry.
- Sparse prefill enabled, active only beyond 8K prompt tokens. Decode and speculative verification retain full attention. 128K uses experimental YaRN factor 16; the checkpoint's configuration is 32K / factor 4.
- Token usage details shows accepted/proposed draft tokens, accepted/batch average (including zero-accept batches), cumulative accepted tokens, and cached/full-prompt KV hit rate. Aggregation is weighted over the active conversation branch, using each request once. Counters persist; the GPU cache itself does not survive reload.
- Actual two-turn browser check: 84 accepted / 92 proposed = 91.3%; 23 batches, mean prefix 3.65 tokens. KV: second turn 1243 / 1349 = 92.1%; cumulative 1243 / 2596 = 47.9%. Reload preserved the displayed counters. Runtime reported version 0.4.1, context 131072, kv_bits 4, full KV bytes 4697620480.

## Native UI and Agent acceptance

Real Bonsai tool calls produced all six types: comparison, grouped checklist, timeline, chart, parameters and diagram. Checked selection, checklist changes, timeline details, bar chart text table, slider calculations, keyboard node selection, refresh restoration and draft-preserving follow-up input. Parameter changes synchronized across two tabs. Rust/WASM enforces definition, expression and state bounds. Unit checks cover independent-field merge, atomic same-field conflict, idempotency, replay and unknown versions.

The six-type model checks used guided complete definitions. The model also emitted invalid definitions, prose pretending to call tools, malformed HTML and repetitive output in other trials. JSON repair and schema feedback help parse calls; they do not make this 1.7B model a reliable general Agent. Actual two-account ACL tests do not apply to this static same-origin implementation. A full screen-reader audit and a manual two-tab conflict/offline matrix have not been completed.

Browser workspace tools executed write/read/edit and shell CSV aggregation in automated checks. A real model write/read produced an HTML artifact whose button changed its content to 42; artifact source and preview survived reload. Glob and grep pass recursive/filter/bounds/injection tests. A real model trial supplied boolean `glob` to grep and received a corrective type error; that trial is not counted as a successful grep task. The right panel renders HTML in a restricted opaque iframe. No terminal is exposed.

Silent image upload was checked through the `+` menu and a real Chinese image question. Uploading the same image again hit the result cache. Both ppocr-rs v0.11.4 WASM variants passed Chinese, English, rotated and blank fixtures. Image OCR text is provided as untrusted text context; the model receives no image pixels. A regression in ordinary image questions was fixed: attachment filenames are no longer sent as readable workspace paths, and direct image-text questions omit file tools. This prevents the observed ENOENT loop; it does not provide general visual recognition.

Authenticated LangSearch requests and source cards were checked. The Key stays in sessionStorage and is absent from prompts and public files. Model-generated source prose sometimes changed URLs; source cards retain the actual API URLs.

Responsive checks at 390, 768, 1024 and 1440px found no page horizontal overflow for the inspected native chart. Dark theme and reduced-motion rendering were checked. Interactive control targets inspected were at least 44px.

## Measured performance and quality

Apple M5 Pro, 48 GiB; Codex embedded Chromium 155. Seven alternating measured pairs after one warm-up pair; 64 generated tokens, throughput `(64-1)/decode_seconds`. Other browser tabs were in use; these are local paired observations, not exclusive-GPU laboratory measurements. Raw data: [benchmarks/v041](benchmarks/v041).

| Experiment | Baseline median (range), tok/s | Candidate median (range), tok/s | Output |
| --- | ---: | ---: | --- |
| INT4 KV vs FP32 KV, 32K configuration, greedy, drafts off | 222.87 (211.21–267.41) | 226.30 (188.56–263.14) | Quantization changes output |
| N-gram drafts, ordinary prose, INT4 KV | 215.73 (189.50–250.83) | 214.90 (175.33–245.31) | All paired tokens identical; no proposals |
| N-gram drafts, repeated HTML, INT4 KV | 219.50 (152.68–238.02) | 518.73 (313.96–601.29) | All paired tokens identical; 40 reused / 41 proposed |

N-gram gains depend on repeated text. The repeated HTML workload gained 2.36×; ordinary prose showed no gain. The archived `accepted` counter originally meant candidate-row reuse; v0.4.1 separately records accepted prefixes and reused rows, including accepted terminal tokens in the former. This instrumentation change does not alter sampling.

Short bilingual diagnostic: 87 scored tokens, same 32K YaRN configuration. Mean NLL: FP32 KV **4.903536**, INT4 KV **5.030834**, delta **+0.127298**. This establishes a quality cost and is not representative long-context evaluation. INT4's small median throughput difference is not a demonstrated speedup given the ranges.

A single 10260-token repeated prompt measured dense prefill 51.818 s and sparse 46.393 s. Output changed and both runs failed the requested answer. Growth/order effects were not removed; no stable sparse speedup or quality result is claimed. Full-model 128K quality, retrieval and RULER remain unvalidated.

GPU checks: synthetic packed KV and split attention at lengths 1, 127, 128, 129, 4097 and 131072 passed quantization error bounds; the 128K one-head result had maximum attention error 2.867e-5 against CPU FP64 over the same quantized KV. It is not a full-model 128K throughput result. INT4 linear reference check: 1365021 logits, zero differences. Sequential vs batch-8 over 512 input tokens: zero final-logit differences, including KV growth. Draft candidate rows matched sequential execution for top_k 1, 20 and 64.

## Checks

- Svelte check: 0 errors, 0 warnings.
- Frontend unit tests: 58 files, 709 tests passed.
- Rust: 3 KV tests and 4 UI-core tests passed.
- Node: UI-core WASM six definitions/calculation bounds; protocol/automatic JSON repair/image routing; bounded n-gram tests passed.
- OCR: both distributed WASM variants passed 8 fixture cases plus invalid input and disposal checks.
- Release WASM build, Svelte production build and static export passed.
- Publication excludes development logs, node_modules, target directories, session data, user images and API credentials.

Protocol and limits: [INTERACTIVE_UI.md](INTERACTIVE_UI.md). KV layout and numerical limitations: [LONG_CONTEXT.md](LONG_CONTEXT.md).

## Final prompt and viewport update

The usage popup now stays inside the viewport and scrolls internally (observed top 16px, bottom 657px at a 714px viewport). The published model generated a native line chart through a real present_ui call. The final prompt uses minimal direct-answer instructions, capability-specific Agent rules and execution-level enforcement of explicit tool prohibition. Fixed examples and remaining failures are recorded in [SYSTEM_PROMPT.md](SYSTEM_PROMPT.md) and `benchmarks/v041/prompt-eval.json`; no overall Agent reliability gain is claimed.
