# Browser Agent and interactive messages

The application keeps model inference in Rust/WASM + WebGPU. A separate Worker executes a JavaScript shell interpreter against a virtual `/workspace`. It does not start an operating-system shell.

## Native cards

`present_ui` publishes a version-1 definition with stable `id`, `type`, `title`, required text `summary`, and nested `data`. Supported types: comparison, grouped checklist, timeline, line/bar/pie chart, parameter controls, and node/edge diagram. `update_ui_state` changes an existing card's state. Text and tool cards retain their streamed offsets; completed cards can appear before the reply finishes. Unknown versions and malformed definitions produce corrective tool errors. Definitions are immutable after publication.

Rust `crates/ui-core` validates definitions, evaluates the safe expression AST, and merges flat state patches. Both native tests and browser WASM invoke the same core. There is no script evaluator or network primitive in the AST. Limits include 32 KiB definitions, 96 KiB frames, depth 16, 4096 evaluation steps, 256 sequence entries, 64 chart labels, 8 series, 12 controls, 32 nodes and 64 edges.

State is in IndexedDB, with transactional revisions, 64 snapshots and 128 idempotency records. BroadcastChannel synchronizes tabs on the same browser origin. Independent fields merge; conflicting edits to the same field reject atomically and offer adopt/reapply. Slider updates are local immediately, debounced for persistence, and flushed on release. Focus, scrolling and expansion remain personal. Editing or deleting messages prunes orphan cards.

An action that needs the Agent appends an intent to the sole composer and preserves its draft. Sending flushes pending card changes, then binds immutable version snapshots to the accepted user input. Later edits do not change that input. This static application has no Marina accounts or cross-device shared-state service. Its sharing boundary is tabs on the same origin/browser profile; server ACL and multi-account acceptance do not apply.

## Virtual workspace and artifacts

Tools: `bash`, `read`, `write`, `edit`, `grep`, `glob`. Shell supports the commands bundled by just-bash, pipes and redirects. Grep includes regex/literal matching, line numbers, case folding and glob filters. Glob supports `*`, `**`, `?`, character classes and dotfiles.

Shell uses hardened execution limits: 10 seconds, 3000 commands, 16 MiB filesystem, 256 KiB interpreter output, 128 KiB source. File writes are capped at 1 MiB; tool text sent to the model is capped at 4000 characters. There are no configured network, Node, Python, npm or native-binary execution capabilities. A Worker is terminated on cancellation, error or a 12-second wall timeout. Worker isolation alone does not remove browser network privileges.

Workspaces persist by origin, deployment base path and conversation. Files, directories, symlinks, modes and times survive reload. Operations serialize with Web Locks and transactional version checks; replay IDs prevent duplicate writes. Text attachments import into `/workspace` once. Conversation deletion deletes its workspace. There is no terminal interface.

The right artifact panel selects generated HTML, previews it, exposes editable source, supports download and sends edit intent through the main composer. Preview uses an opaque-origin `sandbox="allow-scripts"` iframe and CSP that blocks fetch/connect, embedded frames, external scripts/styles and forms. Local workspace CSS/JS/images can be bundled into the preview. Parent storage and DOM are inaccessible. This is not a claim that all possible iframe navigation is impossible.

## Images and OCR

The `+` menu places image upload immediately below Add files. ppocr-rs v0.11.4 runs silently in a separate browser Worker. Attachments keep the image thumbnail and extracted text; the language model receives extracted text, not pixels. It can read text in images but is not a vision model.

Verified model assets use CacheStorage. OCR result cache keys include SHA-256 of the normalized image and the OCR version/configuration; up to 128 results are retained. A result hit avoids loading the OCR WASM. Errors and cancellation remain visible on the attachment; sending waits for image processing.

## Tool JSON repair

Completed tool frames are first parsed strictly, then repaired with pinned `jsonrepair`. Repaired calls must still name an enabled tool and carry object arguments, and pass each tool's schema and bounds. A narrow schema repair also maps `type: object` to a uniquely identified card shape and wraps an existing flattened expression/op/args tuple. Repair does not invent missing required fields or repair generated application logic. Partial or cancelled generic mutations are not executed. Invalid definitions become tool errors so the model can retry.

## LangSearch

Settings → Tools accepts a user-supplied LangSearch Key in sessionStorage. It is not included in prompts, tool arguments, chat exports, repository files or public assets. `web_search` sends the query to the fixed official HTTPS endpoint with a 30-second deadline and returns bounded text and exact source URLs. The built-in tool permission gate applies. Results render as native source links. Source content remains untrusted data.

A public client cannot hide a shared API Key, including in encrypted JavaScript or WASM. A shared site key requires a separately hosted proxy with server-side secrets and quotas. This project does not publish the developer's Key.

## Source and rebuild

`ui-source` vendors llama.cpp's UI at commit `3d65c90d04d337e88f2b1f7f0061f40a5324e662`. Install its pinned dependencies with `cd ui-source && npm ci`. Run `./build.sh`, `node build-ui.mjs`, then `node build-site.mjs`. The latter exports to ignored `work/site`; GitHub Actions deploys that static output. Generated UI/WASM assets are checked in so Pages deployment does not need a Rust toolchain or npm install.

Added dependencies are pinned: just-bash 3.4.2, jsonrepair 3.15.0, minimatch 10.2.6, fflate 0.8.2, buffer 6.0.3 and process 0.11.10. A small browser shim resolves just-bash's static node:zlib imports using fflate; it does not expose Node. Third-party license copies are under `web/licenses`, OCR provenance under `web/ocr/NOTICE`; upstream UI retains its LICENSE.
