/** PP-OCRv6 tiny: independent resident WASM model, no external runtime. */
// Minimal (v128,v128,v128)->v128 module containing f32x4.relaxed_madd.
export function supportsRelaxedSimd() {
  return WebAssembly.validate(
    new Uint8Array([
      0, 97, 115, 109, 1, 0, 0, 0, 1, 8, 1, 96, 3, 123, 123, 123, 1, 123, 3, 2,
      1, 0, 10, 13, 1, 11, 0, 32, 0, 32, 1, 32, 2, 253, 133, 2, 11,
    ]),
  );
}
export async function createOcr({
  wasmBytes,
  wasmUrl = new URL("./ppocr_rs.wasm", import.meta.url),
  relaxedWasmUrl,
  precision = "int8",
  detMaxSide = 960,
  recMinWidth = 320,
} = {}) {
  const modes = ["fp32", "int8", "int4", "binary", "w1a8", "w4a4"];
  const mode = modes.indexOf(precision);
  if (mode < 0) throw new RangeError(`Unknown precision: ${precision}`);
  for (const value of [detMaxSide, recMinWidth]) {
    if (!Number.isInteger(value) || value < 0 || value > 0xffffffff)
      throw new RangeError("Invalid dimensions");
  }
  if (!wasmBytes) {
    const selectedUrl =
      relaxedWasmUrl && supportsRelaxedSimd() ? relaxedWasmUrl : wasmUrl;
    const response = await fetch(selectedUrl);
    if (!response.ok)
      throw new Error(`WASM download failed: HTTP ${response.status}`);
    wasmBytes = await response.arrayBuffer();
  }
  const { instance } = await WebAssembly.instantiate(wasmBytes, {
    ppocr: { now_ms: () => performance.now() },
  });
  const e = instance.exports;
  let handle = e.ppocr_create();
  const decoder = new TextDecoder();
  function output(status) {
    // Inference can grow memory; create the view after the exported call.
    const ptr = e.ppocr_output_ptr(handle),
      len = e.ppocr_output_len(handle);
    const value = JSON.parse(
      decoder.decode(new Uint8Array(e.memory.buffer, ptr, len)),
    );
    if (status) throw new Error(value.error);
    return value;
  }
  let info;
  try {
    info = output(e.ppocr_init(handle, mode, detMaxSide, recMinWidth));
  } catch (error) {
    e.ppocr_destroy(handle);
    handle = 0;
    throw error;
  }
  function run(input, kind) {
    if (!handle) throw new Error("OCR instance is disposed");
    const bytes = input instanceof ArrayBuffer ? new Uint8Array(input) : input;
    if (!(bytes instanceof Uint8Array))
      throw new TypeError(
        "Expected encoded image bytes (Uint8Array or ArrayBuffer)",
      );
    if (bytes.byteLength > 20 * 1024 * 1024)
      throw new RangeError("Image exceeds 20 MiB");
    const ptr = e.ppocr_input_resize(handle, bytes.byteLength);
    if (!ptr) throw new Error("Input allocation failed");
    new Uint8Array(e.memory.buffer, ptr, bytes.byteLength).set(bytes);
    return output(e.ppocr_run(handle, kind));
  }
  return Object.freeze({
    info: Object.freeze({ ...info, precision }),
    ocr: (bytes) => run(bytes, 0),
    recognize: (bytes) => run(bytes, 1),
    dispose() {
      if (handle) {
        e.ppocr_destroy(handle);
        handle = 0;
      }
    },
  });
}
