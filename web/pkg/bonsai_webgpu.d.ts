/* tslint:disable */
/* eslint-disable */

export class Engine {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    adapter(): string;
    static create(tokenizer_bytes: Uint8Array, capacity: number, batch: number): Promise<Engine>;
    decode(tokens: Uint32Array): string;
    encode(text: string): Uint32Array;
    /**
     * Returns [token-id-as-u32-bits, score] pairs from GPU partitions, globally sorted.
     */
    forward(tokens: Uint32Array, top_k: number): Promise<any>;
    /**
     * Real WebGPU pack and projection checks against a Rust scalar oracle.
     */
    kernel_check(): Promise<any>;
    load_blob(bytes: Uint8Array, layer: number): void;
    logits(token: number): Promise<Float32Array>;
    logits_chunk(tokens: Uint32Array): Promise<Float32Array>;
    memory(): any;
    position(): number;
    /**
     * Instrumented first layer for cross-backend operator comparison.
     * Developer-only dispatch profile. Separate passes perturb timings, so
     * use these to locate expensive kernels, not as end-to-end benchmarks.
     */
    profile(token: number, position: number): Promise<any>;
    reset(): void;
    set_reference(reference: boolean): void;
    trace(token: number): Promise<any>;
    truncate(pos: number): void;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_engine_free: (a: number, b: number) => void;
    readonly engine_adapter: (a: number) => [number, number];
    readonly engine_create: (a: number, b: number, c: number, d: number) => any;
    readonly engine_decode: (a: number, b: number, c: number) => [number, number, number, number];
    readonly engine_encode: (a: number, b: number, c: number) => [number, number, number, number];
    readonly engine_forward: (a: number, b: number, c: number, d: number) => any;
    readonly engine_kernel_check: (a: number) => any;
    readonly engine_load_blob: (a: number, b: any, c: number) => [number, number];
    readonly engine_logits: (a: number, b: number) => any;
    readonly engine_logits_chunk: (a: number, b: number, c: number) => any;
    readonly engine_memory: (a: number) => [number, number, number];
    readonly engine_position: (a: number) => number;
    readonly engine_profile: (a: number, b: number, c: number) => any;
    readonly engine_reset: (a: number) => void;
    readonly engine_set_reference: (a: number, b: number) => void;
    readonly engine_trace: (a: number, b: number) => any;
    readonly engine_truncate: (a: number, b: number) => [number, number];
    readonly wasm_bindgen_fddd185fc99cd29e___convert__closures_____invoke___js_sys_fbe64af394509c1f___Function_fn_wasm_bindgen_fddd185fc99cd29e___JsValue_____wasm_bindgen_fddd185fc99cd29e___sys__Undefined___js_sys_fbe64af394509c1f___Function_fn_wasm_bindgen_fddd185fc99cd29e___JsValue_____wasm_bindgen_fddd185fc99cd29e___sys__Undefined_______true_: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen_fddd185fc99cd29e___convert__closures_____invoke___wasm_bindgen_fddd185fc99cd29e___JsValue__core_ed718c3d60ebd546___result__Result_____wasm_bindgen_fddd185fc99cd29e___JsError___true_: (a: number, b: number, c: any) => [number, number];
    readonly wasm_bindgen_fddd185fc99cd29e___convert__closures_____invoke___wasm_bindgen_fddd185fc99cd29e___JsValue______true_: (a: number, b: number, c: any) => void;
    readonly wasm_bindgen_fddd185fc99cd29e___convert__closures_____invoke___wasm_bindgen_fddd185fc99cd29e___JsValue______true__19: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_destroy_closure: (a: number, b: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
