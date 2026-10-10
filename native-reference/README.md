# Native reference

This is the Rust + MLX C++ reference used for the recorded comparisons. Requires macOS ARM64 and an MLX C++ SDK containing `include/`, `lib/libmlx.dylib`, and `lib/mlx.metallib`.

```sh
BONSAI_MLX_SDK=/path/to/mlx-sdk cargo build --release --features mlx --bin bonsai-mlx-chat-worker
cp target/release/bonsai-mlx-chat-worker ../benchmarks/bonsai-mlx-benchmark
```

The benchmark scripts set `BONSAI_BACKEND=mlx` and check the worker's reported backend before recording results. The browser implementation requires no MLX SDK, native library or Python inference process.
