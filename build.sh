#!/bin/sh
set -eu
BONSAI_ROOT=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
BONSAI_RUSTC=$(rustup which rustc)
BONSAI_RUST_TOOLCHAIN=$(dirname "$(dirname "$BONSAI_RUSTC")")
export BONSAI_RUST_TOOLCHAIN
export PATH="$BONSAI_RUST_TOOLCHAIN/bin:$PATH"
export RUSTC="$BONSAI_RUSTC"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$BONSAI_ROOT/work/target}"
export RUSTFLAGS='--cfg getrandom_backend="wasm_js"'
mkdir -p "$BONSAI_ROOT/.build"
if [ "$(uname -s)" = Darwin ]; then
 cat > "$BONSAI_ROOT/.build/wasm-linker.sh" <<'SH'
#!/bin/sh
export DYLD_LIBRARY_PATH="$BONSAI_RUST_TOOLCHAIN/lib"
exec "$BONSAI_RUST_TOOLCHAIN/lib/rustlib/aarch64-apple-darwin/bin/rust-lld" "$@"
SH
 chmod +x "$BONSAI_ROOT/.build/wasm-linker.sh"
 export CARGO_TARGET_WASM32_UNKNOWN_UNKNOWN_LINKER="$BONSAI_ROOT/.build/wasm-linker.sh"
fi
wasm-pack build "$BONSAI_ROOT" --target web --out-dir web/pkg --release
