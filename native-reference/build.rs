use std::{env, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=BONSAI_MLX_SDK");
    println!("cargo:rerun-if-changed=src/mlx_bridge.cpp");
    println!("cargo:rerun-if-changed=src/w1a4.metal");
    if env::var_os("CARGO_FEATURE_MLX").is_none() {
        return;
    }
    assert!(
        env::var("CARGO_CFG_TARGET_OS").unwrap() == "macos",
        "MLX backend requires macOS"
    );
    let sdk = PathBuf::from(
        env::var_os("BONSAI_MLX_SDK")
            .expect("set BONSAI_MLX_SDK to the MLX SDK directory (include/ and lib/)"),
    );
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .warnings(false)
        .include(sdk.join("include"))
        .file("src/mlx_bridge.cpp")
        .flag("-O3")
        .compile("bonsai_mlx_bridge");
    println!(
        "cargo:rustc-link-search=native={}",
        sdk.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=mlx");
    println!("cargo:rustc-link-lib=dylib=c++");
    println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
}
