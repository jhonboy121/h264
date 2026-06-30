//! Build script that selects exactly one mutually-exclusive SIMD backend cfg
//! for the current target, so the DSP dispatchers can gate their scalar and
//! SIMD paths without any dead (`unreachable`) code.
//!
//! Emits exactly one of:
//! - `simd_neon`    — `simd` feature on and `target_arch == "aarch64"`
//! - `simd_wasm128` — `simd` feature on, `target_arch == "wasm32"`, and the
//!   target feature list contains `simd128`
//! - `no_simd`      — everything else (the scalar reference path)

fn main() {
    let simd = std::env::var_os("CARGO_FEATURE_SIMD").is_some();
    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_simd128 = features.split(',').any(|f| f == "simd128");

    // Always declare the custom cfgs so `unexpected_cfgs` stays silent.
    println!("cargo::rustc-check-cfg=cfg(simd_neon)");
    println!("cargo::rustc-check-cfg=cfg(simd_wasm128)");
    println!("cargo::rustc-check-cfg=cfg(no_simd)");

    if simd && arch == "aarch64" {
        println!("cargo::rustc-cfg=simd_neon");
    } else if simd && arch == "wasm32" && has_simd128 {
        println!("cargo::rustc-cfg=simd_wasm128");
    } else {
        println!("cargo::rustc-cfg=no_simd");
    }
}
