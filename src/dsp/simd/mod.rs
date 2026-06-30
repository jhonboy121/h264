//! SIMD backends for the hottest DSP kernels, gated behind the `simd` feature.
//!
//! Every function here is a **bit-exact** drop-in replacement for the scalar
//! reference of the same name in the parent `dsp` modules (`transform`, `sad`,
//! `mc`). The public kernels keep their scalar body as a private `*_scalar`
//! function and dispatch to these on supported targets, falling back to scalar
//! everywhere else. Because the SIMD path is required to be byte-identical to
//! scalar, the *existing* kernel unit tests (and the conformance / round-trip
//! integration tests) validate the SIMD path unchanged when built with
//! `--features simd`.
//!
//! - **NEON** (`aarch64`): part of the architecture baseline, so no runtime
//!   detection is needed — the intrinsics are always available.
//! - **wasm `simd128`** (`wasm32` + `target_feature = "simd128"`): enabled at
//!   build time via `-C target-feature=+simd128`; otherwise the scalar path is
//!   compiled.
//!
//! Kernels that are *not* SIMD-accelerated (and stay scalar by design) are noted
//! in the relevant module: the luma centre half-pel `mc_hor_ver22` (two-pass with
//! a wide intermediate — kept scalar to stay provably bit-exact), `mc_copy` (a
//! plain memcpy the compiler already lowers well), and the deblock edge filters
//! (per-line data-dependent branching that does not map to a bit-exact mask).

#[cfg(simd_neon)]
pub mod neon;

#[cfg(simd_wasm128)]
pub mod wasm;
