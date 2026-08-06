fn main() {
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/");

    // The `ira` feature links the IRA Fortran library. Its location is a build
    // input rather than a guess: a wrong default here fails at link time with a
    // message about a symbol rather than about a missing library.
    if std::env::var("CARGO_FEATURE_IRA").is_ok() {
        let dir = std::env::var("IRA_LIB_DIR")
            .expect("feature `ira` requires IRA_LIB_DIR pointing at libira");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=ira");
        println!("cargo:rerun-if-env-changed=IRA_LIB_DIR");
    }

    // The `gpr` feature links gpr_optim's C API, which carries the
    // gradient-enhanced Gaussian process. Same rule as IRA: the location is a
    // build input, and the rpath is emitted so a linked binary runs without the
    // caller having to set LD_LIBRARY_PATH by hand.
    if std::env::var("CARGO_FEATURE_GPR").is_ok() {
        let dir = std::env::var("GPR_OPTIM_LIB_DIR")
            .expect("feature `gpr` requires GPR_OPTIM_LIB_DIR pointing at libgpr_optim");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=gpr_optim");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo:rerun-if-env-changed=GPR_OPTIM_LIB_DIR");
    }
}
