fn main() {
    // The `vesin-nl` feature compiles vesin's own cell-list sources rather
    // than reimplementing them. VESIN_SRC points at a checkout of
    // https://github.com/Luthaf/vesin.
    if std::env::var("CARGO_FEATURE_VESIN_NL").is_ok() {
        let src = std::env::var("VESIN_SRC")
            .expect("feature `vesin-nl` requires VESIN_SRC pointing at a vesin checkout");
        let base = format!("{src}/vesin");
        let mut build = cc::Build::new();
        build
            .cpp(true)
            .std("c++17")
            .include(format!("{base}/include"))
            .include(format!("{base}/src"))
            .define("VESIN_STATIC", None);
        for f in [
            "vesin.cpp",
            "cpu_cell_list.cpp",
            "verlet.cpp",
            "shared_libraries.cpp",
            "threadpool.cpp",
            "vesin_cuda_stub.cpp",
        ] {
            build.file(format!("{base}/src/{f}"));
        }
        build.compile("vesin");
        println!("cargo:rerun-if-env-changed=VESIN_SRC");
    }
    println!("cargo:rerun-if-changed=cbindgen.toml");
    println!("cargo:rerun-if-changed=include/");

    // The `ira` feature links the IRA Fortran library. Its location is a build
    // input rather than a guess: a wrong default here fails at link time with a
    // message about a symbol rather than about a missing library.
    // The `lammps` feature links the LAMMPS C library for in-process MD
    // segments. Same discipline as `ira`: the location is a build input.
    if std::env::var("CARGO_FEATURE_LAMMPS").is_ok() {
        let dir = std::env::var("LAMMPS_LIB_DIR")
            .expect("feature `lammps` requires LAMMPS_LIB_DIR pointing at liblammps");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=lammps");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo:rerun-if-env-changed=LAMMPS_LIB_DIR");
    }
    if std::env::var("CARGO_FEATURE_IRA").is_ok() {
        let dir = std::env::var("IRA_LIB_DIR")
            .expect("feature `ira` requires IRA_LIB_DIR pointing at libira");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=ira");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
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
