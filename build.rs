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
            "threadpool.cpp",
            "shared_libraries.cpp",
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
    if std::env::var("CARGO_FEATURE_IRA").is_ok() {
        let dir = std::env::var("IRA_LIB_DIR")
            .expect("feature `ira` requires IRA_LIB_DIR pointing at libira");
        println!("cargo:rustc-link-search=native={dir}");
        println!("cargo:rustc-link-lib=dylib=ira");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{dir}");
        println!("cargo:rerun-if-env-changed=IRA_LIB_DIR");
    }
}
