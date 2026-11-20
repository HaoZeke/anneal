fn main() {
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../CppCore/rgpot");
    let morse = root.join("Morse");
    let gupta = root.join("Gupta");
    let mut build = cc::Build::new();
    // The kernels declare loop counters in the for statement and variables
    // after statements. GCC before 5, the manylinux2014 cross compiler,
    // defaults to gnu90 and rejects both. MSVC takes C99 by default.
    if !build.get_compiler().is_like_msvc() {
        build.std("gnu99");
    }
    build
        .file(morse.join("cambridge_morse.c"))
        .file(gupta.join("gupta.c"))
        .include(&morse)
        .include(&gupta)
        .flag("-ffp-contract=off")
        .compile("rgpot_cluster_kernels");
    for name in [
        "Morse/cambridge_morse.c",
        "Morse/cambridge_morse.h",
        "Morse/cambridge_morse_body.inc",
        "Morse/cluster_lj_body.inc",
        "Gupta/gupta.c",
        "Gupta/gupta.h",
        "Gupta/gupta_body.inc",
        "cpu_features.h",
    ] {
        println!("cargo:rerun-if-changed={}", root.join(name).display());
    }
}
