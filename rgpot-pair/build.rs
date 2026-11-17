fn main() {
    let root = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../CppCore/rgpot");
    let morse = root.join("Morse");
    let gupta = root.join("Gupta");
    cc::Build::new()
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
    ] {
        println!("cargo:rerun-if-changed={}", root.join(name).display());
    }
}
