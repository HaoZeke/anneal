use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=GATE_PAD_KB");
    let pad_kb: usize = env::var("GATE_PAD_KB")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(0);
    let pad_bytes = pad_kb.saturating_mul(1024);
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));
    let dest = out_dir.join("gate_padding.rs");
    let contents = if pad_bytes == 0 {
        String::from(
            "pub const GATE_PAD_KB_BUILT: usize = 0;\n#[inline(never)]\npub fn touch_padding() -> u8 { 0 }\n",
        )
    } else {
        format!(
            "pub const GATE_PAD_KB_BUILT: usize = {pad_kb};\n#[used]\n#[unsafe(link_section = \".rodata.gate_pad\")]\nstatic GATE_PAD: [u8; {pad_bytes}] = [0xA5; {pad_bytes}];\n#[inline(never)]\npub fn touch_padding() -> u8 {{\n    std::hint::black_box(GATE_PAD[0]) ^ std::hint::black_box(GATE_PAD[{last}])\n}}\n",
            last = pad_bytes - 1,
        )
    };
    fs::write(dest, contents).expect("write gate_padding.rs");
}
