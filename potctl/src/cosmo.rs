//! Cosmopolitan APE build + rustc linker shim (replaces potctl/cosmo/*.sh|*.bash).
//!
//! Policy: potctl owns repo/CI control plane in Rust; no project-local bash for cosmo.
//! The linker entry point is a separate bin (`potctl-cosmo-ld`) because rustc invokes
//! the target `linker` as an executable with only link args (no subcommand).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::OnceLock;

// --- shared helpers ----------------------------------------------------------

fn err(msg: impl Into<String>) -> String {
    msg.into()
}

fn which(name: &str) -> Option<PathBuf> {
    let path = env::var_os("PATH")?;
    for dir in env::split_paths(&path) {
        let cand = dir.join(name);
        if cand.is_file() {
            return Some(cand);
        }
    }
    None
}

fn find_exec_named(root: &Path, name: &str) -> Option<PathBuf> {
    if !root.is_dir() {
        return None;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|s| s.to_str()) == Some(name) && is_exec(&p) {
                return Some(p);
            }
        }
    }
    None
}

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn find_first_file_named(root: &Path, name: &str) -> Option<PathBuf> {
    if !root.is_dir() {
        return None;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.file_name().and_then(|s| s.to_str()) == Some(name) && p.is_file() {
                return Some(p);
            }
        }
    }
    None
}

fn find_path_suffix(root: &Path, parts: &[&str]) -> Option<PathBuf> {
    if !root.is_dir() {
        return None;
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
            } else if p.is_file() {
                let s = p.to_string_lossy();
                if parts.iter().all(|seg| s.contains(seg)) {
                    return Some(p);
                }
            }
        }
    }
    None
}

#[cfg(unix)]
fn is_exec(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    fs::metadata(p)
        .map(|m| m.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

#[cfg(not(unix))]
fn is_exec(p: &Path) -> bool {
    p.is_file()
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn cosmo_env() -> PathBuf {
    env::var_os("COSMO")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/opt/cosmo"))
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn trace_enabled() -> bool {
    env::var("POTCTL_COSMO_LINKER_TRACE").ok().as_deref() == Some("1")
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn trace(msg: &str) {
    if trace_enabled() {
        eprintln!("cosmo-linker: {msg}");
    }
}

// --- linker (potctl-cosmo-ld) ------------------------------------------------

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn cosmo_toolchain_root(cc: &Path) -> PathBuf {
    let d = cc
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));
    if d.file_name().and_then(|s| s.to_str()) == Some("bin") {
        d.parent().unwrap_or(&d).to_path_buf()
    } else {
        d
    }
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn basename_str(p: &Path) -> String {
    p.file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_string()
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn resolve_cosmocc(cosmo: &Path) -> Option<PathBuf> {
    if let Ok(cc) = env::var("COSMO_CC") {
        let p = PathBuf::from(&cc);
        if is_exec(&p) {
            let b = basename_str(&p);
            if b == "cosmocc" || b == "cosmoc++" {
                return Some(p);
            }
        }
    }
    if let Some(p) = which("cosmocc") {
        return Some(p);
    }
    let dot = cosmo.join(".cosmocc");
    if let Some(p) = find_exec_named(&dot, "cosmocc") {
        return Some(p);
    }
    let bin = cosmo.join("bin/cosmocc");
    if is_exec(&bin) {
        return Some(bin);
    }
    None
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn resolve_x86_cosmo_gcc(cosmo: &Path) -> Option<PathBuf> {
    if let Ok(cc) = env::var("COSMO_CC") {
        let p = PathBuf::from(&cc);
        if is_exec(&p) {
            let b = basename_str(&p);
            if b == "x86_64-linux-cosmo-gcc" || b == "x86_64-unknown-cosmo-cc" {
                return Some(p);
            }
        }
    }
    if let Some(p) = which("x86_64-linux-cosmo-gcc") {
        return Some(p);
    }
    let dot = cosmo.join(".cosmocc");
    if let Some(p) = find_exec_named(&dot, "x86_64-linux-cosmo-gcc") {
        return Some(p);
    }
    let p = cosmo.join("o/third_party/gcc/bin/x86_64-linux-cosmo-gcc");
    if is_exec(&p) {
        return Some(p);
    }
    None
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn resolve_cosmo_cc(cosmo: &Path) -> Result<PathBuf, String> {
    if let Some(p) = resolve_cosmocc(cosmo) {
        return Ok(p);
    }
    if let Some(p) = resolve_x86_cosmo_gcc(cosmo) {
        return Ok(p);
    }
    Err(err(format!(
        "no cosmo linker (cosmocc or x86_64-linux-cosmo-gcc); COSMO={}",
        cosmo.display()
    )))
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn cc_is_cosmocc(cc: &Path) -> bool {
    let b = basename_str(cc);
    b == "cosmocc" || b == "cosmoc++"
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn resolve_aarch64_cc(cosmo: &Path, cc: &Path) -> Option<PathBuf> {
    if let Some(p) = which("aarch64-linux-cosmo-gcc") {
        return Some(p);
    }
    let dot = cosmo.join(".cosmocc");
    if let Some(p) = find_exec_named(&dot, "aarch64-linux-cosmo-gcc") {
        return Some(p);
    }
    let p = cosmo.join("o/third_party/gcc/bin/aarch64-linux-cosmo-gcc");
    if is_exec(&p) {
        return Some(p);
    }
    let root = cosmo_toolchain_root(cc);
    let p = root.join("bin/aarch64-linux-cosmo-gcc");
    if is_exec(&p) {
        return Some(p);
    }
    None
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn find_prebuilt_aarch64_obj(cosmo: &Path, cc: &Path) -> Option<PathBuf> {
    let root = cosmo_toolchain_root(cc);
    for f in [
        root.join("aarch64-linux-cosmo/lib/crt.o"),
        root.join("aarch64-linux-cosmo/lib/crt1.o"),
        root.join("aarch64-linux-cosmo/lib/ape.o"),
    ] {
        if f.is_file() {
            return Some(f);
        }
    }
    let dot = cosmo.join(".cosmocc");
    if let Some(p) = find_path_suffix(&dot, &["aarch64-linux-cosmo", "lib", "crt.o"]) {
        return Some(p);
    }
    if let Some(p) = find_path_suffix(&dot, &["aarch64-linux-cosmo", "lib"]) {
        if p.extension().and_then(|e| e.to_str()) == Some("o") {
            return Some(p);
        }
    }
    None
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
struct Aarch64Stubs {
    main_o: PathBuf,
    pad_o: PathBuf,
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
static A64_MAIN_PLACED: AtomicBool = AtomicBool::new(false);
#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
static A64_STUBS: OnceLock<Aarch64Stubs> = OnceLock::new();

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn compile_aarch64_c_to_o(
    a64_cc: &Path,
    csrc: &Path,
    outo: &Path,
    work: &Path,
) -> Result<(), String> {
    let root = cosmo_toolchain_root(a64_cc);
    let a64lib = root.join("aarch64-linux-cosmo/lib");
    let a64inc = root.join("aarch64-linux-cosmo/include");
    let err_path = work.join("a64-compile.err");

    let try_compile = |extra: &[String]| -> bool {
        let mut cmd = Command::new(a64_cc);
        for e in extra {
            cmd.arg(e);
        }
        cmd.arg("-c").arg(csrc).arg("-o").arg(outo);
        let out = cmd.output().ok();
        if let Some(ref o) = out {
            let _ = fs::write(&err_path, &o.stderr);
            if !o.status.success() || !outo.is_file() {
                return false;
            }
            // Reject accidental x86 objects.
            if let Ok(file_out) = Command::new("file").arg(outo).output() {
                let s = String::from_utf8_lossy(&file_out.stdout);
                if s.to_ascii_lowercase().contains("x86-64")
                    || s.to_ascii_lowercase().contains("intel 80386")
                {
                    return false;
                }
            }
            return true;
        }
        false
    };

    let mut attempts: Vec<Vec<String>> = Vec::new();
    if a64lib.is_dir() {
        let b = format!("-B{}", a64lib.display());
        attempts.push(vec![
            "-nostdinc".into(),
            "-ffreestanding".into(),
            "-fno-builtin".into(),
            b.clone(),
        ]);
        attempts.push(vec!["-ffreestanding".into(), "-fno-builtin".into(), b]);
    }
    attempts.push(vec![
        "-nostdinc".into(),
        "-ffreestanding".into(),
        "-fno-builtin".into(),
    ]);
    attempts.push(vec!["-ffreestanding".into(), "-fno-builtin".into()]);
    if a64inc.is_dir() {
        attempts.push(vec![
            "-nostdinc".into(),
            "-isystem".into(),
            a64inc.display().to_string(),
            "-ffreestanding".into(),
            "-fno-builtin".into(),
        ]);
    }
    attempts.push(vec![]);

    for a in &attempts {
        if try_compile(a) {
            return Ok(());
        }
    }
    // Always surface compiler stderr on failure (CI needs this without trace flag).
    if let Ok(e) = fs::read_to_string(&err_path) {
        if !e.trim().is_empty() {
            eprintln!(
                "cosmo-linker: aarch64 compile failed for {}:\n{e}",
                csrc.display()
            );
        }
    }
    Err(err(format!(
        "aarch64 compile failed for {}",
        csrc.display()
    )))
}

/// Minimal aarch64 ELF64 ET_REL with global `main` = `mov w0,#0; ret` (no libc).
/// Cosmocc's aarch64 fat pass only needs a real aarch64 ELF companion with `main` once.
#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn write_minimal_aarch64_main_o(path: &Path) -> Result<(), String> {
    // Layout (offsets in bytes):
    // 0x00: ELF header (64)
    // 0x40: .text (8 bytes code)
    // 0x48: .shstrtab
    // 0x60: .strtab (symbol names)
    // 0x70: .symtab (3 entries x 24)
    // 0xB8: section headers (5 x 64 = 320) at 0xB8
    // Total ~0x1F8
    let mut b = vec![0u8; 0x200];
    // e_ident
    b[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    b[4] = 2; // ELFCLASS64
    b[5] = 1; // ELFDATA2LSB
    b[6] = 1; // EV_CURRENT
              // e_type ET_REL
    b[16] = 1;
    b[17] = 0;
    // e_machine EM_AARCH64 = 183
    b[18] = 183;
    b[19] = 0;
    // e_version
    b[20] = 1;
    // e_ehsize
    b[52] = 64;
    // e_shentsize
    b[58] = 64;
    // e_shnum = 5
    b[60] = 5;
    // e_shstrndx = 2
    b[62] = 2;
    // e_shoff = 0xB8
    b[40] = 0xB8;

    // .text at 0x40: mov w0,#0; ret
    b[0x40..0x44].copy_from_slice(&0x52800000u32.to_le_bytes()); // mov w0, #0
    b[0x44..0x48].copy_from_slice(&0xd65f03c0u32.to_le_bytes()); // ret

    // .shstrtab at 0x48: \0 .text \0 .shstrtab \0 .strtab \0 .symtab \0
    let shstr: &[u8] = b"\0.text\0.shstrtab\0.strtab\0.symtab\0";
    b[0x48..0x48 + shstr.len()].copy_from_slice(shstr);

    // .strtab at 0x68: \0 main \0
    let strtab: &[u8] = b"\0main\0";
    let strtab_off = 0x68usize;
    b[strtab_off..strtab_off + strtab.len()].copy_from_slice(strtab);

    // .symtab at 0x70: null + main (STT_FUNC, STB_GLOBAL, shndx=1)
    let symtab_off = 0x70usize;
    // entry 0: null
    // entry 1: main — st_name=1, st_info=STB_GLOBAL<<4|STT_FUNC=0x12, st_shndx=1, st_value=0, st_size=8
    let e1 = symtab_off + 24;
    b[e1..e1 + 4].copy_from_slice(&1u32.to_le_bytes()); // st_name
    b[e1 + 4] = 0x12; // GLOBAL FUNC
    b[e1 + 6] = 1; // st_shndx low
    b[e1 + 7] = 0;
    b[e1 + 8..e1 + 16].copy_from_slice(&0u64.to_le_bytes()); // st_value
    b[e1 + 16..e1 + 24].copy_from_slice(&8u64.to_le_bytes()); // st_size

    // Section headers at 0xB8 (5 entries)
    let sh_off = 0xB8usize;
    let write_sh = |b: &mut [u8],
                    idx: usize,
                    name: u32,
                    sh_type: u32,
                    flags: u64,
                    addr: u64,
                    off: u64,
                    size: u64,
                    link: u32,
                    info: u32,
                    addralign: u64,
                    entsize: u64| {
        let o = sh_off + idx * 64;
        b[o..o + 4].copy_from_slice(&name.to_le_bytes());
        b[o + 4..o + 8].copy_from_slice(&sh_type.to_le_bytes());
        b[o + 8..o + 16].copy_from_slice(&flags.to_le_bytes());
        b[o + 16..o + 24].copy_from_slice(&addr.to_le_bytes());
        b[o + 24..o + 32].copy_from_slice(&off.to_le_bytes());
        b[o + 32..o + 40].copy_from_slice(&size.to_le_bytes());
        b[o + 40..o + 44].copy_from_slice(&link.to_le_bytes());
        b[o + 44..o + 48].copy_from_slice(&info.to_le_bytes());
        b[o + 48..o + 56].copy_from_slice(&addralign.to_le_bytes());
        b[o + 56..o + 64].copy_from_slice(&entsize.to_le_bytes());
    };
    // 0 NULL
    write_sh(&mut b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    // 1 .text PROGBITS AX off=0x40 size=8
    write_sh(&mut b, 1, 1, 1, 6, 0, 0x40, 8, 0, 0, 4, 0);
    // 2 .shstrtab STRTAB off=0x48 size=shstr.len
    write_sh(&mut b, 2, 7, 3, 0, 0, 0x48, shstr.len() as u64, 0, 0, 1, 0);
    // 3 .strtab STRTAB off=0x68
    write_sh(
        &mut b,
        3,
        17,
        3,
        0,
        0,
        strtab_off as u64,
        strtab.len() as u64,
        0,
        0,
        1,
        0,
    );
    // 4 .symtab SYMTAB off=0x70 size=48 (2 entries), link=3 (.strtab), info=1, entsize=24
    write_sh(&mut b, 4, 25, 2, 0, 0, symtab_off as u64, 48, 3, 1, 8, 24);

    fs::write(path, &b).map_err(|e| err(e.to_string()))?;
    Ok(())
}

/// Minimal aarch64 pad object: empty .text, no global symbols (avoids multidef when copied many times).
#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn write_minimal_aarch64_pad_o(path: &Path) -> Result<(), String> {
    let mut b = vec![0u8; 0x180];
    b[0..4].copy_from_slice(&[0x7f, b'E', b'L', b'F']);
    b[4] = 2;
    b[5] = 1;
    b[6] = 1;
    b[16] = 1; // ET_REL
    b[18] = 183; // EM_AARCH64
    b[20] = 1;
    b[52] = 64;
    b[58] = 64;
    b[60] = 4; // shnum
    b[62] = 2; // shstrndx
    b[40] = 0x80; // shoff
                  // .text: one nop
    b[0x40..0x44].copy_from_slice(&0xd503201fu32.to_le_bytes());
    let shstr: &[u8] = b"\0.text\0.shstrtab\0.strtab\0";
    b[0x48..0x48 + shstr.len()].copy_from_slice(shstr);
    // .strtab at 0x60 is one NUL. The buffer is already zero there.
    b[0x60] = 0;
    let sh_off = 0x80usize;
    let write_sh =
        |b: &mut [u8], idx: usize, name: u32, sh_type: u32, flags: u64, off: u64, size: u64| {
            let o = sh_off + idx * 64;
            b[o..o + 4].copy_from_slice(&name.to_le_bytes());
            b[o + 4..o + 8].copy_from_slice(&sh_type.to_le_bytes());
            b[o + 8..o + 16].copy_from_slice(&flags.to_le_bytes());
            b[o + 24..o + 32].copy_from_slice(&off.to_le_bytes());
            b[o + 32..o + 40].copy_from_slice(&size.to_le_bytes());
            if sh_type == 1 {
                b[o + 48..o + 56].copy_from_slice(&4u64.to_le_bytes());
            } else {
                b[o + 48..o + 56].copy_from_slice(&1u64.to_le_bytes());
            }
        };
    write_sh(&mut b, 0, 0, 0, 0, 0, 0);
    write_sh(&mut b, 1, 1, 1, 6, 0x40, 4);
    write_sh(&mut b, 2, 7, 3, 0, 0x48, shstr.len() as u64);
    write_sh(&mut b, 3, 17, 3, 0, 0x60, 1);
    fs::write(path, &b).map_err(|e| err(e.to_string()))?;
    Ok(())
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn compile_aarch64_main_via_stdin(a64_cc: &Path, outo: &Path, work: &Path) -> bool {
    let err_path = work.join("a64-stdin.err");
    let src = "int main(int argc, char **argv) { (void)argc; (void)argv; return 0; }\n";
    let root = cosmo_toolchain_root(a64_cc);
    let a64lib = root.join("aarch64-linux-cosmo/lib");
    let mut tries: Vec<Vec<String>> = vec![
        vec![
            "-x".into(),
            "c".into(),
            "-c".into(),
            "-".into(),
            "-o".into(),
        ],
        vec![
            "-nostdinc".into(),
            "-ffreestanding".into(),
            "-fno-builtin".into(),
            "-x".into(),
            "c".into(),
            "-c".into(),
            "-".into(),
            "-o".into(),
        ],
    ];
    if a64lib.is_dir() {
        tries.insert(
            0,
            vec![
                "-nostdinc".into(),
                "-ffreestanding".into(),
                "-fno-builtin".into(),
                format!("-B{}", a64lib.display()),
                "-x".into(),
                "c".into(),
                "-c".into(),
                "-".into(),
                "-o".into(),
            ],
        );
    }
    for mut args in tries {
        let mut cmd = Command::new(a64_cc);
        // insert -o path at end
        args.push(outo.display().to_string());
        for a in &args {
            cmd.arg(a);
        }
        cmd.stdin(Stdio::piped());
        cmd.stdout(Stdio::null());
        cmd.stderr(Stdio::piped());
        let Ok(mut child) = cmd.spawn() else {
            continue;
        };
        if let Some(mut stdin) = child.stdin.take() {
            let _ = std::io::Write::write_all(&mut stdin, src.as_bytes());
        }
        let Ok(out) = child.wait_with_output() else {
            continue;
        };
        let _ = fs::write(&err_path, &out.stderr);
        if out.status.success() && outo.is_file() {
            if let Ok(file_out) = Command::new("file").arg(outo).output() {
                let s = String::from_utf8_lossy(&file_out.stdout).to_ascii_lowercase();
                if s.contains("x86-64") || s.contains("intel 80386") {
                    continue;
                }
            }
            return true;
        }
    }
    if let Ok(e) = fs::read_to_string(&err_path) {
        if !e.trim().is_empty() {
            eprintln!("cosmo-linker: aarch64 stdin compile failed:\n{e}");
        }
    }
    false
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn build_aarch64_stubs_once(
    cosmo: &Path,
    cc: &Path,
    a64_cc: Option<&Path>,
    work: &Path,
) -> Result<(), String> {
    if A64_STUBS.get().is_some() {
        return Ok(());
    }
    let src_main = work.join("a64-main.c");
    let src_pad = work.join("a64-pad.c");
    let main_o = work.join("a64-main.o");
    let pad_o = work.join("a64-pad.o");
    fs::write(
        &src_main,
        "int main(int argc, char **argv) { (void)argc; (void)argv; return 0; }\n",
    )
    .map_err(|e| err(e.to_string()))?;
    // static: local symbol so the same pad .o can be copied beside every input without multidef.
    fs::write(&src_pad, "static void potctl_cosmo_a64_pad(void) {}\n")
        .map_err(|e| err(e.to_string()))?;

    let prebuilt = find_prebuilt_aarch64_obj(cosmo, cc);
    // Prefer real aarch64-linux-cosmo-gcc when available; else emit minimal ELF companions
    // so cosmocc's aarch64 fat pass always gets a `main` once and pad objects without multidef.
    let mut main_ok = false;
    let mut pad_ok = false;
    if let Some(a64) = a64_cc {
        main_ok = compile_aarch64_c_to_o(a64, &src_main, &main_o, work).is_ok();
        if !main_ok {
            main_ok = compile_aarch64_main_via_stdin(a64, &main_o, work);
        }
        pad_ok = compile_aarch64_c_to_o(a64, &src_pad, &pad_o, work).is_ok();
    }
    if !main_ok {
        write_minimal_aarch64_main_o(&main_o)?;
        main_ok = main_o.is_file();
        trace("using embedded minimal aarch64 main.o companion");
    }
    if !main_ok || !main_o.is_file() {
        return Err(err("cannot produce aarch64 main companion"));
    }

    if !pad_ok {
        if let Some(ref pre) = prebuilt {
            fs::copy(pre, &pad_o).map_err(|e| err(e.to_string()))?;
            pad_ok = true;
        } else if write_minimal_aarch64_pad_o(&pad_o).is_ok() {
            pad_ok = pad_o.is_file();
        }
    }
    if !pad_ok {
        return Err(err("cannot produce aarch64 pad companion"));
    }
    let _ = A64_STUBS.set(Aarch64Stubs { main_o, pad_o });
    Ok(())
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn ensure_aarch64_companion(o: &Path) -> Result<(), String> {
    if !o.is_file() {
        return Ok(());
    }
    let stubs = A64_STUBS
        .get()
        .ok_or_else(|| err("aarch64 stubs not built"))?;
    let sib_dir = o.parent().unwrap_or(Path::new(".")).join(".aarch64");
    fs::create_dir_all(&sib_dir).map_err(|e| err(e.to_string()))?;
    let sib = sib_dir.join(o.file_name().unwrap_or_default());
    let stub = if !A64_MAIN_PLACED.swap(true, Ordering::SeqCst) {
        &stubs.main_o
    } else {
        &stubs.pad_o
    };
    fs::copy(stub, &sib).map_err(|e| err(e.to_string()))?;
    Ok(())
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn resolve_ar(cosmo: &Path, cc: &Path) -> PathBuf {
    if let Some(p) = which("ar") {
        return p;
    }
    let dot = cosmo.join(".cosmocc");
    if let Some(p) = find_exec_named(&dot, "ar") {
        return p;
    }
    let root = cosmo_toolchain_root(cc);
    if let Some(p) = find_exec_named(&root, "ar") {
        return p;
    }
    PathBuf::from("ar")
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn extract_rlib_objs(rlib: &Path, dest: &Path, ar_bin: &Path) -> Vec<PathBuf> {
    let _ = fs::create_dir_all(dest);
    let _ = Command::new(ar_bin)
        .current_dir(dest)
        .arg("x")
        .arg(rlib)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let mut out = Vec::new();
    if let Ok(rd) = fs::read_dir(dest) {
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) == Some("o") && p.is_file() {
                out.push(p);
            }
        }
    }
    out
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn inject_cosmo_lib_paths(cosmo: &Path, cc: &Path, out: &mut Vec<String>) {
    let cc_root = cosmo_toolchain_root(cc);
    let mut add_lib_dir = |dir: &Path| {
        if dir.is_dir() {
            out.push(format!("-B{}", dir.display()));
            out.push(format!("-L{}", dir.display()));
        }
    };
    for d in [
        cc_root.join("x86_64-linux-cosmo/lib"),
        cc_root.join("lib/gcc/x86_64-linux-cosmo/14.1.0"),
        cc_root.join("libexec/gcc/x86_64-linux-cosmo/14.1.0"),
    ] {
        add_lib_dir(&d);
    }
    let dot = cosmo.join(".cosmocc");
    if dot.is_dir() {
        let mut stack = vec![dot];
        let mut n = 0usize;
        while let Some(dir) = stack.pop() {
            if n >= 20 {
                break;
            }
            let Ok(rd) = fs::read_dir(&dir) else {
                continue;
            };
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    stack.push(p);
                } else if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                    if matches!(name, "libc.a" | "crt1.o" | "crt.o" | "crtbeginT.o") {
                        if let Some(parent) = p.parent() {
                            add_lib_dir(parent);
                            n += 1;
                        }
                    }
                }
            }
        }
    }
}

#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn cosmocc_wants_aarch64_companions() -> bool {
    // cosmocc multi-arch fat pass needs aarch64 companions unless restricted to x86_64 only.
    let arches = env::var("COSMOCC_ARCHES").unwrap_or_else(|_| "x86_64".into());
    arches
        .split_whitespace()
        .any(|a| a == "aarch64" || a == "arm64")
        || arches.trim().is_empty()
        || arches == "all"
}

/// rustc nightly `std` process/unix/pidfd calls `waitid`, which Cosmopolitan libc does not
/// export. Provide a weak-style C stub (strong symbol here) that returns `ENOSYS` so the APE
/// links; potctl does not rely on pidfd wait in the cosmo CLI path.
#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
const WAITID_STUB_C: &str = r#"
/* potctl-cosmo-ld: waitid shim for rustc std PidFd (Cosmopolitan has no waitid). */
#include <errno.h>
typedef int idtype_t;
typedef int id_t;
typedef struct { int si_signo; int si_errno; int si_code; } siginfo_t;
int waitid(idtype_t idtype, id_t id, siginfo_t *infop, int options) {
    (void)idtype;
    (void)id;
    (void)infop;
    (void)options;
    errno = ENOSYS;
    return -1;
}
"#;

/// Compile libc compat stubs (waitid, …) with the same cosmo driver rustc uses.
#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
fn compile_libc_compat_stub(cc: &Path, work: &Path) -> Result<PathBuf, String> {
    let src = work.join("cosmo-libc-compat.c");
    let obj = work.join("cosmo-libc-compat.o");
    fs::write(&src, WAITID_STUB_C).map_err(|e| err(e.to_string()))?;
    let status = Command::new(cc)
        .args(["-c", "-o"])
        .arg(&obj)
        .arg(&src)
        .status()
        .map_err(|e| err(format!("compile cosmo-libc-compat: {e}")))?;
    if !status.success() {
        return Err(err(format!(
            "compile cosmo-libc-compat failed (exit {:?})",
            status.code()
        )));
    }
    Ok(obj)
}

/// Entry for `potctl-cosmo-ld` (rustc linker).
#[allow(dead_code, reason = "only potctl-cosmo-ld calls this")]
pub fn link_main(args: &[String]) -> Result<(), String> {
    let cosmo = cosmo_env();
    let cc = resolve_cosmo_cc(&cosmo)?;
    let use_cosmocc = cc_is_cosmocc(&cc);
    let a64_cc = resolve_aarch64_cc(&cosmo, &cc);
    // Always plant aarch64 companions when using cosmocc unless COSMOCC_ARCHES is strictly
    // single-arch x86_64 (CI default). cosmocc 3.x still often runs an aarch64 pass anyway,
    // so default is to prepare companions whenever cosmocc is the driver.
    let need_a64 = use_cosmocc
        && (cosmocc_wants_aarch64_companions()
            || env::var("POTCTL_COSMO_FORCE_A64_COMPANIONS")
                .map(|v| v != "0")
                .unwrap_or(true));

    let work = env::temp_dir().join(format!("potctl-cosmo-rlib-{}", std::process::id()));
    fs::create_dir_all(&work).map_err(|e| err(e.to_string()))?;

    if need_a64 {
        build_aarch64_stubs_once(&cosmo, &cc, a64_cc.as_deref(), &work)?;
    }

    // Always inject waitid (and future libc gaps) before other objects so rustc std links.
    let compat_obj = compile_libc_compat_stub(&cc, &work)?;
    if need_a64 {
        ensure_aarch64_companion(&compat_obj)?;
    }

    let mut prefix_args: Vec<String> = Vec::new();
    if !use_cosmocc {
        inject_cosmo_lib_paths(&cosmo, &cc, &mut prefix_args);
    }
    // Compat objects first (before user/rlib objs and -lc) so missing Cosmo symbols resolve.
    prefix_args.push(compat_obj.to_string_lossy().into_owned());

    let ar_bin = resolve_ar(&cosmo, &cc);
    let mut out_args: Vec<String> = Vec::new();
    let mut rlib_i = 0usize;

    for o in args {
        match o.as_str() {
            "-lunwind" | "-Wl,-Bdynamic" | "-pg" | "-mnop-mcount" | "-nodefaultlibs"
            | "-nostdlib" | "-nostartfiles" => continue,
            s if s.ends_with(".rlib") => {
                rlib_i += 1;
                let rdir = work.join(format!("r{rlib_i}"));
                let objs = extract_rlib_objs(Path::new(s), &rdir, &ar_bin);
                if objs.is_empty() {
                    out_args.push("-Wl,--whole-archive".into());
                    out_args.push(s.to_string());
                    out_args.push("-Wl,--no-whole-archive".into());
                } else {
                    for obj in objs {
                        if need_a64 {
                            ensure_aarch64_companion(&obj)?;
                        }
                        out_args.push(obj.to_string_lossy().into_owned());
                    }
                }
            }
            s if s.ends_with(".o") => {
                let p = Path::new(s);
                if p.is_file() && need_a64 {
                    ensure_aarch64_companion(p)?;
                }
                out_args.push(s.to_string());
            }
            s => out_args.push(s.to_string()),
        }
    }

    trace(&format!(
        "CC={} cosmocc={} need_a64={} AARCH64_CC={} COSMOCC_ARCHES={} rlibs_extracted={} prefix={} nargs={} compat={}",
        cc.display(),
        use_cosmocc as u8,
        need_a64 as u8,
        a64_cc
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "none".into()),
        env::var("COSMOCC_ARCHES").unwrap_or_default(),
        rlib_i,
        prefix_args.len(),
        out_args.len(),
        compat_obj.display(),
    ));

    let mut cmd = Command::new(&cc);
    // Preserve cosmo/cosmocc env from the cargo/rustc invocation.
    for (k, v) in env::vars() {
        if k.starts_with("COSMO") || k == "ARCH" || k == "MODE" {
            cmd.env(k, v);
        }
    }
    for a in &prefix_args {
        cmd.arg(a);
    }
    for a in &out_args {
        cmd.arg(a);
    }
    let output = cmd
        .output()
        .map_err(|e| err(format!("exec {}: {e}", cc.display())))?;
    if !output.stderr.is_empty() {
        let _ = std::io::Write::write_all(&mut std::io::stderr(), &output.stderr);
    }
    if !output.stdout.is_empty() {
        let _ = std::io::Write::write_all(&mut std::io::stdout(), &output.stdout);
    }
    let _ = fs::remove_dir_all(&work);
    if output.status.success() {
        Ok(())
    } else {
        Err(err(format!(
            "cosmo link failed (exit {:?})",
            output.status.code()
        )))
    }
}

// --- build (potctl cosmo build) ----------------------------------------------

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn prepend_path(dir: &Path) {
    let mut paths = vec![dir.to_path_buf()];
    if let Some(bin) = dir.join("bin").is_dir().then(|| dir.join("bin")) {
        paths.push(bin);
    }
    if let Ok(cur) = env::var("PATH") {
        for p in env::split_paths(&cur) {
            paths.push(p);
        }
    }
    let joined = env::join_paths(paths).ok();
    if let Some(j) = joined {
        env::set_var("PATH", j);
    }
}

/// Minimal target-spec fixup without pulling serde_json (keep potctl deps lean).
/// Always writes an absolute path to `potctl-cosmo-ld` so rustc does not depend on PATH.
#[allow(dead_code, reason = "only the potctl bin calls this")]
fn prepare_target_json(src_json: &Path, linker: &Path, dst_json: &Path) -> Result<(), String> {
    let linker_abs = fs::canonicalize(linker).unwrap_or_else(|_| linker.to_path_buf());
    let linker_s = linker_abs
        .display()
        .to_string()
        .replace('\\', "\\\\")
        .replace('"', "\\\"");
    let raw = fs::read_to_string(src_json).map_err(|e| err(e.to_string()))?;
    let obsolete = ["\"allows-weak-linkage\"", "\"is-builtin\""];
    let mut lines: Vec<String> = Vec::new();
    for line in raw.lines() {
        if obsolete.iter().any(|k| line.contains(k)) {
            continue;
        }
        let trimmed = line.trim_start();
        if trimmed.starts_with("\"linker\"") {
            let indent = &line[..line.len() - line.trim_start().len()];
            lines.push(format!("{indent}\"linker\": \"{linker_s}\","));
            continue;
        }
        if trimmed.starts_with("\"target-pointer-width\"") && trimmed.contains("\"64\"") {
            let indent = &line[..line.len() - line.trim_start().len()];
            lines.push(format!("{indent}\"target-pointer-width\": 64,"));
            continue;
        }
        lines.push(line.to_string());
    }
    let mut text = lines.join("\n");
    if !text.ends_with('\n') {
        text.push('\n');
    }
    fs::write(dst_json, &text).map_err(|e| err(e.to_string()))?;
    Ok(())
}

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn find_dbg_artifact(search_roots: &[&Path], arch: &str) -> Option<PathBuf> {
    for root in search_roots {
        if !root.exists() {
            continue;
        }
        let mut stack = vec![root.to_path_buf()];
        while let Some(dir) = stack.pop() {
            let Ok(rd) = fs::read_dir(&dir) else {
                continue;
            };
            for ent in rd.flatten() {
                let p = ent.path();
                if p.is_dir() {
                    // Skip huge irrelevant trees lightly.
                    if p.file_name().and_then(|s| s.to_str()) == Some(".git") {
                        continue;
                    }
                    stack.push(p);
                } else if p.file_name().and_then(|s| s.to_str()) == Some("potctl.com.dbg") {
                    return Some(p);
                }
            }
        }
        // Fallback: any potctl* under cosmo arch target.
        let cosmo_arch = root.join(format!("cosmo-{arch}"));
        if cosmo_arch.is_dir() {
            let mut stack = vec![cosmo_arch];
            while let Some(dir) = stack.pop() {
                let Ok(rd) = fs::read_dir(&dir) else {
                    continue;
                };
                for ent in rd.flatten() {
                    let p = ent.path();
                    if p.is_dir() {
                        stack.push(p);
                    } else if let Some(name) = p.file_name().and_then(|s| s.to_str()) {
                        if name.starts_with("potctl") && p.is_file() {
                            return Some(p);
                        }
                    }
                }
            }
        }
    }
    None
}

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn find_apelink(cosmo: &Path) -> Option<PathBuf> {
    if let Some(p) = which("apelink") {
        return Some(p);
    }
    for c in [
        cosmo.join("o/tool/build/apelink.com"),
        cosmo.join("o/tool/build/apelink"),
    ] {
        if is_exec(&c) {
            return Some(c);
        }
    }
    find_exec_named(cosmo, "apelink")
}

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn find_ape_elf(cosmo: &Path) -> Option<PathBuf> {
    find_path_suffix(cosmo, &["ape", "ape.elf"]).or_else(|| find_first_file_named(cosmo, "ape.elf"))
}

/// `potctl cosmo build` — host potctl orchestrates nightly cargo + cosmo target.
#[allow(dead_code, reason = "only the potctl bin calls this")]
pub fn build_ape(repo_root: &Path) -> Result<(), String> {
    let cosmo = env::var_os("COSMO")
        .map(PathBuf::from)
        .ok_or_else(|| err("set COSMO to Cosmopolitan monorepo root (after make toolchain)"))?;
    if !cosmo.is_dir() {
        return Err(err(format!("COSMO={} is not a directory", cosmo.display())));
    }
    if which("rustup").is_none() {
        return Err(err("rustup required"));
    }
    if which("cargo").is_none() {
        return Err(err("cargo required"));
    }

    let cosmo_dir = repo_root.join("potctl/cosmo");
    let out_dir = env::var_os("POTCTL_COSMO_OUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| cosmo_dir.join("out"));
    let archs = env::var("POTCTL_COSMO_ARCHS").unwrap_or_else(|_| "x86_64".into());
    let nightly = env::var("POTCTL_COSMO_NIGHTLY").unwrap_or_else(|_| "nightly".into());

    fs::create_dir_all(&out_dir).map_err(|e| err(e.to_string()))?;
    prepend_path(&cosmo);

    let linker_bin = env::current_exe().map_err(|e| err(e.to_string()))?;
    // Prefer sibling potctl-cosmo-ld if present (same cargo target dir / install prefix).
    let linker = {
        let sib = linker_bin
            .parent()
            .unwrap_or(Path::new("."))
            .join("potctl-cosmo-ld");
        if is_exec(&sib) {
            sib
        } else if linker_bin
            .file_name()
            .and_then(|s| s.to_str())
            .map(|n| n.contains("potctl-cosmo-ld"))
            .unwrap_or(false)
        {
            linker_bin.clone()
        } else {
            // Build potctl-cosmo-ld on the host first if missing.
            ensure_host_cosmo_ld(repo_root, &linker_bin)?
        }
    };

    let _ = Command::new("rustup")
        .args(["toolchain", "install", &nightly, "--profile", "minimal"])
        .status();
    let _ = Command::new("rustup")
        .args(["component", "add", "rust-src", "--toolchain", &nightly])
        .status();

    let mut dbg_bins: Vec<PathBuf> = Vec::new();

    for arch in archs.split_whitespace() {
        let target_src = match arch {
            "x86_64" => cosmo_dir.join("x86_64-unknown-linux-cosmo.json"),
            "aarch64" => cosmo_dir.join("aarch64-unknown-linux-cosmo.json"),
            other => return Err(err(format!("unknown arch in POTCTL_COSMO_ARCHS: {other}"))),
        };
        if !target_src.is_file() {
            return Err(err(format!("missing {}", target_src.display())));
        }

        let build_tmp = out_dir.join(format!("build-{arch}"));
        let _ = fs::remove_dir_all(&build_tmp);
        fs::create_dir_all(&build_tmp).map_err(|e| err(e.to_string()))?;
        let target_json = build_tmp.join("target.json");
        prepare_target_json(&target_src, &linker, &target_json)?;

        let cargo_cfg = build_tmp.join(".cargo");
        fs::create_dir_all(&cargo_cfg).map_err(|e| err(e.to_string()))?;
        fs::write(
            cargo_cfg.join("config.toml"),
            "[unstable]\njson-target-spec = true\n",
        )
        .map_err(|e| err(e.to_string()))?;

        let target_dir = repo_root.join(format!("target/cosmo-{arch}"));
        // Drop prior cosmocc aarch64 stub dirs that break the fat pass.
        prune_aarch64_dirs(&target_dir);

        env::set_var("COSMO", &cosmo);
        env::set_var("ARCH", arch);
        env::set_var("CARGO_UNSTABLE_JSON_TARGET_SPEC", "true");
        env::set_var(
            "COSMOCC_ARCHES",
            env::var("COSMOCC_ARCHES").unwrap_or_else(|_| archs.clone()),
        );
        env::set_var("CARGO_TARGET_DIR", &target_dir);

        eprintln!("==> cargo +{nightly} build -p potctl (cosmo {arch})");
        // Cosmo APE: only the potctl CLI, without host cosmo-host builder (ci/lockstep only).
        // potctl-cosmo-ld is host-only and must never be built for this target.
        let status = Command::new("cargo")
            .arg(format!("+{nightly}"))
            .args([
                "-Z",
                "json-target-spec",
                "-Z",
                "build-std=core,alloc,std,panic_abort",
                "build",
                "--manifest-path",
            ])
            .arg(repo_root.join("potctl/Cargo.toml"))
            .args([
                "--release",
                "--bin",
                "potctl",
                "--no-default-features",
                "--target",
            ])
            .arg(&target_json)
            .current_dir(&build_tmp)
            .env("COSMO", &cosmo)
            .env("ARCH", arch)
            .env("CARGO_UNSTABLE_JSON_TARGET_SPEC", "true")
            .env(
                "COSMOCC_ARCHES",
                env::var("COSMOCC_ARCHES").unwrap_or_else(|_| archs.clone()),
            )
            .env("CARGO_TARGET_DIR", &target_dir)
            // Ensure linker is on PATH when target JSON uses bare name "potctl-cosmo-ld".
            .env("PATH", {
                let mut paths = vec![linker.parent().unwrap_or(Path::new(".")).to_path_buf()];
                if let Ok(cur) = env::var("PATH") {
                    for p in env::split_paths(&cur) {
                        paths.push(p);
                    }
                }
                env::join_paths(paths).unwrap_or_default()
            })
            .status()
            .map_err(|e| err(format!("cargo: {e}")))?;
        if !status.success() {
            return Err(err(format!(
                "cosmo cargo build failed for arch={arch} (see potctl/cosmo/README.md)"
            )));
        }

        let dbg = find_dbg_artifact(
            &[
                target_dir.as_path(),
                repo_root.join("target").as_path(),
                build_tmp.as_path(),
            ],
            arch,
        )
        .ok_or_else(|| {
            err(format!(
                "could not find potctl.com.dbg after cosmo build (arch={arch})"
            ))
        })?;
        let staged = out_dir.join(format!("potctl-{arch}.com.dbg"));
        fs::copy(&dbg, &staged).map_err(|e| err(e.to_string()))?;
        eprintln!("    dbg: {}", staged.display());
        dbg_bins.push(staged);
    }

    let ape_out = out_dir.join("potctl.com");
    let skip_apelink = env::var("POTCTL_COSMO_SKIP_APELINK").ok().as_deref() == Some("1");
    if skip_apelink || dbg_bins.is_empty() {
        fs::copy(&dbg_bins[0], &ape_out).map_err(|e| err(e.to_string()))?;
        eprintln!("skip apelink; copied first dbg -> {}", ape_out.display());
    } else {
        let mut linked = false;
        if let (Some(apelink), Some(ape_elf)) = (find_apelink(&cosmo), find_ape_elf(&cosmo)) {
            eprintln!(
                "==> apelink ({}) -> {}",
                apelink.display(),
                ape_out.display()
            );
            let mut cmd = Command::new(&apelink);
            cmd.arg("-l").arg(&ape_elf).arg("-o").arg(&ape_out);
            for d in &dbg_bins {
                cmd.arg(d);
            }
            if cmd.status().map(|s| s.success()).unwrap_or(false) {
                linked = true;
            } else {
                let mut cmd2 = Command::new(&apelink);
                cmd2.arg("-o").arg(&ape_out).arg("-l").arg(&ape_elf);
                for d in &dbg_bins {
                    cmd2.arg(d);
                }
                linked = cmd2.status().map(|s| s.success()).unwrap_or(false);
            }
        }
        if !linked {
            eprintln!(
                "warn: apelink skipped/failed; using cosmo dbg as potctl.com (typical for cosmo .com.dbg)"
            );
            fs::copy(&dbg_bins[0], &ape_out).map_err(|e| err(e.to_string()))?;
        }
    }

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Ok(m) = fs::metadata(&ape_out) {
            let mut perms = m.permissions();
            perms.set_mode(perms.mode() | 0o111);
            let _ = fs::set_permissions(&ape_out, perms);
        }
    }

    eprintln!("==> built {}", ape_out.display());
    let _ = Command::new("file").arg(&ape_out).status();
    let _ = Command::new("ls").args(["-la"]).arg(&ape_out).status();

    if env::var("POTCTL_COSMO_SMOKE")
        .unwrap_or_else(|_| "1".into())
        .as_str()
        == "1"
    {
        let mut child = Command::new(&ape_out)
            .arg("--help")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .ok();
        if let Some(ref mut c) = child {
            let _ = c.wait();
        }
    }
    Ok(())
}

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn prune_aarch64_dirs(root: &Path) {
    if !root.is_dir() {
        return;
    }
    let mut stack = vec![root.to_path_buf()];
    let mut to_rm = Vec::new();
    while let Some(dir) = stack.pop() {
        let Ok(rd) = fs::read_dir(&dir) else {
            continue;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                if p.file_name().and_then(|s| s.to_str()) == Some(".aarch64") {
                    to_rm.push(p);
                } else {
                    stack.push(p);
                }
            }
        }
    }
    for p in to_rm {
        let _ = fs::remove_dir_all(p);
    }
}

#[allow(dead_code, reason = "only the potctl bin calls this")]
fn ensure_host_cosmo_ld(repo_root: &Path, potctl_exe: &Path) -> Result<PathBuf, String> {
    let sib = potctl_exe
        .parent()
        .unwrap_or(Path::new("."))
        .join("potctl-cosmo-ld");
    if is_exec(&sib) {
        return Ok(sib);
    }
    eprintln!("==> building host potctl-cosmo-ld (rustc cosmo linker)");
    let status = Command::new("cargo")
        .args([
            "build",
            "--release",
            "-p",
            "potctl",
            "--bin",
            "potctl-cosmo-ld",
        ])
        .current_dir(repo_root)
        .status()
        .map_err(|e| err(format!("cargo build potctl-cosmo-ld: {e}")))?;
    if !status.success() {
        return Err(err("failed to build potctl-cosmo-ld on host"));
    }
    let candidates = [
        repo_root.join("target/release/potctl-cosmo-ld"),
        repo_root.join("potctl/target/release/potctl-cosmo-ld"),
    ];
    for c in &candidates {
        if is_exec(c) {
            return Ok(c.clone());
        }
    }
    if is_exec(&sib) {
        return Ok(sib);
    }
    Err(err(
        "potctl-cosmo-ld not found after host build; run: cargo build --release -p potctl --bin potctl-cosmo-ld",
    ))
}
