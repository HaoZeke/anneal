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

    // pyo3 is built with `extension-module`, so its build script does not
    // link libpython. A test binary is not loaded by a Python interpreter.
    // On Linux the link fails with undefined Python symbols. On macOS,
    // allowing those symbols to stay open lets the link finish, and the
    // loader then stops the process because they are missing. Link the
    // interpreter for every artifact of this package except a Maturin
    // wheel. Maturin sets PYO3_BUILD_EXTENSION_MODULE, and this script
    // then leaves libpython out so the extension stays unpinned.
    link_libpython_for_rust_executables();
}

fn link_libpython_for_rust_executables() {
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    let os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if os != "linux" && os != "macos" {
        return;
    }
    if std::env::var("CARGO_FEATURE_PYTHON").is_err() {
        return;
    }
    if std::env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some() {
        return;
    }

    let python = python_interpreter();
    let script = "\
import sysconfig
libdir = sysconfig.get_config_var('LIBDIR') or ''
libpl = sysconfig.get_config_var('LIBPL') or ''
ldversion = sysconfig.get_config_var('LDVERSION') or ''
framework = sysconfig.get_config_var('PYTHONFRAMEWORKPREFIX') or ''
shared = sysconfig.get_config_var('Py_ENABLE_SHARED')
pyframework = sysconfig.get_config_var('PYTHONFRAMEWORK')

def on(value):
    return value not in (None, 0, '0', '')

print(libdir)
print(libpl)
print(ldversion)
print(framework)
print('1' if on(shared) or on(pyframework) else '0')
";
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "feature `python` requires `{python}` on PATH so Rust executables can link libpython: {err}"
            )
        });
    if !output.status.success() {
        panic!(
            "feature `python` could not read LIBDIR from `{python}`: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8(output.stdout).expect("python sysconfig stdout is utf-8");
    let mut lines = text.lines();
    let libdir = lines.next().unwrap_or("");
    let libpl = lines.next().unwrap_or("");
    let ldversion = lines.next().unwrap_or("");
    let framework = lines.next().unwrap_or("");
    let shared = lines.next().unwrap_or("0") == "1";
    if ldversion.is_empty() || (libdir.is_empty() && libpl.is_empty()) {
        panic!("feature `python` needs LDVERSION and LIBDIR or LIBPL from `{python}`");
    }
    if !libdir.is_empty() {
        println!("cargo:rustc-link-search=native={libdir}");
    }
    if !libpl.is_empty() && libpl != libdir {
        println!("cargo:rustc-link-search=native={libpl}");
    }
    let lib_name = format!("python{ldversion}");
    // `rustc-link-lib` is what the lib test harness actually consumes.
    // A per-kind link argument for tests does not reach that harness.
    if shared {
        println!("cargo:rustc-link-lib=dylib={lib_name}");
    } else {
        println!("cargo:rustc-link-lib=static={lib_name}");
    }
    // python-build-standalone, which the test jobs install, records an rpath
    // inside the interpreter. The test harness is not next to that library.
    if !libdir.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    }
    if !framework.is_empty() {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{framework}");
    }
}

fn python_interpreter() -> String {
    if let Ok(python) = std::env::var("PYO3_PYTHON")
        && !python.is_empty()
    {
        return python;
    }
    for candidate in ["python3", "python"] {
        let found = std::process::Command::new(candidate)
            .arg("-c")
            .arg("import sys")
            .output()
            .map(|output| output.status.success())
            .unwrap_or(false);
        if found {
            return candidate.to_string();
        }
    }
    panic!(
        "feature `python` requires PYO3_PYTHON, python3, or python on PATH so Rust executables can link libpython"
    );
}
