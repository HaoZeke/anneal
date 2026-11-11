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

    // `pyo3` is built with `extension-module` (directly, and again through
    // `eindir-core/py-objective`), so it does not link libpython. A test
    // binary is not loaded by a Python interpreter. On macOS, `-undefined
    // dynamic_lookup` lets that binary link, then dyld fails at process
    // start with `_PyBaseObject_Type` missing in the flat namespace.
    // Link the interpreter that `cargo test` will run against. Maturin sets
    // `PYO3_BUILD_EXTENSION_MODULE` and passes the lookup flag itself; linking
    // libpython there would pin the extension to the build interpreter.
    link_libpython_for_macos_tests();
}

fn link_libpython_for_macos_tests() {
    println!("cargo:rerun-if-env-changed=PYO3_BUILD_EXTENSION_MODULE");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    if std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() != Some("macos") {
        return;
    }
    if std::env::var("CARGO_FEATURE_PYTHON").is_err() {
        return;
    }
    if std::env::var_os("PYO3_BUILD_EXTENSION_MODULE").is_some() {
        return;
    }

    let python = python_interpreter();
    let script = r#"
import sysconfig
libdir = sysconfig.get_config_var("LIBDIR") or ""
ldversion = sysconfig.get_config_var("LDVERSION") or ""
framework = sysconfig.get_config_var("PYTHONFRAMEWORKPREFIX") or ""
shared = sysconfig.get_config_var("Py_ENABLE_SHARED")
pyframework = sysconfig.get_config_var("PYTHONFRAMEWORK")

def on(value):
    return value not in (None, 0, "0", "")

print(libdir)
print(ldversion)
print(framework)
print("1" if on(shared) or on(pyframework) else "0")
"#;
    let output = std::process::Command::new(&python)
        .arg("-c")
        .arg(script)
        .output()
        .unwrap_or_else(|err| {
            panic!(
                "feature `python` on macOS requires `{python}` on PATH so the test binaries can link libpython: {err}"
            )
        });
    if !output.status.success() {
        panic!(
            "feature `python` on macOS could not read LIBDIR from `{python}`: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    let text = String::from_utf8(output.stdout).expect("python sysconfig stdout is utf-8");
    let mut lines = text.lines();
    let libdir = lines.next().unwrap_or("");
    let ldversion = lines.next().unwrap_or("");
    let framework = lines.next().unwrap_or("");
    let shared = lines.next().unwrap_or("0") == "1";
    if libdir.is_empty() || ldversion.is_empty() {
        panic!("feature `python` on macOS needs LIBDIR and LDVERSION from `{python}`");
    }
    println!("cargo:rustc-link-search=native={libdir}");
    let lib_name = format!("python{ldversion}");
    if shared {
        println!("cargo:rustc-link-lib=dylib={lib_name}");
    } else {
        println!("cargo:rustc-link-lib=static={lib_name}");
    }
    // python-build-standalone, which GitHub's setup-python installs, records
    // `@rpath/libpythonX.Y.dylib`. The test harness is not next to that
    // library, so the rpath has to name LIBDIR.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
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
        "feature `python` on macOS requires PYO3_PYTHON, python3, or python on PATH so the test binaries can link libpython"
    );
}
