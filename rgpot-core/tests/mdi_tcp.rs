// MIT License
// Copyright 2023--present rgpot developers

#![cfg(all(feature = "mdi", unix))]

use std::ffi::c_void;
use std::io::{BufRead, BufReader, Read};
use std::net::TcpListener;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rgpot_core::c_api::potential::{rgpot_potential_free, rgpot_potential_new};
use rgpot_core::mdi::{serve_mdi, MdiEngineConfig};
use rgpot_core::status::rgpot_status_t;
use rgpot_core::tensor::rgpot_tensor_owned_cpu_f64_2d;
use rgpot_core::types::{rgpot_force_input_t, rgpot_force_out_t};
use rgpot_core::units::{BOHR_TO_ANGSTROM, EV_TO_HARTREE};

struct Probe {
    calls: AtomicUsize,
    geometry_ok: AtomicBool,
    scale: f64,
    fail: bool,
    malformed: bool,
    nonfinite_energy: bool,
}

unsafe extern "C" fn probe_potential(
    user_data: *mut c_void,
    input: *const rgpot_force_input_t,
    output: *mut rgpot_force_out_t,
) -> rgpot_status_t {
    let probe = unsafe { &*(user_data as *const Probe) };
    probe.calls.fetch_add(1, Ordering::SeqCst);
    if probe.fail {
        return rgpot_status_t::RGPOT_INVALID_PARAMETER;
    }
    let input = unsafe { &*input };
    let positions =
        unsafe { std::slice::from_raw_parts((*input.positions).dl_tensor.data as *const f64, 6) };
    let cell =
        unsafe { std::slice::from_raw_parts((*input.box_matrix).dl_tensor.data as *const f64, 9) };
    let call = probe.calls.load(Ordering::SeqCst) - 1;
    let expected_distance = if call == 0 {
        2.0 * BOHR_TO_ANGSTROM
    } else {
        3.0 * BOHR_TO_ANGSTROM
    };
    let tolerance = f64::EPSILON * 64.0;
    if (positions[3] - expected_distance).abs() > tolerance
        || (cell[0] - 10.0 * BOHR_TO_ANGSTROM).abs() > tolerance
    {
        probe.geometry_ok.store(false, Ordering::SeqCst);
    }
    let native_force = probe.scale / (EV_TO_HARTREE * BOHR_TO_ANGSTROM);
    let forces = [native_force; 6];
    let output = unsafe { &mut *output };
    output.energy = if probe.nonfinite_energy {
        f64::NAN
    } else {
        probe.scale / EV_TO_HARTREE
    };
    output.variance = 0.0;
    let (rows, columns) = if probe.malformed { (3, 2) } else { (2, 3) };
    output.forces = unsafe { rgpot_tensor_owned_cpu_f64_2d(forces.as_ptr(), rows, columns) };
    rgpot_status_t::RGPOT_SUCCESS
}

fn mdi_library() -> Option<String> {
    let output = Command::new("python3")
        .args([
            "-c",
            "import pathlib, mdi; root = pathlib.Path(mdi.__file__).parent; print(next(path for name in ('libmdi.so', 'libmdi.dylib', 'mdi.dll') if (path := root / name).is_file()))",
        ])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8(output.stdout)
        .ok()
        .map(|path| path.trim().to_owned())
        .filter(|path| !path.is_empty())
}

fn server_child(scenario: &str) {
    let port = std::env::var("RGPOT_MDI_TEST_PORT").expect("test port");
    let library = std::env::var("RGPOT_MDI_TEST_LIBRARY").expect("MDI library");
    let scale = if scenario == "second" { 2.0 } else { 1.0 };
    let probe = Probe {
        calls: AtomicUsize::new(0),
        geometry_ok: AtomicBool::new(true),
        scale,
        fail: scenario == "callback",
        malformed: scenario == "malformed",
        nonfinite_energy: scenario == "energy_nan",
    };
    let pot = unsafe {
        rgpot_potential_new(
            probe_potential,
            (&probe as *const Probe).cast_mut().cast::<c_void>(),
            None,
        )
    };
    let config = MdiEngineConfig::default()
        .with_options(format!(
            "-role ENGINE -name RGPOT -method TCP -hostname 127.0.0.1 -port {port}"
        ))
        .with_library_path(library);
    let result = unsafe { serve_mdi(pot, &config) };
    unsafe { rgpot_potential_free(pot) };
    if matches!(
        scenario,
        "missing" | "zero" | "nan" | "callback" | "malformed" | "energy_nan"
    ) {
        assert!(result.is_err(), "invalid MDI exchange must fail");
        let expected_calls =
            usize::from(matches!(scenario, "callback" | "malformed" | "energy_nan"));
        assert_eq!(probe.calls.load(Ordering::SeqCst), expected_calls);
    } else {
        assert_eq!(result.expect("serve MDI driver"), 2);
        assert_eq!(probe.calls.load(Ordering::SeqCst), 2);
        assert!(probe.geometry_ok.load(Ordering::SeqCst));
    }
}

fn driver_source() -> &'static str {
    r#"
import math
import sys
import mdi

port = int(sys.argv[1])
scenario = sys.argv[2]
scale = float(sys.argv[3])
mdi.MDI_Init(f"-role DRIVER -name TEST -method TCP -hostname 127.0.0.1 -port {port}")
print("LISTENING", flush=True)
comm = mdi.MDI_Accept_Communicator()

if scenario == "missing":
    mdi.MDI_Send_Command("<ENERGY", comm)
    try:
        mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
    except Exception:
        pass
    raise SystemExit(0)
if scenario == "zero":
    mdi.MDI_Send_Command(">NATOMS", comm)
    mdi.MDI_Send(0, 1, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command("<NATOMS", comm)
    try:
        mdi.MDI_Recv(1, mdi.MDI_INT, comm)
    except Exception:
        pass
    raise SystemExit(0)

if scenario == "nan":
    mdi.MDI_Send_Command(">NATOMS", comm)
    mdi.MDI_Send(2, 1, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command(">TYPES", comm)
    mdi.MDI_Send([0, 0], 2, mdi.MDI_INT, comm)
    mdi.MDI_Send_Command(">COORDS", comm)
    mdi.MDI_Send([0.0, 0.0, 0.0, float("nan"), 0.0, 0.0], 6, mdi.MDI_DOUBLE, comm)
    mdi.MDI_Send_Command("<ENERGY", comm)
    try:
        mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
    except Exception:
        pass
    raise SystemExit(0)

commands = (
    ">NATOMS", "<NATOMS", ">COORDS", "<COORDS", ">CELL", "<CELL",
    ">ELEMENTS", ">TYPES", "<ENERGY", "<FORCES", "<PE", "EXIT",
)
for command in commands:
    assert mdi.MDI_Check_Command_Exists("@DEFAULT", command, comm) == 1

mdi.MDI_Send_Command(">NATOMS", comm)
mdi.MDI_Send(2, 1, mdi.MDI_INT, comm)
mdi.MDI_Send_Command("<NATOMS", comm)
assert mdi.MDI_Recv(1, mdi.MDI_INT, comm) == 2
mdi.MDI_Send_Command(">ELEMENTS", comm)
mdi.MDI_Send([18, 18], 2, mdi.MDI_INT, comm)
mdi.MDI_Send_Command(">TYPES", comm)
mdi.MDI_Send([0, 0], 2, mdi.MDI_INT, comm)
coords = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0]
mdi.MDI_Send_Command(">COORDS", comm)
mdi.MDI_Send(coords, 6, mdi.MDI_DOUBLE, comm)
mdi.MDI_Send_Command("<COORDS", comm)
assert list(mdi.MDI_Recv(6, mdi.MDI_DOUBLE, comm)) == coords
cell = [10.0, 0.0, 0.0, 0.0, 11.0, 0.0, 0.0, 0.0, 12.0]
mdi.MDI_Send_Command(">CELL", comm)
mdi.MDI_Send(cell, 9, mdi.MDI_DOUBLE, comm)
mdi.MDI_Send_Command("<CELL", comm)
assert list(mdi.MDI_Recv(9, mdi.MDI_DOUBLE, comm)) == cell
if scenario in ("callback", "malformed", "energy_nan"):
    mdi.MDI_Send_Command("<ENERGY", comm)
    try:
        mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
    except Exception:
        pass
    raise SystemExit(0)
mdi.MDI_Send_Command("<ENERGY", comm)
energy = float(mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm))
mdi.MDI_Send_Command("<FORCES", comm)
forces = list(mdi.MDI_Recv(6, mdi.MDI_DOUBLE, comm))
mdi.MDI_Send_Command("<PE", comm)
pe = float(mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm))
tolerance = sys.float_info.epsilon * 64.0
assert math.isclose(energy, scale, rel_tol=tolerance, abs_tol=tolerance)
assert math.isclose(pe, scale, rel_tol=tolerance, abs_tol=tolerance)
assert all(math.isclose(value, scale, rel_tol=tolerance, abs_tol=tolerance) for value in forces)
coords[3] = 3.0
mdi.MDI_Send_Command(">COORDS", comm)
mdi.MDI_Send(coords, 6, mdi.MDI_DOUBLE, comm)
mdi.MDI_Send_Command("<FORCES", comm)
mdi.MDI_Recv(6, mdi.MDI_DOUBLE, comm)
mdi.MDI_Send_Command("<ENERGY", comm)
mdi.MDI_Recv(1, mdi.MDI_DOUBLE, comm)
mdi.MDI_Send_Command("EXIT", comm)
"#
}

fn run_scenario(scenario: &str, scale: f64, library: &str) {
    let listener = TcpListener::bind(("127.0.0.1", 0)).expect("reserve TCP port");
    let port = listener.local_addr().expect("local address").port();
    drop(listener);
    let temp = tempfile::tempdir().expect("temporary driver directory");
    let driver_path = temp.path().join("driver.py");
    std::fs::write(&driver_path, driver_source()).expect("write MDI driver");
    let mut driver = Command::new("python3")
        .arg(&driver_path)
        .arg(port.to_string())
        .arg(scenario)
        .arg(scale.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start MDI driver");
    let stdout = driver.stdout.take().expect("driver stdout");
    let mut reader = BufReader::new(stdout);
    let mut line = String::new();
    reader.read_line(&mut line).expect("driver readiness");
    assert_eq!(line.trim(), "LISTENING");

    let status = Command::new(std::env::current_exe().expect("test executable"))
        .args(["--exact", "mdi_tcp_round_trip", "--nocapture"])
        .env("RGPOT_MDI_TEST_SCENARIO", scenario)
        .env("RGPOT_MDI_TEST_PORT", port.to_string())
        .env("RGPOT_MDI_TEST_LIBRARY", library)
        .status()
        .expect("start MDI engine child");
    assert!(status.success(), "MDI engine child failed for {scenario}");
    let mut remaining = String::new();
    reader
        .read_to_string(&mut remaining)
        .expect("read driver stdout");
    let output = driver.wait_with_output().expect("wait for MDI driver");
    assert!(
        output.status.success(),
        "MDI driver failed for {scenario}: {}{remaining}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn mdi_tcp_round_trip() {
    if let Ok(scenario) = std::env::var("RGPOT_MDI_TEST_SCENARIO") {
        server_child(&scenario);
        return;
    }
    let Some(library) = mdi_library() else {
        eprintln!("skipping MDI TCP test: Python mdi package is unavailable");
        return;
    };
    run_scenario("first", 1.0, &library);
    run_scenario("second", 2.0, &library);
    for scenario in [
        "missing",
        "zero",
        "nan",
        "callback",
        "malformed",
        "energy_nan",
    ] {
        run_scenario(scenario, 1.0, &library);
    }
}
