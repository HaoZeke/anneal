//! Short molecular-dynamics segments through external engines.
//!
//! Diversity that the sampler's own move set cannot produce comes from
//! dynamics: a burst of thermostatted MD decorrelates a configuration
//! along the landscape's own soft modes, crossing the low barriers
//! first, which is the physics an isotropic proposal ignores. In the
//! bridge layer the segments are not merely diversity: umbrella and
//! forward-flux constructions define their crossing statistics over
//! dynamical trajectories, so an MD segment propagator is what makes
//! the recorded flux and the committor surrogate dynamical objects
//! rather than artifacts of a proposal family.
//!
//! The contract is propagate-and-report: an engine takes flattened
//! coordinates, a step count, a reduced temperature, and a seed, and
//! returns the final coordinates. Every MD step is a force evaluation
//! and must be charged by the caller; nothing here touches any ledger.
//!
//! LAMMPS is the first-class engine, driven in process through its
//! documented C API (`liblammps`): one persistent instance per engine,
//! coordinates scattered and gathered in memory, no input files and no
//! subprocess. Enabled by the `lammps` feature, which links against
//! `LAMMPS_LIB_DIR`. GROMACS has no embeddable C API (its gmxapi
//! targets workflow orchestration, not in-process propagation), so the
//! GROMACS engine drives the `gmx` binary as a subprocess with the
//! reduced system mapped onto its unit system: one length unit to one
//! nanometer, one energy unit to one kilojoule per mole, one mass unit
//! to one atomic mass unit, under which the reduced temperature maps
//! through the gas constant to `T[K] = T* / 0.00831446` and the
//! reduced time step carries over numerically in picoseconds.

use ndarray::{Array1, ArrayView1};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Failure of an external MD segment.
#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    /// The engine input could not be written or output read.
    #[error("engine io: {0}")]
    Io(#[from] std::io::Error),
    /// The engine reported an error or exited abnormally.
    #[error("engine run failed: {0}")]
    Run(String),
    /// The engine output did not parse back to the input's shape.
    #[error("engine output malformed: {0}")]
    Output(String),
}

/// Propagate-and-report: the whole engine contract.
pub trait SegmentEngine {
    /// Engine name for traces and manifests.
    fn name(&self) -> &'static str;
    /// Advance `x` (flattened 3N coordinates, reduced units) by `steps`
    /// MD steps at reduced temperature `temperature`, deterministic in
    /// `seed`, returning the final coordinates.
    fn propagate(
        &self,
        x: ArrayView1<f64>,
        steps: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Array1<f64>, EngineError>;
}

fn bounding_box(x: ArrayView1<f64>, margin: f64) -> ([f64; 3], [f64; 3]) {
    let mut low = [f64::INFINITY; 3];
    let mut high = [f64::NEG_INFINITY; 3];
    for atom in x.as_slice().expect("coordinates are contiguous").chunks(3) {
        for axis in 0..3 {
            low[axis] = low[axis].min(atom[axis]);
            high[axis] = high[axis].max(atom[axis]);
        }
    }
    for axis in 0..3 {
        low[axis] -= margin;
        high[axis] += margin;
    }
    (low, high)
}

/// Positive 31-bit seeds on distinct streams, as LAMMPS and GROMACS
/// random-number commands require.
fn derived_seed(seed: u64, stream: u64) -> u64 {
    ((seed ^ stream.wrapping_mul(0x9e37_79b9_7f4a_7c15)) % 899_999_999).max(1)
}

#[cfg(feature = "lammps")]
mod lammps_ffi {
    use super::{EngineError, SegmentEngine, derived_seed};
    use ndarray::{Array1, ArrayView1};
    use std::cell::RefCell;
    use std::ffi::{CString, c_char, c_double, c_int, c_void};

    unsafe extern "C" {
        fn lammps_open_no_mpi(
            argc: c_int,
            argv: *mut *mut c_char,
            ptr: *mut *mut c_void,
        ) -> *mut c_void;
        fn lammps_close(handle: *mut c_void);
        fn lammps_command(handle: *mut c_void, cmd: *const c_char) -> *mut c_char;
        fn lammps_scatter_atoms(
            handle: *mut c_void,
            name: *const c_char,
            dtype: c_int,
            count: c_int,
            data: *mut c_void,
        );
        fn lammps_gather_atoms(
            handle: *mut c_void,
            name: *const c_char,
            dtype: c_int,
            count: c_int,
            data: *mut c_void,
        );
        fn lammps_has_error(handle: *mut c_void) -> c_int;
        fn lammps_get_last_error_message(
            handle: *mut c_void,
            buffer: *mut c_char,
            buffer_size: c_int,
        ) -> c_int;
    }

    /// A slurm allocation's PMI environment makes an embedded MPI
    /// singleton try to rendezvous with a launcher that never launched
    /// it. Scrubbing the variables before the first `MPI_Init` lets
    /// the library initialize as the genuine singleton it is. Process
    /// global, done once, before any instance exists.
    fn scrub_launcher_environment() {
        let doomed: Vec<String> = std::env::vars()
            .map(|(key, _)| key)
            .filter(|key| {
                key.starts_with("SLURM_") || key.starts_with("PMI_") || key.starts_with("PMIX_")
            })
            .collect();
        for key in doomed {
            // Single-threaded engine construction; no reader races.
            unsafe { std::env::remove_var(key) };
        }
    }

    struct Instance {
        handle: *mut c_void,
        atoms: usize,
        segment: u64,
    }

    /// LAMMPS driven in process through its C API, reduced
    /// Lennard-Jones units, one persistent instance reused across
    /// segments.
    pub struct LammpsLj {
        instance: RefCell<Option<Instance>>,
    }

    // The engine is moved between threads but never shared: RefCell
    // guards the single-owner discipline the trait implies.
    unsafe impl Send for LammpsLj {}

    impl LammpsLj {
        /// An engine with no live instance; the instance boots on the
        /// first segment, sized to that segment's atom count.
        pub fn new() -> Self {
            Self {
                instance: RefCell::new(None),
            }
        }
    }

    impl Default for LammpsLj {
        fn default() -> Self {
            Self::new()
        }
    }

    impl Drop for LammpsLj {
        fn drop(&mut self) {
            if let Some(instance) = self.instance.borrow_mut().take() {
                unsafe { lammps_close(instance.handle) };
            }
        }
    }

    fn run_command(handle: *mut c_void, command: &str) -> Result<(), EngineError> {
        let text = CString::new(command).expect("commands carry no interior nul");
        unsafe { lammps_command(handle, text.as_ptr()) };
        if unsafe { lammps_has_error(handle) } != 0 {
            let mut buffer = vec![0u8; 512];
            unsafe {
                lammps_get_last_error_message(
                    handle,
                    buffer.as_mut_ptr().cast(),
                    buffer.len() as c_int,
                )
            };
            let end = buffer.iter().position(|&b| b == 0).unwrap_or(buffer.len());
            return Err(EngineError::Run(format!(
                "{command}: {}",
                String::from_utf8_lossy(&buffer[..end])
            )));
        }
        Ok(())
    }

    fn boot(atoms: usize) -> Result<Instance, EngineError> {
        scrub_launcher_environment();
        let args: Vec<CString> = ["anneal", "-log", "none", "-screen", "none", "-nocite"]
            .iter()
            .map(|s| CString::new(*s).expect("static strings carry no nul"))
            .collect();
        let mut argv: Vec<*mut c_char> = args.iter().map(|s| s.as_ptr().cast_mut()).collect();
        let handle = unsafe {
            lammps_open_no_mpi(argv.len() as c_int, argv.as_mut_ptr(), std::ptr::null_mut())
        };
        if handle.is_null() {
            return Err(EngineError::Run("lammps_open_no_mpi returned null".into()));
        }
        for command in [
            "units lj".to_owned(),
            "dimension 3".to_owned(),
            "boundary f f f".to_owned(),
            "atom_style atomic".to_owned(),
            "atom_modify map yes".to_owned(),
            "region simbox block -1000 1000 -1000 1000 -1000 1000".to_owned(),
            "create_box 1 simbox".to_owned(),
            format!("create_atoms 1 random {atoms} 12345 simbox"),
            "mass 1 1.0".to_owned(),
            "pair_style lj/cut 5.0".to_owned(),
            "pair_coeff 1 1 1.0 1.0".to_owned(),
            "neigh_modify every 1 delay 0 check yes".to_owned(),
            "timestep 0.005".to_owned(),
            "fix integrate all nve".to_owned(),
        ] {
            run_command(handle, &command)?;
        }
        Ok(Instance {
            handle,
            atoms,
            segment: 0,
        })
    }

    impl SegmentEngine for LammpsLj {
        fn name(&self) -> &'static str {
            "lammps"
        }

        fn propagate(
            &self,
            x: ArrayView1<f64>,
            steps: usize,
            temperature: f64,
            seed: u64,
        ) -> Result<Array1<f64>, EngineError> {
            let atoms = x.len() / 3;
            let mut slot = self.instance.borrow_mut();
            if slot
                .as_ref()
                .is_some_and(|instance| instance.atoms != atoms)
            {
                let stale = slot.take().expect("occupied slot just observed");
                unsafe { lammps_close(stale.handle) };
            }
            if slot.is_none() {
                *slot = Some(boot(atoms)?);
            }
            let instance = slot.as_mut().expect("instance booted above");
            instance.segment += 1;
            let mut coordinates = x.to_vec();
            let name = CString::new("x").expect("static name");
            unsafe {
                lammps_scatter_atoms(
                    instance.handle,
                    name.as_ptr(),
                    1,
                    3,
                    coordinates.as_mut_ptr().cast(),
                )
            };
            let velocity_seed = derived_seed(seed, instance.segment);
            let thermostat_seed = derived_seed(seed, instance.segment.wrapping_add(1));
            run_command(
                instance.handle,
                &format!("velocity all create {temperature} {velocity_seed} dist gaussian"),
            )?;
            run_command(
                instance.handle,
                &format!("fix lang all langevin {temperature} {temperature} 0.5 {thermostat_seed}"),
            )?;
            run_command(instance.handle, &format!("run {steps}"))?;
            run_command(instance.handle, "unfix lang")?;
            let mut out = vec![0.0_f64; atoms * 3];
            unsafe {
                lammps_gather_atoms(
                    instance.handle,
                    name.as_ptr(),
                    1,
                    3,
                    out.as_mut_ptr().cast::<c_double>().cast(),
                )
            };
            if out.iter().any(|value| !value.is_finite()) {
                return Err(EngineError::Output("non-finite coordinates".into()));
            }
            Ok(Array1::from(out))
        }
    }
}

#[cfg(feature = "lammps")]
pub use lammps_ffi::LammpsLj;

const GROMACS_KELVIN_PER_REDUCED: f64 = 1.0 / 0.008_314_46;

/// The gro configuration for a one-species reduced cluster, shifted so
/// the cluster sits inside a box wide enough for the cutoff.
pub fn gromacs_gro(x: ArrayView1<f64>) -> String {
    let n = x.len() / 3;
    let (low, high) = bounding_box(x, 11.0);
    let box_edge = (high[0] - low[0])
        .max(high[1] - low[1])
        .max(high[2] - low[2]);
    let mut out = String::new();
    out.push_str("LJ cluster segment\n");
    out.push_str(&format!("{n}\n"));
    for (index, atom) in x
        .as_slice()
        .expect("coordinates are contiguous")
        .chunks(3)
        .enumerate()
    {
        // Fixed gro columns: resid, resname, atomname, atom number,
        // then three positions at 8.3 width.
        out.push_str(&format!(
            "{:>5}{:<5}{:>5}{:>5}{:8.3}{:8.3}{:8.3}\n",
            index + 1,
            "CLU",
            "LJ",
            (index + 1) % 100_000,
            atom[0] - low[0],
            atom[1] - low[1],
            atom[2] - low[2]
        ));
    }
    out.push_str(&format!("{box_edge:.4} {box_edge:.4} {box_edge:.4}\n"));
    out
}

/// The reduced Lennard-Jones topology in GROMACS units.
pub fn gromacs_topology(atoms: usize) -> String {
    format!(
        "[ defaults ]\n\
         1 1 no 1.0 1.0\n\n\
         [ atomtypes ]\n\
         LJ 18 1.0 0.0 A 1.0 1.0\n\n\
         [ moleculetype ]\n\
         CLU 0\n\n\
         [ atoms ]\n\
         1 LJ 1 CLU LJ 1 0.0 1.0\n\n\
         [ system ]\n\
         reduced lj cluster\n\n\
         [ molecules ]\n\
         CLU {atoms}\n"
    )
}

/// The mdp parameters for a stochastic-dynamics segment.
pub fn gromacs_mdp(steps: usize, temperature: f64, seed: u64) -> String {
    let kelvin = temperature * GROMACS_KELVIN_PER_REDUCED;
    let seed = derived_seed(seed, 3);
    format!(
        "integrator = sd\n\
         dt = 0.005\n\
         nsteps = {steps}\n\
         cutoff-scheme = Verlet\n\
         nstlist = 10\n\
         pbc = xyz\n\
         rlist = 5.2\n\
         rvdw = 5.0\n\
         coulombtype = cut-off\n\
         rcoulomb = 5.0\n\
         tc-grps = System\n\
         ref-t = {kelvin:.4}\n\
         tau-t = 0.5\n\
         ld-seed = {seed}\n\
         gen-vel = yes\n\
         gen-temp = {kelvin:.4}\n\
         gen-seed = {seed}\n\
         nstxout = 0\n\
         nstvout = 0\n\
         nstlog = 0\n\
         nstenergy = 0\n"
    )
}

/// Parse a gro file's positions back to flattened coordinates, undoing
/// the shift applied on the way in.
pub fn parse_gro(text: &str, atoms: usize, shift: [f64; 3]) -> Result<Array1<f64>, EngineError> {
    let lines: Vec<&str> = text.lines().collect();
    if lines.len() < atoms + 3 {
        return Err(EngineError::Output(format!(
            "gro held {} lines, expected {}",
            lines.len(),
            atoms + 3
        )));
    }
    let mut values = Vec::with_capacity(atoms * 3);
    for line in &lines[2..2 + atoms] {
        if line.len() < 44 {
            return Err(EngineError::Output("gro row too short".into()));
        }
        for column in 0..3 {
            let field = line[20 + 8 * column..28 + 8 * column].trim();
            values.push(
                field
                    .parse::<f64>()
                    .map_err(|_| EngineError::Output(format!("bad gro float {field}")))?
                    + shift[column],
            );
        }
    }
    Ok(Array1::from(values))
}

/// GROMACS as a reduced-unit Lennard-Jones segment engine, subprocess
/// driven because no embeddable C API exists.
pub struct GromacsLj {
    /// The `gmx` binary.
    pub binary: PathBuf,
    /// Scratch directory for segment files.
    pub workdir: PathBuf,
}

impl GromacsLj {
    fn run(&self, args: &[&str]) -> Result<(), EngineError> {
        let output = Command::new(&self.binary)
            .current_dir(&self.workdir)
            .args(args)
            .output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: String = stderr
                .lines()
                .rev()
                .take(6)
                .collect::<Vec<_>>()
                .into_iter()
                .rev()
                .collect::<Vec<_>>()
                .join(" | ");
            return Err(EngineError::Run(format!("gmx {}: {tail}", args[0])));
        }
        Ok(())
    }
}

impl SegmentEngine for GromacsLj {
    fn name(&self) -> &'static str {
        "gromacs"
    }

    fn propagate(
        &self,
        x: ArrayView1<f64>,
        steps: usize,
        temperature: f64,
        seed: u64,
    ) -> Result<Array1<f64>, EngineError> {
        let atoms = x.len() / 3;
        let (low, _) = bounding_box(x, 11.0);
        std::fs::create_dir_all(&self.workdir)?;
        std::fs::write(self.workdir.join("conf.gro"), gromacs_gro(x))?;
        std::fs::write(self.workdir.join("topol.top"), gromacs_topology(atoms))?;
        std::fs::write(
            self.workdir.join("segment.mdp"),
            gromacs_mdp(steps, temperature, seed),
        )?;
        self.run(&[
            "grompp",
            "-f",
            "segment.mdp",
            "-c",
            "conf.gro",
            "-p",
            "topol.top",
            "-o",
            "segment.tpr",
            "-maxwarn",
            "2",
        ])?;
        self.run(&["mdrun", "-deffnm", "segment", "-nt", "1"])?;
        let confout = std::fs::read_to_string(self.workdir.join("segment.gro"))?;
        parse_gro(&confout, atoms, low)
    }
}

/// Build an engine from its name. LAMMPS runs in process through
/// `liblammps` (feature `lammps`); its binary path is unused and the
/// workdir irrelevant. GROMACS needs the `gmx` binary path and a
/// scratch directory.
pub fn engine_by_name(
    name: &str,
    binary: &Path,
    workdir: &Path,
) -> Option<Box<dyn SegmentEngine + Send>> {
    match name {
        #[cfg(feature = "lammps")]
        "lammps" => {
            let _ = (binary, workdir);
            Some(Box::new(LammpsLj::new()))
        }
        #[cfg(not(feature = "lammps"))]
        "lammps" => None,
        "gromacs" => Some(Box::new(GromacsLj {
            binary: binary.to_path_buf(),
            workdir: workdir.to_path_buf(),
        })),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn dimer() -> Array1<f64> {
        array![0.0, 0.0, 0.0, 1.12, 0.0, 0.0]
    }

    #[test]
    fn derived_seeds_are_positive_31_bit_and_stream_distinct() {
        assert!(derived_seed(0, 1) >= 1);
        assert!(derived_seed(u64::MAX, 2) < 900_000_000);
        assert_ne!(derived_seed(7, 1), derived_seed(7, 2));
    }

    #[test]
    fn a_gro_file_round_trips_with_its_shift() {
        let x = dimer();
        let text = gromacs_gro(x.view());
        let (low, _) = bounding_box(x.view(), 11.0);
        let back = parse_gro(&text, 2, low).unwrap();
        for (a, b) in x.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-3, "gro precision loss {a} vs {b}");
        }
    }

    #[test]
    fn the_gromacs_temperature_map_is_the_gas_constant() {
        let mdp = gromacs_mdp(100, 1.0, 7);
        let expected = format!("ref-t = {:.4}", 1.0 / 0.008_314_46);
        assert!(mdp.contains(&expected), "{mdp}");
        assert!(mdp.contains("nsteps = 100"));
    }

    #[test]
    fn unknown_engines_are_refused() {
        assert!(engine_by_name("espresso", Path::new("x"), Path::new("y")).is_none());
    }

    #[cfg(feature = "lammps")]
    #[test]
    fn a_lammps_segment_moves_a_dimer_and_keeps_it_bound() {
        let engine = LammpsLj::new();
        let x = dimer();
        let y = engine.propagate(x.view(), 50, 0.3, 42).unwrap();
        assert_eq!(y.len(), x.len());
        let displacement = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        assert!(displacement > 1e-6, "dynamics did not move the dimer");
        let bond = ((y[0] - y[3]).powi(2) + (y[1] - y[4]).powi(2) + (y[2] - y[5]).powi(2)).sqrt();
        assert!(
            (0.6..3.0).contains(&bond),
            "dimer unbound or collapsed: bond {bond}"
        );
        // Determinism: the same seed replays the same trajectory on a
        // fresh instance.
        let engine_again = LammpsLj::new();
        let z = engine_again.propagate(x.view(), 50, 0.3, 42).unwrap();
        for (a, b) in y.iter().zip(z.iter()) {
            assert!((a - b).abs() < 1e-12, "same seed diverged: {a} vs {b}");
        }
    }
}
