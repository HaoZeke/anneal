// MIT License
// Copyright 2023--present rgpot developers

//! Optional MolSSI Driver Interface engine frontend.
//!
//! The frontend is deliberately only a protocol adapter. It dispatches every
//! evaluation through [`rgpot_potential_calculate`](crate::c_api::potential::rgpot_potential_calculate),
//! so no concrete potential is known here. MDI quantities are atomic units;
//! callbacks continue to receive rgpot's native Angstrom/eV units.

use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::path::{Path, PathBuf};

use libloading::Library;

use crate::c_api::potential::rgpot_potential_calculate;
use crate::eindir::rgpot_potential_t;
use crate::status::rgpot_status_t;
use crate::tensor::{
    copy_f64_tensor, rgpot_tensor_cpu_f64_2d, rgpot_tensor_cpu_f64_matrix3,
    rgpot_tensor_cpu_i32_1d, rgpot_tensor_free, validate_positions,
};
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};
use crate::units::{bohr_to_angstrom, ev_per_angstrom_to_hartree_per_bohr, ev_to_hartree};

const NODE: &str = "@DEFAULT";
const COMMANDS: &[&str] = &[
    ">NATOMS",
    "<NATOMS",
    ">COORDS",
    "<COORDS",
    ">CELL",
    "<CELL",
    ">ELEMENTS",
    ">TYPES",
    "<ENERGY",
    "<FORCES",
    "<PE",
    "EXIT",
];

type MdiComm = c_int;
type MdiDatatype = c_int;
type InitCode = unsafe extern "C" fn() -> c_int;
type InitWithOptions = unsafe extern "C" fn(*const c_char) -> c_int;
type RegisterNode = unsafe extern "C" fn(*const c_char) -> c_int;
type RegisterCommand = unsafe extern "C" fn(*const c_char, *const c_char) -> c_int;
type AcceptCommunicator = unsafe extern "C" fn(*mut MdiComm) -> c_int;
type RecvCommand = unsafe extern "C" fn(*mut c_char, MdiComm) -> c_int;
type Send = unsafe extern "C" fn(*const c_void, c_int, MdiDatatype, MdiComm) -> c_int;
type Recv = unsafe extern "C" fn(*mut c_void, c_int, MdiDatatype, MdiComm) -> c_int;
type GetMethod = unsafe extern "C" fn(*mut c_int, MdiComm) -> c_int;

/// Runtime configuration for the optional MDI engine.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct MdiEngineConfig {
    /// Options passed verbatim to `MDI_Init_with_options`.
    pub options: String,
    /// Explicit official MDI Library shared object.
    pub library_path: Option<PathBuf>,
}

impl Default for MdiEngineConfig {
    fn default() -> Self {
        Self {
            options: "-role ENGINE -name RGPOT -method TCP -hostname localhost -port 8021".into(),
            library_path: None,
        }
    }
}

impl MdiEngineConfig {
    /// Set the MDI Library option string.
    pub fn with_options(mut self, options: impl Into<String>) -> Self {
        self.options = options.into();
        self
    }

    /// Set the path to the official MDI Library shared object.
    pub fn with_library_path(mut self, path: impl Into<PathBuf>) -> Self {
        self.library_path = Some(path.into());
        self
    }
}

#[derive(Debug)]
#[non_exhaustive]
pub struct MdiError(String);

impl std::fmt::Display for MdiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for MdiError {}

type Result<T> = std::result::Result<T, MdiError>;

struct MdiApi {
    _library: Library,
    init_code: InitCode,
    init_with_options: InitWithOptions,
    register_node: RegisterNode,
    register_command: RegisterCommand,
    accept_communicator: AcceptCommunicator,
    recv_command: RecvCommand,
    send: Send,
    recv: Recv,
    get_method: GetMethod,
    int_type: MdiDatatype,
    double_type: MdiDatatype,
    tcp_method: c_int,
    library_version: [c_int; 3],
    command_length: usize,
    get_current_code: Option<unsafe extern "C" fn() -> c_int>,
    get_communicator:
        Option<unsafe extern "C" fn(usize, MdiComm, *mut *mut MdiCommunicatorPrefix) -> c_int>,
}

#[repr(C)]
struct MdiCommunicatorPrefix {
    mdi_version: [c_int; 3],
    nodes: *mut c_void,
    method_data: *mut c_void,
    send: *mut c_void,
    recv: *mut c_void,
    delete: *mut c_void,
    code_id: usize,
    sockfd: c_int,
    method_id: c_int,
    id: MdiComm,
}

impl MdiApi {
    fn load(path: Option<&Path>) -> Result<Self> {
        let candidates = if let Some(path) = path {
            vec![path.to_path_buf()]
        } else if let Some(path) = std::env::var_os("RGPOT_MDI_LIBRARY") {
            vec![PathBuf::from(path)]
        } else {
            vec![
                PathBuf::from("libmdi.so"),
                PathBuf::from("libmdi.so.1"),
                PathBuf::from("libmdi.dylib"),
                PathBuf::from("mdi.dll"),
            ]
        };
        let mut errors = Vec::new();
        for candidate in candidates {
            let library = match unsafe { Library::new(&candidate) } {
                Ok(library) => library,
                Err(error) => {
                    errors.push(format!("{}: {error}", candidate.display()));
                    continue;
                }
            };
            return unsafe { Self::from_library(library) };
        }
        Err(MdiError(format!(
            "could not load the official MDI Library; set RGPOT_MDI_LIBRARY: {}",
            errors.join("; ")
        )))
    }

    unsafe fn from_library(library: Library) -> Result<Self> {
        unsafe fn function<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
            unsafe { library.get::<T>(name) }
                .map(|symbol| *symbol)
                .map_err(|error| {
                    MdiError(format!(
                        "MDI Library has no {}: {error}",
                        String::from_utf8_lossy(&name[..name.len() - 1])
                    ))
                })
        }
        unsafe fn constant(library: &Library, name: &[u8]) -> Result<c_int> {
            let pointer = unsafe { function::<*const c_int>(library, name)? };
            if pointer.is_null() {
                return Err(MdiError("MDI Library exported a null constant".into()));
            }
            Ok(unsafe { *pointer })
        }

        let init_code = unsafe { function(&library, b"MDI_Init_code\0")? };
        let init_with_options = unsafe { function(&library, b"MDI_Init_with_options\0")? };
        let register_node = unsafe { function(&library, b"MDI_Register_Node\0")? };
        let register_command = unsafe { function(&library, b"MDI_Register_Command\0")? };
        let accept_communicator = unsafe { function(&library, b"MDI_Accept_Communicator\0")? };
        let recv_command = unsafe { function(&library, b"MDI_Recv_Command\0")? };
        let send = unsafe { function(&library, b"MDI_Send\0")? };
        let recv = unsafe { function(&library, b"MDI_Recv\0")? };
        let get_method = unsafe { function(&library, b"MDI_Get_method\0")? };
        let int_type = unsafe { constant(&library, b"MDI_INT\0")? };
        let double_type = unsafe { constant(&library, b"MDI_DOUBLE\0")? };
        let tcp_method = unsafe { constant(&library, b"MDI_TCP\0")? };
        let library_version = [
            unsafe { constant(&library, b"MDI_MAJOR_VERSION\0")? },
            unsafe { constant(&library, b"MDI_MINOR_VERSION\0")? },
            unsafe { constant(&library, b"MDI_PATCH_VERSION\0")? },
        ];
        let command_length =
            usize::try_from(unsafe { constant(&library, b"MDI_COMMAND_LENGTH\0")? })
                .map_err(|_| MdiError("MDI_COMMAND_LENGTH is negative".into()))?;
        let get_current_code = unsafe {
            library
                .get::<unsafe extern "C" fn() -> c_int>(b"MDI_Get_Current_Code\0")
                .ok()
                .map(|symbol| *symbol)
        };
        let get_communicator = unsafe {
            library
                .get::<
                    unsafe extern "C" fn(
                        usize,
                        MdiComm,
                        *mut *mut MdiCommunicatorPrefix,
                    ) -> c_int,
                >(b"get_communicator\0")
                .ok()
                .map(|symbol| *symbol)
        };
        Ok(Self {
            _library: library,
            init_code,
            init_with_options,
            register_node,
            register_command,
            accept_communicator,
            recv_command,
            send,
            recv,
            get_method,
            int_type,
            double_type,
            tcp_method,
            library_version,
            command_length,
            get_current_code,
            get_communicator,
        })
    }

    fn check(code: c_int, operation: &str) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(MdiError(format!(
                "{operation} failed with MDI status {code}"
            )))
        }
    }

    fn register(&self) -> Result<()> {
        let node = CString::new(NODE).map_err(|error| MdiError(error.to_string()))?;
        Self::check(
            unsafe { (self.register_node)(node.as_ptr()) },
            "MDI_Register_Node",
        )?;
        for command in COMMANDS {
            let command = CString::new(*command).map_err(|error| MdiError(error.to_string()))?;
            Self::check(
                unsafe { (self.register_command)(node.as_ptr(), command.as_ptr()) },
                "MDI_Register_Command",
            )?;
        }
        Ok(())
    }

    fn recv_values<T>(&self, count: usize, datatype: MdiDatatype, comm: MdiComm) -> Result<Vec<T>>
    where
        T: Default + Clone,
    {
        self.enable_tcp_nodelay(comm)?;
        let wire_count =
            c_int::try_from(count).map_err(|_| MdiError("MDI value count exceeds int".into()))?;
        let mut values = vec![T::default(); count];
        Self::check(
            unsafe {
                (self.recv)(
                    values.as_mut_ptr().cast::<c_void>(),
                    wire_count,
                    datatype,
                    comm,
                )
            },
            "MDI_Recv",
        )?;
        self.enable_tcp_nodelay(comm)?;
        Ok(values)
    }

    fn send_values<T>(&self, values: &[T], datatype: MdiDatatype, comm: MdiComm) -> Result<()> {
        let count = c_int::try_from(values.len())
            .map_err(|_| MdiError("MDI value count exceeds int".into()))?;
        Self::check(
            unsafe { (self.send)(values.as_ptr().cast::<c_void>(), count, datatype, comm) },
            "MDI_Send",
        )
    }

    fn enable_tcp_nodelay(&self, comm: MdiComm) -> Result<()> {
        let mut method = 0;
        Self::check(
            unsafe { (self.get_method)(&mut method, comm) },
            "MDI_Get_method",
        )?;
        if method != self.tcp_method {
            return Ok(());
        }
        if self.library_version[0] != 1 || self.library_version[1] != 4 {
            return Err(MdiError(format!(
                "TCP_NODELAY socket access supports MDI Library 1.4.x, found {}.{}.{}",
                self.library_version[0], self.library_version[1], self.library_version[2]
            )));
        }
        let current = self.get_current_code.ok_or_else(|| {
            MdiError("MDI Library cannot expose its TCP socket for TCP_NODELAY".into())
        })?;
        let lookup = self.get_communicator.ok_or_else(|| {
            MdiError("MDI Library cannot expose its TCP socket for TCP_NODELAY".into())
        })?;
        let code = unsafe { current() };
        let code =
            usize::try_from(code).map_err(|_| MdiError("negative MDI code identifier".into()))?;
        let mut pointer = std::ptr::null_mut();
        Self::check(
            unsafe { lookup(code, comm, &mut pointer) },
            "MDI communicator lookup",
        )?;
        if pointer.is_null() {
            return Err(MdiError("MDI communicator lookup returned null".into()));
        }
        #[cfg(unix)]
        {
            let communicator = unsafe { &*pointer };
            if communicator.code_id != code
                || communicator.method_id != self.tcp_method
                || communicator.id != comm
            {
                return Err(MdiError(
                    "MDI 1.4 communicator layout failed validation".into(),
                ));
            }
            let mut socket_type: c_int = 0;
            let mut socket_type_len = std::mem::size_of::<c_int>() as libc::socklen_t;
            let status = unsafe {
                libc::getsockopt(
                    communicator.sockfd,
                    libc::SOL_SOCKET,
                    libc::SO_TYPE,
                    (&mut socket_type as *mut c_int).cast::<c_void>(),
                    &mut socket_type_len,
                )
            };
            if status != 0 || socket_type != libc::SOCK_STREAM {
                return Err(MdiError(
                    "MDI 1.4 communicator does not contain a TCP stream socket".into(),
                ));
            }
            let enabled: c_int = 1;
            let status = unsafe {
                libc::setsockopt(
                    communicator.sockfd,
                    libc::IPPROTO_TCP,
                    libc::TCP_NODELAY,
                    (&enabled as *const c_int).cast::<c_void>(),
                    std::mem::size_of::<c_int>() as libc::socklen_t,
                )
            };
            if status != 0 {
                return Err(MdiError(format!(
                    "setsockopt(TCP_NODELAY) failed: {}",
                    std::io::Error::last_os_error()
                )));
            }
            #[cfg(target_os = "linux")]
            {
                let status = unsafe {
                    libc::setsockopt(
                        communicator.sockfd,
                        libc::IPPROTO_TCP,
                        libc::TCP_QUICKACK,
                        (&enabled as *const c_int).cast::<c_void>(),
                        std::mem::size_of::<c_int>() as libc::socklen_t,
                    )
                };
                if status != 0 {
                    return Err(MdiError(format!(
                        "setsockopt(TCP_QUICKACK) failed: {}",
                        std::io::Error::last_os_error()
                    )));
                }
            }
        }
        #[cfg(not(unix))]
        {
            return Err(MdiError(
                "TCP_NODELAY is not implemented for MDI on this platform".into(),
            ));
        }
        Ok(())
    }
}

struct Engine<'a> {
    api: &'a MdiApi,
    pot: *const rgpot_potential_t,
    natoms: Option<usize>,
    species: Option<Vec<i32>>,
    coords_bohr: Option<Vec<f64>>,
    cell_bohr: [f64; 9],
    cached: Option<(f64, Vec<f64>)>,
    evaluations: usize,
}

impl<'a> Engine<'a> {
    fn new(api: &'a MdiApi, pot: *const rgpot_potential_t) -> Self {
        Self {
            api,
            pot,
            natoms: None,
            species: None,
            coords_bohr: None,
            cell_bohr: [0.0; 9],
            cached: None,
            evaluations: 0,
        }
    }

    fn invalidate(&mut self) {
        self.cached = None;
    }

    fn set_natoms(&mut self, natoms: i32) -> Result<()> {
        let natoms = usize::try_from(natoms)
            .map_err(|_| MdiError(">NATOMS must be a positive integer".into()))?;
        if natoms == 0 {
            return Err(MdiError(">NATOMS must be a positive integer".into()));
        }
        if self.natoms != Some(natoms) {
            self.natoms = Some(natoms);
            self.species = None;
            self.coords_bohr = None;
            self.invalidate();
        }
        Ok(())
    }

    fn require_natoms(&self) -> Result<usize> {
        self.natoms
            .ok_or_else(|| MdiError("send >NATOMS before this command".into()))
    }

    fn set_finite(&mut self, values: Vec<f64>, label: &str) -> Result<Vec<f64>> {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(MdiError(format!(
                "{label} contains a non-finite value at index {index}"
            )));
        }
        Ok(values)
    }

    fn evaluate(&mut self) -> Result<&(f64, Vec<f64>)> {
        if self.cached.is_none() {
            let natoms = self.require_natoms()?;
            let species = self.species.as_ref().ok_or_else(|| {
                MdiError("send >ELEMENTS or >TYPES before requesting results".into())
            })?;
            let coords = self
                .coords_bohr
                .as_ref()
                .ok_or_else(|| MdiError("send >COORDS before requesting results".into()))?;
            let mut positions: Vec<f64> = coords.iter().copied().map(bohr_to_angstrom).collect();
            let mut cell: Vec<f64> = self
                .cell_bohr
                .iter()
                .copied()
                .map(bohr_to_angstrom)
                .collect();
            let mut species = species.clone();
            let input = rgpot_force_input_t {
                positions: unsafe {
                    rgpot_tensor_cpu_f64_2d(positions.as_mut_ptr(), natoms as i64, 3)
                },
                atomic_numbers: unsafe {
                    rgpot_tensor_cpu_i32_1d(species.as_mut_ptr(), natoms as i64)
                },
                box_matrix: unsafe { rgpot_tensor_cpu_f64_matrix3(cell.as_mut_ptr()) },
            };
            let mut output = rgpot_force_out_t {
                forces: std::ptr::null_mut(),
                energy: 0.0,
                variance: 0.0,
            };
            let status = unsafe { rgpot_potential_calculate(self.pot, &input, &mut output) };
            let forces = if status == rgpot_status_t::RGPOT_SUCCESS {
                if !output.energy.is_finite() {
                    Err(MdiError("potential energy is non-finite".into()))
                } else {
                    copy_f64_tensor(output.forces, "potential forces", natoms * 3, true)
                        .and_then(|forces| {
                            let force_rows = validate_positions(output.forces)?;
                            if force_rows != natoms {
                                Err(format!(
                                    "potential forces: expected shape=[{natoms},3], got [{force_rows},3]"
                                ))
                            } else {
                                Ok(forces)
                            }
                        })
                        .map_err(MdiError)
                }
            } else {
                Err(MdiError(format!(
                    "rgpot potential evaluation failed with status {}",
                    status as i32
                )))
            };
            unsafe {
                rgpot_tensor_free(output.forces);
                rgpot_tensor_free(input.positions);
                rgpot_tensor_free(input.atomic_numbers);
                rgpot_tensor_free(input.box_matrix);
            }
            let forces = forces?;
            self.cached = Some((
                ev_to_hartree(output.energy),
                forces
                    .into_iter()
                    .map(ev_per_angstrom_to_hartree_per_bohr)
                    .collect(),
            ));
            self.evaluations += 1;
        }
        self.cached
            .as_ref()
            .ok_or_else(|| MdiError("evaluation cache was not populated".into()))
    }

    fn command(&mut self, command: &str, comm: MdiComm) -> Result<bool> {
        match command {
            "EXIT" => return Ok(false),
            ">NATOMS" => {
                let values = self.api.recv_values::<i32>(1, self.api.int_type, comm)?;
                self.set_natoms(values[0])?;
            }
            "<NATOMS" => {
                let natoms = i32::try_from(self.require_natoms()?)
                    .map_err(|_| MdiError("atom count exceeds MDI_INT".into()))?;
                self.api.send_values(&[natoms], self.api.int_type, comm)?;
            }
            ">ELEMENTS" | ">TYPES" => {
                let natoms = self.require_natoms()?;
                let values = self
                    .api
                    .recv_values::<i32>(natoms, self.api.int_type, comm)?;
                if values.iter().any(|value| *value < 0) {
                    return Err(MdiError(format!("{command} values must be non-negative")));
                }
                if self.species.as_ref() != Some(&values) {
                    self.species = Some(values);
                    self.invalidate();
                }
            }
            ">COORDS" => {
                let count = self
                    .require_natoms()?
                    .checked_mul(3)
                    .ok_or_else(|| MdiError("coordinate count overflow".into()))?;
                let values = self
                    .api
                    .recv_values::<f64>(count, self.api.double_type, comm)?;
                let values = self.set_finite(values, ">COORDS")?;
                if self.coords_bohr.as_ref() != Some(&values) {
                    self.coords_bohr = Some(values);
                    self.invalidate();
                }
            }
            "<COORDS" => {
                let coords = self
                    .coords_bohr
                    .as_ref()
                    .ok_or_else(|| MdiError("send >COORDS before <COORDS".into()))?;
                self.api.send_values(coords, self.api.double_type, comm)?;
            }
            ">CELL" => {
                let values = self.api.recv_values::<f64>(9, self.api.double_type, comm)?;
                let values = self.set_finite(values, ">CELL")?;
                let cell: [f64; 9] = values
                    .try_into()
                    .map_err(|_| MdiError(">CELL must contain nine values".into()))?;
                if self.cell_bohr != cell {
                    self.cell_bohr = cell;
                    self.invalidate();
                }
            }
            "<CELL" => {
                self.api
                    .send_values(&self.cell_bohr, self.api.double_type, comm)?;
            }
            "<ENERGY" | "<PE" => {
                let energy = self.evaluate()?.0;
                self.api
                    .send_values(&[energy], self.api.double_type, comm)?;
            }
            "<FORCES" => {
                let forces = self.evaluate()?.1.clone();
                self.api.send_values(&forces, self.api.double_type, comm)?;
            }
            _ => {
                return Err(MdiError(format!(
                    "unsupported MDI command {command:?} at {NODE}"
                )));
            }
        }
        Ok(true)
    }

    fn serve(mut self, comm: MdiComm) -> Result<usize> {
        let buffer_len = self
            .api
            .command_length
            .checked_add(1)
            .ok_or_else(|| MdiError("MDI command length overflow".into()))?;
        let mut buffer = vec![0_i8; buffer_len];
        loop {
            buffer.fill(0);
            self.api.enable_tcp_nodelay(comm)?;
            MdiApi::check(
                unsafe { (self.api.recv_command)(buffer.as_mut_ptr(), comm) },
                "MDI_Recv_Command",
            )?;
            self.api.enable_tcp_nodelay(comm)?;
            let command = unsafe { CStr::from_ptr(buffer.as_ptr()) }
                .to_str()
                .map_err(|error| MdiError(format!("MDI command is not UTF-8: {error}")))?;
            if !self.command(command, comm)? {
                return Ok(self.evaluations);
            }
        }
    }
}

/// Serve one MDI communicator from any rgpot potential.
///
/// The call blocks through `EXIT`. The returned count is the number of
/// potential evaluations, which can be smaller than the number of result
/// commands because energy and forces share one coordinate-keyed cache entry.
///
/// # Safety
///
/// `pot` must remain a valid `rgpot_potential_t` for the duration of the call.
pub unsafe fn serve_mdi(pot: *const rgpot_potential_t, config: &MdiEngineConfig) -> Result<usize> {
    if pot.is_null() {
        return Err(MdiError("MDI potential is null".into()));
    }
    let api = MdiApi::load(config.library_path.as_deref())?;
    let options =
        CString::new(config.options.as_str()).map_err(|error| MdiError(error.to_string()))?;
    MdiApi::check(unsafe { (api.init_code)() }, "MDI_Init_code")?;
    MdiApi::check(
        unsafe { (api.init_with_options)(options.as_ptr()) },
        "MDI_Init_with_options",
    )?;
    api.register()?;
    let mut comm = 0;
    MdiApi::check(
        unsafe { (api.accept_communicator)(&mut comm) },
        "MDI_Accept_Communicator",
    )?;
    api.enable_tcp_nodelay(comm)?;
    Engine::new(&api, pot).serve(comm)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_and_setters() {
        let config = MdiEngineConfig::default()
            .with_options("-role ENGINE -name TEST -method TCP -port 1")
            .with_library_path("/tmp/libmdi.so");
        assert!(config.options.contains("-name TEST"));
        assert_eq!(
            config.library_path.as_deref(),
            Some(Path::new("/tmp/libmdi.so"))
        );
    }

    #[test]
    fn unit_conversion_matches_atomic_units() {
        assert_eq!(bohr_to_angstrom(1.0), crate::units::BOHR_TO_ANGSTROM);
        assert_eq!(ev_to_hartree(27.211_386_245_988), 1.0);
        assert_eq!(
            ev_per_angstrom_to_hartree_per_bohr(27.211_386_245_988),
            crate::units::BOHR_TO_ANGSTROM
        );
    }
}
