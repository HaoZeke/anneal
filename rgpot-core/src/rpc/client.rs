// MIT License
// Copyright 2023--present rgpot developers

//! Async RPC client wrapping Cap'n Proto `Potential.calculate()`.
//!
//! The client owns a tokio runtime so that the C API can call it
//! synchronously.
//!
//! ## DLPack Integration
//!
//! Input tensors are read from `DLManagedTensorVersioned` pointers (CPU only
//! for RPC — GPU tensors would need a device-to-host copy first).  The
//! response forces are wrapped in an **owning** DLPack tensor so the caller
//! can free them via `rgpot_tensor_free`.

use capnp::Error as CapnpError;
use capnp_rpc::{rpc_twoparty_capnp, twoparty, RpcSystem};
use dlpk::sys::DLDeviceType;
use futures::AsyncReadExt;
use tokio::runtime::Runtime;

use crate::rpc::schema::potential;
use crate::tensor::create_owned_f64_tensor;
use crate::types::{rgpot_force_input_t, rgpot_force_out_t};

/// RPC client that connects to a remote rgpot server.
pub struct RpcClient {
    runtime: Runtime,
    addr: String,
}

impl RpcClient {
    /// Create a new RPC client targeting `host:port`.
    ///
    /// The connection is established lazily on the first `calculate` call.
    pub fn new(host: &str, port: u16) -> Result<Self, String> {
        let runtime = Runtime::new().map_err(|e| format!("failed to create tokio runtime: {e}"))?;
        Ok(Self {
            runtime,
            addr: format!("{host}:{port}"),
        })
    }

    /// Perform a synchronous RPC calculation.
    ///
    /// Internally this blocks on the tokio runtime with a `LocalSet`
    /// (required because `capnp_rpc::RpcSystem` is `!Send`).
    pub fn calculate(
        &mut self,
        input: &rgpot_force_input_t,
        output: &mut rgpot_force_out_t,
    ) -> Result<(), String> {
        let local = tokio::task::LocalSet::new();
        local.block_on(&self.runtime, self.calculate_async(input, output))
    }

    async fn calculate_async(
        &self,
        input: &rgpot_force_input_t,
        output: &mut rgpot_force_out_t,
    ) -> Result<(), String> {
        // --- Extract data from DLPack tensors (CPU only) ---
        let n = unsafe { input.n_atoms() }
            .ok_or_else(|| "cannot determine n_atoms from input tensors".to_string())?;

        let (positions, atmnrs, box_data) = unsafe { extract_cpu_input(input, n)? };

        // --- Connect ---
        let stream = tokio::net::TcpStream::connect(&self.addr)
            .await
            .map_err(|e| format!("connection failed: {e}"))?;
        stream
            .set_nodelay(true)
            .map_err(|e| format!("set_nodelay failed: {e}"))?;

        let (reader, writer) = tokio_util::compat::TokioAsyncReadCompatExt::compat(stream).split();

        let network = twoparty::VatNetwork::new(
            futures::io::BufReader::new(reader),
            futures::io::BufWriter::new(writer),
            rpc_twoparty_capnp::Side::Client,
            Default::default(),
        );

        let mut rpc_system = RpcSystem::new(Box::new(network), None);
        let potential_client: potential::Client =
            rpc_system.bootstrap(rpc_twoparty_capnp::Side::Server);

        tokio::task::spawn_local(rpc_system);

        // --- Refuse an incompatible or outdated server before any work ---
        crate::compat::check_server(&potential_client, &crate::compat::Expectation::default())
            .await?;

        // --- Build capnp request ---
        let mut request = potential_client.calculate_request();
        {
            let mut fip = request.get().init_fip();

            let mut pos_builder = fip.reborrow().init_pos(positions.len() as u32);
            for (i, &val) in positions.iter().enumerate() {
                pos_builder.set(i as u32, val);
            }

            let mut atm_builder = fip.reborrow().init_atmnrs(atmnrs.len() as u32);
            for (i, &val) in atmnrs.iter().enumerate() {
                atm_builder.set(i as u32, val);
            }

            let mut box_builder = fip.init_box(9);
            for (i, &val) in box_data.iter().enumerate() {
                box_builder.set(i as u32, val);
            }
        }

        // --- Send and receive ---
        let response = request
            .send()
            .promise
            .await
            .map_err(|e: CapnpError| format!("RPC call failed: {e}"))?;

        let result = response
            .get()
            .map_err(|e| format!("failed to read response: {e}"))?
            .get_result()
            .map_err(|e| format!("failed to get result: {e}"))?;

        output.energy = result.get_energy();

        let forces = result
            .get_forces()
            .map_err(|e| format!("failed to read forces: {e}"))?;

        if forces.len() as usize != n * 3 {
            return Err(format!(
                "force array size mismatch: expected {}, got {}",
                n * 3,
                forces.len()
            ));
        }

        // Create an owning DLPack tensor for the forces
        let forces_vec: Vec<f64> = (0..forces.len()).map(|i| forces.get(i)).collect();
        output.forces = create_owned_f64_tensor(forces_vec, vec![n as i64, 3]);

        Ok(())
    }
}

type CpuInput<'a> = (&'a [f64], &'a [i32], &'a [f64]);

/// Extract CPU data slices from DLPack input tensors.
///
/// # Safety
/// All tensor pointers in `input` must be valid DLPack tensors on kDLCPU.
unsafe fn extract_cpu_input(input: &rgpot_force_input_t, n: usize) -> Result<CpuInput<'_>, String> {
    // Positions
    if input.positions.is_null() {
        return Err("positions tensor is NULL".into());
    }
    let pos_t = unsafe { &(*input.positions).dl_tensor };
    if pos_t.device.device_type != DLDeviceType::kDLCPU {
        return Err("RPC requires CPU tensors; positions is not on CPU".into());
    }
    let positions = unsafe { std::slice::from_raw_parts(pos_t.data as *const f64, n * 3) };

    // Atomic numbers
    if input.atomic_numbers.is_null() {
        return Err("atomic_numbers tensor is NULL".into());
    }
    let atm_t = unsafe { &(*input.atomic_numbers).dl_tensor };
    if atm_t.device.device_type != DLDeviceType::kDLCPU {
        return Err("RPC requires CPU tensors; atomic_numbers is not on CPU".into());
    }
    let atmnrs = unsafe { std::slice::from_raw_parts(atm_t.data as *const i32, n) };

    // Box matrix
    if input.box_matrix.is_null() {
        return Err("box_matrix tensor is NULL".into());
    }
    let box_t = unsafe { &(*input.box_matrix).dl_tensor };
    if box_t.device.device_type != DLDeviceType::kDLCPU {
        return Err("RPC requires CPU tensors; box_matrix is not on CPU".into());
    }
    let box_data = unsafe { std::slice::from_raw_parts(box_t.data as *const f64, 9) };

    Ok((positions, atmnrs, box_data))
}
