// MIT License
// Copyright 2023--present rgpot developers

//! Capability handshake: decide from a backend's `Capabilities` message
//! whether this build can drive it, before any work is dispatched.
//!
//! The host states what it speaks as an [`Expectation`]. A backend that
//! declares an incompatible protocol family or major, a lower protocol minor
//! than required, a different schema identity, an incompatible eindir bridge
//! or DLPack revision, or that lacks a required operation is refused with a
//! message naming the first mismatch. Metadata a backend leaves unset (empty
//! family, zero bridge major) is not a declaration and is not checked.

use crate::Potentials_capnp::capabilities::{Builder, Operation, Reader};

/// Protocol family this build speaks.
pub const PROTOCOL_FAMILY: &str = "rgpot.potentials";
/// Wire-incompatible protocol revision.
pub const PROTOCOL_MAJOR: u16 = 1;
/// Additive protocol revision.
pub const PROTOCOL_MINOR: u16 = 0;
/// Cap'n Proto identity of `Potentials.capnp`.
pub const SCHEMA_ID: &str = "0xbd1f89fa17369103";

/// What the host requires of a backend.
#[derive(Debug, Clone)]
pub struct Expectation {
    pub family: &'static str,
    pub protocol_major: u16,
    /// Lowest protocol minor the host accepts.
    pub protocol_minor_min: u16,
    pub schema_id: &'static str,
    pub bridge_abi_major: u16,
    /// Highest bridge minor the host can consume.
    pub bridge_abi_minor_max: u16,
    pub bridge_layout: u32,
    pub dlpack_major: u16,
    /// Highest DLPack minor the host can consume.
    pub dlpack_minor_max: u16,
    /// Bridge feature bits the host understands.
    pub bridge_features: u64,
    pub required_operations: Vec<Operation>,
}

impl Default for Expectation {
    fn default() -> Self {
        let stamp = eindir_core::ffi::eindir_core_abi_stamp();
        Self {
            family: PROTOCOL_FAMILY,
            protocol_major: PROTOCOL_MAJOR,
            protocol_minor_min: 0,
            schema_id: SCHEMA_ID,
            bridge_abi_major: stamp.abi_major as u16,
            bridge_abi_minor_max: stamp.abi_minor as u16,
            bridge_layout: stamp.objective_layout,
            dlpack_major: stamp.dlpack_major as u16,
            dlpack_minor_max: stamp.dlpack_minor as u16,
            bridge_features: stamp.features,
            required_operations: vec![Operation::Energy, Operation::Forces],
        }
    }
}

fn text(result: capnp::Result<capnp::text::Reader<'_>>, field: &str) -> Result<String, String> {
    result
        .map_err(|e| format!("capabilities.{field} unreadable: {e}"))?
        .to_string()
        .map_err(|e| format!("capabilities.{field} is not UTF-8: {e}"))
}

/// Check `caps` against `want`; the error names the first incompatibility.
pub fn check_capabilities(caps: Reader<'_>, want: &Expectation) -> Result<(), String> {
    let family = text(caps.get_protocol_family(), "protocolFamily")?;
    if !family.is_empty() {
        if family != want.family {
            return Err(format!(
                "protocol family {family:?} is not {:?}",
                want.family
            ));
        }
        if caps.get_protocol_major() != want.protocol_major {
            return Err(format!(
                "protocol major {} is not {}",
                caps.get_protocol_major(),
                want.protocol_major
            ));
        }
        if caps.get_protocol_minor() < want.protocol_minor_min {
            return Err(format!(
                "protocol minor {} is below the required {}",
                caps.get_protocol_minor(),
                want.protocol_minor_min
            ));
        }
    }
    let schema_id = text(caps.get_schema_id(), "schemaId")?;
    if !schema_id.is_empty() && schema_id != want.schema_id {
        return Err(format!("schema id {schema_id} is not {}", want.schema_id));
    }
    if caps.get_bridge_abi_major() != 0 {
        if caps.get_bridge_abi_major() != want.bridge_abi_major {
            return Err(format!(
                "eindir bridge ABI major {} is not {}",
                caps.get_bridge_abi_major(),
                want.bridge_abi_major
            ));
        }
        if caps.get_bridge_abi_minor() > want.bridge_abi_minor_max {
            return Err(format!(
                "eindir bridge ABI minor {} exceeds the supported {}",
                caps.get_bridge_abi_minor(),
                want.bridge_abi_minor_max
            ));
        }
        if caps.get_bridge_layout() != want.bridge_layout {
            return Err(format!(
                "eindir objective layout {} is not {}",
                caps.get_bridge_layout(),
                want.bridge_layout
            ));
        }
        if caps.get_bridge_features() & !want.bridge_features != 0 {
            return Err(format!(
                "eindir bridge features {:#x} include bits outside {:#x}",
                caps.get_bridge_features(),
                want.bridge_features
            ));
        }
    }
    if caps.get_dlpack_major() != 0 {
        if caps.get_dlpack_major() != want.dlpack_major {
            return Err(format!(
                "DLPack major {} is not {}",
                caps.get_dlpack_major(),
                want.dlpack_major
            ));
        }
        if caps.get_dlpack_minor() > want.dlpack_minor_max {
            return Err(format!(
                "DLPack minor {} exceeds the supported {}",
                caps.get_dlpack_minor(),
                want.dlpack_minor_max
            ));
        }
    }
    let ops = caps
        .get_operations()
        .map_err(|e| format!("capabilities.operations unreadable: {e}"))?;
    let served: Vec<Operation> = ops.iter().filter_map(Result::ok).collect();
    for required in &want.required_operations {
        if !served.contains(required) {
            return Err(format!("operation {required:?} is not served"));
        }
    }
    Ok(())
}

/// Decode a flat `Capabilities` message held as aligned words and check it.
pub fn check_capabilities_words(words: &[u64], want: &Expectation) -> Result<(), String> {
    let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr().cast::<u8>(), words.len() * 8) };
    let mut cursor = bytes;
    let message = capnp::serialize::read_message_from_flat_slice(
        &mut cursor,
        capnp::message::ReaderOptions::new(),
    )
    .map_err(|e| format!("capabilities unreadable: {e}"))?;
    let caps = message
        .get_root::<Reader<'_>>()
        .map_err(|e| format!("capabilities unreadable: {e}"))?;
    check_capabilities(caps, want)
}

/// Fill the compatibility fields of `caps` with what this build speaks.
pub fn fill_compatibility(mut caps: Builder<'_>) {
    let want = Expectation::default();
    caps.set_protocol_family(want.family);
    caps.set_protocol_major(want.protocol_major);
    caps.set_protocol_minor(PROTOCOL_MINOR);
    caps.set_schema_id(want.schema_id);
    caps.set_bridge_abi_major(want.bridge_abi_major);
    caps.set_bridge_abi_minor(want.bridge_abi_minor_max);
    caps.set_bridge_layout(want.bridge_layout);
    caps.set_dlpack_major(want.dlpack_major);
    caps.set_dlpack_minor(want.dlpack_minor_max);
    caps.set_bridge_features(want.bridge_features);
    #[cfg(rgpot_schema_build_identity)]
    {
        caps.set_build_version(env!("CARGO_PKG_VERSION"));
        caps.set_build_revision(env!("RGPOT_SOURCE_REVISION"));
    }
}

/// Fill `caps` as the self-description of an rgpot RPC server.
pub fn describe_server(mut caps: Builder<'_>) {
    caps.set_backend_name("rgpot");
    caps.set_backend_version(env!("CARGO_PKG_VERSION"));
    caps.set_available(true);
    let mut ops = caps.reborrow().init_operations(2);
    ops.set(0, Operation::Energy);
    ops.set(1, Operation::Forces);
    fill_compatibility(caps);
}

/// Ask a connected server for its `Capabilities` and refuse it when they are
/// incompatible. A server that predates `Potential.getCapabilities` answers
/// UNIMPLEMENTED, which is reported as a missing call with the upgrade path.
#[cfg(feature = "rpc")]
pub async fn check_server(
    server: &crate::Potentials_capnp::potential::Client,
    want: &Expectation,
) -> Result<(), String> {
    let response = match server.get_capabilities_request().send().promise.await {
        Ok(response) => response,
        Err(e) if e.kind == capnp::ErrorKind::Unimplemented => {
            return Err("the server does not implement Potential.getCapabilities; \
                        upgrade the server to an rgpot release whose Potentials.capnp \
                        carries it"
                .to_string());
        }
        Err(e) => return Err(format!("capabilities request failed: {e}")),
    };
    let caps = response
        .get()
        .and_then(|r| r.get_capabilities())
        .map_err(|e| format!("capabilities unreadable: {e}"))?;
    check_capabilities(caps, want).map_err(|why| format!("server refused: {why}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use capnp::message::Builder as Message;

    fn message(edit: impl FnOnce(&mut Builder<'_>)) -> Message<capnp::message::HeapAllocator> {
        let mut msg = Message::new_default();
        {
            let mut caps = msg.init_root::<Builder<'_>>();
            fill_compatibility(caps.reborrow());
            let mut ops = caps.reborrow().init_operations(2);
            ops.set(0, Operation::Energy);
            ops.set(1, Operation::Forces);
            edit(&mut caps);
        }
        msg
    }

    fn verdict(edit: impl FnOnce(&mut Builder<'_>), want: &Expectation) -> Result<(), String> {
        let msg = message(edit);
        check_capabilities(msg.get_root_as_reader::<Reader<'_>>().unwrap(), want)
    }

    #[test]
    fn matching_metadata_is_accepted() {
        assert_eq!(verdict(|_| {}, &Expectation::default()), Ok(()));
    }

    #[test]
    fn unstated_metadata_is_not_a_declaration() {
        let mut msg = Message::new_default();
        {
            let mut caps = msg.init_root::<Builder<'_>>();
            let mut ops = caps.reborrow().init_operations(2);
            ops.set(0, Operation::Energy);
            ops.set(1, Operation::Forces);
        }
        let reader = msg.get_root_as_reader::<Reader<'_>>().unwrap();
        assert_eq!(check_capabilities(reader, &Expectation::default()), Ok(()));
    }

    #[test]
    fn rejection_matrix() {
        let want = Expectation::default();
        type Edit = Box<dyn FnOnce(&mut Builder<'_>)>;
        let cases: Vec<(&str, Edit, &str)> = vec![
            (
                "family",
                Box::new(|c| c.set_protocol_family("other.family")),
                "protocol family",
            ),
            (
                "major",
                Box::new(|c| c.set_protocol_major(2)),
                "protocol major",
            ),
            (
                "schema",
                Box::new(|c| c.set_schema_id("0xdeadbeef")),
                "schema id",
            ),
            (
                "bridge major",
                Box::new(|c| c.set_bridge_abi_major(2)),
                "bridge ABI major",
            ),
            (
                "bridge minor",
                Box::new(|c| c.set_bridge_abi_minor(9)),
                "bridge ABI minor",
            ),
            (
                "layout",
                Box::new(|c| c.set_bridge_layout(7)),
                "objective layout",
            ),
            (
                "features",
                Box::new(|c| c.set_bridge_features(1 << 40)),
                "bridge features",
            ),
            (
                "dlpack major",
                Box::new(|c| c.set_dlpack_major(2)),
                "DLPack major",
            ),
            (
                "dlpack minor",
                Box::new(|c| c.set_dlpack_minor(9)),
                "DLPack minor",
            ),
            (
                "operation",
                Box::new(|c| {
                    let mut ops = c.reborrow().init_operations(1);
                    ops.set(0, Operation::Energy);
                }),
                "Forces",
            ),
        ];
        for (name, edit, needle) in cases {
            let err = verdict(edit, &want).expect_err(name);
            assert!(err.contains(needle), "{name}: {err}");
        }
    }

    #[cfg(rgpot_schema_build_identity)]
    #[test]
    fn producer_reports_the_build_identity() {
        let msg = message(|_| {});
        let caps = msg.get_root_as_reader::<Reader<'_>>().unwrap();
        assert_eq!(
            caps.get_build_version().unwrap().to_str().unwrap(),
            env!("CARGO_PKG_VERSION")
        );
        assert_eq!(
            caps.get_build_revision().unwrap().to_str().unwrap(),
            env!("RGPOT_SOURCE_REVISION")
        );
    }

    #[test]
    fn expectation_default_tracks_the_eindir_stamp() {
        let want = Expectation::default();
        assert_eq!(
            (
                want.bridge_abi_major,
                want.bridge_abi_minor_max,
                want.bridge_layout,
                want.dlpack_major,
                want.dlpack_minor_max,
                want.bridge_features
            ),
            (1, 0, 1, 1, 0, 0x3),
            "update Expectation in CppCore/rgpot/abi/Compat.hpp with these"
        );
    }

    #[test]
    fn protocol_minor_below_requirement_is_rejected() {
        let want = Expectation {
            protocol_minor_min: 2,
            ..Expectation::default()
        };
        let err = verdict(|_| {}, &want).unwrap_err();
        assert!(err.contains("protocol minor"), "{err}");
        assert_eq!(verdict(|c| c.set_protocol_minor(2), &want), Ok(()));
    }
}

#[cfg(all(test, feature = "rpc"))]
mod server_tests {
    use super::*;
    use crate::Potentials_capnp::potential;
    use capnp::capability::Promise;

    /// Predates getCapabilities: relies on the generated default, UNIMPLEMENTED.
    struct Old;
    impl potential::Server for Old {}

    struct Current;
    impl potential::Server for Current {
        fn get_capabilities(
            &mut self,
            _p: potential::GetCapabilitiesParams,
            mut r: potential::GetCapabilitiesResults,
        ) -> Promise<(), capnp::Error> {
            describe_server(r.get().init_capabilities());
            Promise::ok(())
        }
    }

    struct OtherFamily;
    impl potential::Server for OtherFamily {
        fn get_capabilities(
            &mut self,
            _p: potential::GetCapabilitiesParams,
            mut r: potential::GetCapabilitiesResults,
        ) -> Promise<(), capnp::Error> {
            let mut caps = r.get().init_capabilities();
            describe_server(caps.reborrow());
            caps.set_protocol_family("other.family");
            Promise::ok(())
        }
    }

    fn verdict<S: potential::Server + 'static>(server: S) -> Result<(), String> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let local = tokio::task::LocalSet::new();
        local.block_on(&rt, async {
            let client: potential::Client = capnp_rpc::new_client(server);
            check_server(&client, &Expectation::default()).await
        })
    }

    #[test]
    fn current_server_is_accepted() {
        assert_eq!(verdict(Current), Ok(()));
    }

    #[test]
    fn server_without_get_capabilities_is_refused_naming_the_call() {
        let err = verdict(Old).unwrap_err();
        assert!(err.contains("Potential.getCapabilities"), "{err}");
        assert!(err.contains("upgrade the server"), "{err}");
    }

    #[test]
    fn server_with_another_protocol_family_is_refused() {
        let err = verdict(OtherFamily).unwrap_err();
        assert!(err.contains("server refused: protocol family"), "{err}");
    }
}
