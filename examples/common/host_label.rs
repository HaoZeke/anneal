// Copyright (c) anneal developers

//! The host and boot columns of a board row: `bench` as is and any other
//! hostname as a salted digest, and the kernel boot id as a salted digest.
//! Same rules and salts as `host_label` and `boot_label` in
//! `scripts/machine_provenance.py`.
//!
//! This file is the only Rust code that reads the host name or the boot id.
//! Examples call `read_host_label` and `read_boot_label` and write what they
//! return, or [`UNKNOWN`] when the kernel gives nothing.

use sha2::{Digest, Sha256};

const SALT: &[u8] = b"anneal-bench-host-label-v1\0";
const BOOT_SALT: &[u8] = b"anneal-bench-boot-label-v1\0";
const SANDBOX_HOSTNAME: &str = "bench";
const HOSTNAME_PATH: &str = "/proc/sys/kernel/hostname";

/// Names the file the boot label stands for. Reading it is done here only.
#[allow(dead_code)]
pub const BOOT_ID_PATH: &str = "/proc/sys/kernel/random/boot_id";

/// Written where a label could not be read. Never a digest of a stand-in.
#[allow(dead_code)]
pub const UNKNOWN: &str = "unknown";

pub fn host_label(hostname: &str) -> Option<String> {
    if hostname.is_empty() {
        return None;
    }
    if hostname == SANDBOX_HOSTNAME {
        return Some(hostname.to_string());
    }
    let digest = Sha256::new()
        .chain_update(SALT)
        .chain_update(hostname.as_bytes())
        .finalize();
    let hex: String = digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Some(format!("host-{hex}"))
}

fn first_line(path: &str) -> Option<String> {
    let text = std::fs::read_to_string(path).ok()?;
    Some(text.lines().next()?.trim().to_string())
}

/// The host label of this machine, or `None` when the kernel name is unreadable.
#[allow(dead_code)]
pub fn read_host_label() -> Option<String> {
    host_label(&first_line(HOSTNAME_PATH)?)
}

/// The boot label of this machine, or `None` when there is no boot id.
#[allow(dead_code)]
pub fn read_boot_label() -> Option<String> {
    boot_label(&first_line(BOOT_ID_PATH)?)
}

/// A `host:port` endpoint with its host as a host label. The port and any
/// scheme stay. An endpoint with no host gives [`UNKNOWN`], never the input.
#[allow(dead_code)]
pub fn endpoint_label(endpoint: &str) -> String {
    let (scheme, rest) = match endpoint.split_once("://") {
        Some((scheme, rest)) => (Some(scheme), rest),
        None => (None, endpoint),
    };
    let (rest, tail) = match rest.find('/') {
        Some(at) => rest.split_at(at),
        None => (rest, ""),
    };
    let (host, port) = match rest.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() && port.bytes().all(|b| b.is_ascii_digit()) => {
            (host, Some(port))
        }
        _ => (rest, None),
    };
    let Some(label) = host_label(host.trim_matches(['[', ']'])) else {
        return UNKNOWN.to_string();
    };
    let mut out = String::new();
    if let Some(scheme) = scheme {
        out.push_str(scheme);
        out.push_str("://");
    }
    out.push_str(&label);
    if let Some(port) = port {
        out.push(':');
        out.push_str(port);
    }
    out.push_str(tail);
    out
}

#[allow(dead_code)]
pub fn boot_label(boot_id: &str) -> Option<String> {
    let boot_id = boot_id.trim();
    if boot_id.is_empty() {
        return None;
    }
    let digest = Sha256::new()
        .chain_update(BOOT_SALT)
        .chain_update(boot_id.as_bytes())
        .finalize();
    let hex: String = digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    Some(format!("boot-{hex}"))
}

#[cfg(test)]
mod tests {
    use super::{UNKNOWN, boot_label, endpoint_label, host_label};

    #[test]
    fn the_sandbox_name_is_kept_and_any_other_name_is_a_salted_digest() {
        assert_eq!(host_label("bench").as_deref(), Some("bench"));
        assert_eq!(
            host_label("elja-compute-17").as_deref(),
            Some("host-99fd2f34271ff224")
        );
        assert_eq!(host_label(""), None);
    }

    #[test]
    fn the_boot_id_is_only_ever_a_salted_digest() {
        assert_eq!(
            boot_label("00000000-1111-2222-3333-444444444444\n").as_deref(),
            Some("boot-84eaa8c774326c3e")
        );
        assert_eq!(boot_label(" \n"), None);
    }

    #[test]
    fn an_endpoint_keeps_its_port_and_scheme_and_loses_its_host() {
        assert_eq!(
            endpoint_label("elja-compute-17:7426"),
            "host-99fd2f34271ff224:7426"
        );
        assert_eq!(
            endpoint_label("tcp://elja-compute-17:9000/x"),
            "tcp://host-99fd2f34271ff224:9000/x"
        );
        assert_eq!(endpoint_label("bench:5"), "bench:5");
        assert_eq!(
            endpoint_label("elja-compute-17:7426"),
            "host-99fd2f34271ff224:7426"
        );
        assert_eq!(endpoint_label(":80"), UNKNOWN);
        assert_eq!(endpoint_label("ipc:///tmp/bank"), UNKNOWN);
    }
}
