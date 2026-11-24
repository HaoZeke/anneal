// Copyright (c) anneal developers

//! The host and boot columns of a board row: `bench` as is and any other
//! hostname as a salted digest, and the kernel boot id as a salted digest.
//! Same rules and salts as `host_label` and `boot_label` in
//! `scripts/machine_provenance.py`.

use sha2::{Digest, Sha256};

const SALT: &[u8] = b"anneal-bench-host-label-v1\0";
const BOOT_SALT: &[u8] = b"anneal-bench-boot-label-v1\0";
const SANDBOX_HOSTNAME: &str = "bench";

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
    use super::{boot_label, host_label};

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
}
