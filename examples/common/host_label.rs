// Copyright (c) anneal developers

//! The host column of a board row: `bench` as is, any other hostname as a
//! salted digest. Same rule and salt as `host_label` in
//! `scripts/machine_provenance.py`.

use sha2::{Digest, Sha256};

const SALT: &[u8] = b"anneal-bench-host-label-v1\0";
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

#[cfg(test)]
mod tests {
    use super::host_label;

    #[test]
    fn the_sandbox_name_is_kept_and_any_other_name_is_a_salted_digest() {
        assert_eq!(host_label("bench").as_deref(), Some("bench"));
        assert_eq!(
            host_label("elja-compute-17").as_deref(),
            Some("host-99fd2f34271ff224")
        );
        assert_eq!(host_label(""), None);
    }
}
