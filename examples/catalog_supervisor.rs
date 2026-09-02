//! Launch catalog workers for successive-halving spawn requests.
//!
//!     catalog_supervisor ENDPOINT CAMPAIGN ENSEMBLE SIGNATURE_HEX -- WORKER [ARGS...]
//!
//! Polls observer status once a second. For each pending spawn the supervisor
//! starts `WORKER` with `CATALOG_REPLICA` set to the next free identity and
//! `SEED_OFFSET` a fresh offset. The supervisor does not Attach; the worker
//! does.

#[cfg(feature = "bank-rpc")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anneal_core::catalog_rpc::CatalogIdentity;
    use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};
    use std::collections::BTreeSet;
    use std::process::{Command, Stdio};

    let arguments: Vec<String> = std::env::args().collect();
    let Some(separator) = arguments.iter().position(|argument| argument == "--") else {
        eprintln!(
            "usage: catalog_supervisor ENDPOINT CAMPAIGN ENSEMBLE SIGNATURE_HEX -- WORKER [ARGS...]"
        );
        std::process::exit(2);
    };
    let header = &arguments[1..separator];
    let worker = &arguments[separator + 1..];
    let [endpoint, campaign, ensemble, signature_hex] = header else {
        eprintln!(
            "usage: catalog_supervisor ENDPOINT CAMPAIGN ENSEMBLE SIGNATURE_HEX -- WORKER [ARGS...]"
        );
        std::process::exit(2);
    };
    if worker.is_empty() {
        eprintln!("catalog_supervisor requires a worker command after --");
        std::process::exit(2);
    }
    let address = endpoint.parse()?;
    let identity = CatalogIdentity {
        campaign: campaign.clone(),
        ensemble: ensemble.clone(),
        replica: u32::MAX,
        signature_digest: parse_signature_digest(signature_hex)?,
    };
    let mut client = CatalogClient::connect(address, identity, ClientConfig::default())?;
    let mut sequence = 1u64;
    let mut in_flight = BTreeSet::new();
    loop {
        let status = client.observer_status(sequence)?;
        sequence += 1;
        for replica in &status.live_replicas {
            in_flight.remove(replica);
        }
        let pending = status
            .spawn_requested
            .saturating_sub(u32::try_from(in_flight.len()).unwrap_or(u32::MAX));
        let mut next_id = status
            .live_replicas
            .iter()
            .copied()
            .chain(in_flight.iter().copied())
            .max()
            .map(|id| id.saturating_add(1))
            .unwrap_or(0);
        for _ in 0..pending {
            while in_flight.contains(&next_id) || status.live_replicas.contains(&next_id) {
                next_id = next_id.saturating_add(1);
            }
            let seed_offset = u64::from(next_id).saturating_mul(1_000_003);
            let mut command = Command::new(&worker[0]);
            command.args(&worker[1..]);
            command.env("CATALOG_REPLICA", next_id.to_string());
            command.env("SEED_OFFSET", seed_offset.to_string());
            command.stdin(Stdio::null());
            command.spawn()?;
            in_flight.insert(next_id);
            next_id = next_id.saturating_add(1);
        }
        std::thread::sleep(std::time::Duration::from_secs(1));
    }
}

#[cfg(feature = "bank-rpc")]
fn parse_signature_digest(value: &str) -> Result<[u8; 32], String> {
    let encoded = value.as_bytes();
    if encoded.len() != 64 {
        return Err("SIGNATURE_HEX must contain exactly 64 hexadecimal digits".to_owned());
    }
    let mut digest = [0u8; 32];
    for (index, byte) in digest.iter_mut().enumerate() {
        let offset = index * 2;
        *byte = (hex_nibble(encoded[offset])? << 4) | hex_nibble(encoded[offset + 1])?;
    }
    Ok(digest)
}

#[cfg(feature = "bank-rpc")]
fn hex_nibble(value: u8) -> Result<u8, String> {
    match value {
        b'0'..=b'9' => Ok(value - b'0'),
        b'a'..=b'f' => Ok(value - b'a' + 10),
        b'A'..=b'F' => Ok(value - b'A' + 10),
        _ => Err("SIGNATURE_HEX contains a non-hexadecimal digit".to_owned()),
    }
}

#[cfg(not(feature = "bank-rpc"))]
fn main() {
    eprintln!("catalog_supervisor requires the bank-rpc feature");
    std::process::exit(2);
}
