//! Live status of a running catalog coordinator, one JSON line per poll.
//!
//! The coordinator serves its ensemble's replicas over Cap'n Proto; this
//! asks it the one question an outsider may: how far along is everyone.
//! No replica slot or journal entry is needed, but the observer must present
//! the exact system-signature digest so status cannot cross PESes.
//!
//!     catalog_status 127.0.0.1:40701 my-campaign my-ensemble SIGNATURE_HEX [seconds]
//!
//! With a poll interval the loop runs until the coordinator goes away;
//! without one it prints a single line and exits. The status crosses the
//! wire as Cap'n Proto either way; the JSON here is a rendering at the
//! terminal edge, and `--raw` skips it, writing each framed reply to
//! stdout for `capnp decode` or any other Cap'n Proto consumer.

#[cfg(feature = "bank-rpc")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use anneal_core::catalog_rpc::CatalogIdentity;
    use anneal_core::catalog_rpc::client::{CatalogClient, ClientConfig};

    let mut arguments: Vec<String> = std::env::args().collect();
    let raw = arguments.iter().any(|argument| argument == "--raw");
    arguments.retain(|argument| argument != "--raw");
    let [_, endpoint, campaign, ensemble, signature_hex, rest @ ..] = arguments.as_slice() else {
        eprintln!("usage: catalog_status ENDPOINT CAMPAIGN ENSEMBLE SIGNATURE_HEX [POLL_SECONDS]");
        std::process::exit(2);
    };
    let interval = rest
        .first()
        .map(|value| value.parse::<u64>())
        .transpose()?
        .map(std::time::Duration::from_secs);
    let address = endpoint.parse()?;
    let identity = CatalogIdentity {
        campaign: campaign.clone(),
        ensemble: ensemble.clone(),
        replica: u32::MAX,
        signature_digest: parse_signature_digest(signature_hex)?,
    };
    let mut client = CatalogClient::connect(address, identity, ClientConfig::default())?;
    let mut sequence = 1u64;
    loop {
        if raw {
            use std::io::Write;
            let frame = client.observer_status_frame(sequence)?;
            sequence += 1;
            std::io::stdout().write_all(&frame)?;
            std::io::stdout().flush()?;
            match interval {
                Some(delay) => {
                    std::thread::sleep(delay);
                    continue;
                }
                None => return Ok(()),
            }
        }
        let status = client.observer_status(sequence)?;
        sequence += 1;
        let replicas = status
            .replicas
            .iter()
            .map(|row| {
                format!(
                    "{{\"replica\":{},\"charged\":{},\"best\":{}}}",
                    row.replica,
                    row.charged_work,
                    if row.best_energy.is_finite() {
                        format!("{:.6}", row.best_energy)
                    } else {
                        "null".to_owned()
                    }
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let seam = status.seam.as_ref().map_or_else(
            || "null".to_owned(),
            |seam| {
                format!(
                    "{{\"lambda2\":{:.6},\"conductance\":{:.6},\"left\":{},\"right\":{},\
                     \"left_basin\":{},\"right_basin\":{}}}",
                    seam.algebraic_connectivity,
                    seam.conductance,
                    seam.community_left,
                    seam.community_right,
                    seam.left_basin,
                    seam.right_basin
                )
            },
        );
        println!(
            "{{\"snapshot\":{},\"epoch\":{},\"submitted\":{},\"required\":{},\
             \"census_visits\":{},\"catalog_entries\":{},\"charged\":{},\"budget\":{},\
             \"landscape_basins\":{},\"unique_saddles\":{},\"unique_edges\":{},\
             \"unique_degenerate_rearrangements\":{},\"certified_connections\":{},\
             \"seam\":{},\"replicas\":[{}]}}",
            status.snapshot_version,
            status.open_epoch,
            status.epoch_submitted,
            status.epoch_required,
            status.census_visits,
            status.active_entries,
            status.aggregate_charged,
            status.aggregate_budget,
            status.landscape_basins,
            status.unique_saddles,
            status.unique_edges,
            status.unique_degenerate_rearrangements,
            status.certified_connections,
            seam,
            replicas
        );
        match interval {
            Some(delay) => std::thread::sleep(delay),
            None => return Ok(()),
        }
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
    eprintln!("catalog_status requires the bank-rpc feature");
    std::process::exit(2);
}
