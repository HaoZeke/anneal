//! Live status of a running catalog coordinator, one JSON line per poll.
//!
//! The coordinator serves its ensemble's replicas over Cap'n Proto; this
//! asks it the one question an outsider may: how far along is everyone.
//! No replica identity, no signature, no journal entry, so it can watch
//! any ensemble whose campaign and ensemble names it knows.
//!
//!     catalog_status 127.0.0.1:40701 my-campaign my-ensemble [seconds]
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
    let [_, endpoint, campaign, ensemble, rest @ ..] = arguments.as_slice() else {
        eprintln!("usage: catalog_status ENDPOINT CAMPAIGN ENSEMBLE [POLL_SECONDS]");
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
        signature_digest: [0; 32],
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
        println!(
            "{{\"snapshot\":{},\"epoch\":{},\"submitted\":{},\"required\":{},\
             \"census_visits\":{},\"catalog_entries\":{},\"charged\":{},\"budget\":{},\
             \"replicas\":[{}]}}",
            status.snapshot_version,
            status.open_epoch,
            status.epoch_submitted,
            status.epoch_required,
            status.census_visits,
            status.active_entries,
            status.aggregate_charged,
            status.aggregate_budget,
            replicas
        );
        match interval {
            Some(delay) => std::thread::sleep(delay),
            None => return Ok(()),
        }
    }
}

#[cfg(not(feature = "bank-rpc"))]
fn main() {
    eprintln!("catalog_status requires the bank-rpc feature");
    std::process::exit(2);
}
