//! Shared CSA bank + packing superbasin over Cap'n Proto (TCP).
//!
//! Usage: bank_server [host:port] [capacity]

fn main() {
    let mut args = std::env::args().skip(1);
    let addr = args.next().unwrap_or_else(|| "0.0.0.0:7424".into());
    let capacity = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    anneal_core::bank_rpc::serve(addr, capacity).expect("bank server");
}
