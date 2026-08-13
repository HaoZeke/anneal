//! Print live CSA bank size, Dcut, member energies, and SOAP well heights.
//!
//! Usage: bank_peek [host:port]

fn main() {
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:7426".into());
    let mut c = anneal_core::bank_rpc::BankClient::connect(&addr)
        .unwrap_or_else(|e| panic!("connect {addr}: {e}"));
    let s = c.snapshot().unwrap_or_else(|e| panic!("snapshot: {e}"));
    println!("bank {addr}");
    println!(
        "  members {}  dcut {:.6}  wells {}",
        s.size,
        s.dcut,
        s.wells.len()
    );
    if !s.energies.is_empty() {
        let mut e = s.energies.clone();
        e.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "  E min {:.6}  max {:.6}  n {}",
            e[0],
            e[e.len() - 1],
            e.len()
        );
        for (i, v) in e.iter().take(12).enumerate() {
            println!("    [{i}] {v:.6}");
        }
        if e.len() > 12 {
            println!("    ... {} more", e.len() - 12);
        }
    }
    if !s.wells.is_empty() {
        let mut h: Vec<f64> = s.wells.iter().map(|(_, h)| *h).collect();
        h.sort_by(|a, b| b.partial_cmp(a).unwrap());
        println!("  well heights (tallest first):");
        for (i, v) in h.iter().take(12).enumerate() {
            println!("    [{i}] {v:.4}");
        }
        if h.len() > 12 {
            println!("    ... {} more", h.len() - 12);
        }
    }
}
