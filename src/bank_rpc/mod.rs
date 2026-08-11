//! Cap'n Proto fabric for a shared CSA bank and packing superbasin.
//!
//! Chains are separate processes. They offer quenched members, deposit
//! bias on unit mean-SOAP packings, and start from the bank. The
//! server serialises admission so Dcut and the first bank stay one
//! object. IRA/SOFI remain on the hop; this wire carries known
//! packings.

pub mod client;
pub mod server;

pub use client::BankClient;
pub use server::serve;
