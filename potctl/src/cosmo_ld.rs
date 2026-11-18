//! Thin rustc linker entry (`potctl-cosmo-ld`). rustc passes only link argv.

#[allow(
    dead_code,
    reason = "host helpers in this file are called by the potctl bin, not the linker"
)]
#[path = "cosmo.rs"]
mod cosmo;

use std::env;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    match cosmo::link_main(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: potctl-cosmo-ld: {e}");
            ExitCode::FAILURE
        }
    }
}
