//! Thin rustc linker entry (`potctl-cosmo-ld`). rustc passes only link argv.

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
