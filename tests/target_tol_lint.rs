//! Repository lint: target-energy hit comparisons must go through `energy_hits`.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

const ALLOWED_RUNTIME_LITERALS: &[(&str, &str)] = &[
    ("scripts/occat_h2h/stub.S", "tol:         .double 0.001"),
    (
        "bench/lj104-lattice/v2/h2h_cap.f90",
        "LOGICAL FUNCTION ENERGY_HITS",
    ),
];

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn no_raw_energy_hit_comparisons_outside_energy_hits() {
    let status = Command::new("python3")
        .arg("scripts/target_tol_lint_scan.py")
        .current_dir(root())
        .status()
        .expect("python3 lint scan");
    assert!(
        status.success(),
        "target_tol_lint_scan.py reported violations (see stderr)"
    );
}

#[test]
fn prefix_fixtures_must_fail_lint_rules() {
    let fixtures = root().join("tests/target_tol_prefix_fixtures.txt");
    let script = format!(
        r#"
import sys
from pathlib import Path
ROOT = Path({:?})
sys.path.insert(0, str(ROOT / "scripts"))
from target_tol_lint_rules import fixture_must_violate
text = Path({:?}).read_text()
failures = []
for line in text.splitlines():
    line = line.strip()
    if not line or line.startswith(chr(35)):
        continue
    if not fixture_must_violate(line):
        failures.append(line)
if failures:
    print("fixtures expected to violate:", file=sys.stderr)
    print('\\n'.join(failures), file=sys.stderr)
    sys.exit(1)
"#,
        root(),
        fixtures,
    );
    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("fixture check");
    assert!(status.success(), "prefix fixture check failed");
}

#[test]
fn lint_detects_injected_experiments_probe() {
    let status = Command::new("python3")
        .arg("-c")
        .arg(
            r#"
import sys
from pathlib import Path
ROOT = Path(".").resolve()
sys.path.insert(0, str(ROOT / "scripts"))
from target_tol_lint_rules import classify_line
assert classify_line(
    "experiments/_target_tol_lint_probe.py",
    "    return x <= thresh",
    "py",
) is not None
"#,
        )
        .current_dir(root())
        .status()
        .expect("probe classify");
    assert!(status.success(), "probe line must violate lint rules");
}

#[test]
fn runtime_allowlisted_literals_match_default_hit_tol() {
    let root = root();
    let script = format!(
        r#"
import sys
from pathlib import Path
ROOT = Path("{root}")
sys.path.insert(0, str(ROOT / "scripts"))
from target_tol import DEFAULT_HIT_TOL
assert float(DEFAULT_HIT_TOL) == 1e-3
"#,
        root = root.display()
    );
    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status()
        .expect("DEFAULT_HIT_TOL");
    assert!(status.success());
    for (rel, needle) in ALLOWED_RUNTIME_LITERALS {
        let path = root.join(rel);
        let text = fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {rel}: {e}"));
        assert!(
            text.contains(needle),
            "{rel} must still contain allowlisted literal marker `{needle}`"
        );
    }
}
