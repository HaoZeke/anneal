//! Declarative campaign configuration.
//!
//! A coordinated campaign is configured by roughly seventy `CATALOG_*`
//! environment variables, and an absent variable silently selects a
//! default. That shape cannot fail loudly: the c48 LJ75 campaign ran
//! for hours with its recognition-refund channel dark because one
//! export was missing from an sbatch script, and nothing anywhere was
//! obliged to say so.
//!
//! This module is the replacement surface. A campaign is a TOML file:
//! unknown keys are rejected, the coordination channels are a required
//! block with no defaults (every channel is an explicit `true` or
//! `false`, so "forgot to mention it" is a parse error, not a silent
//! off), and the resolved configuration is echoed as a banner so every
//! worker log opens by saying which channels it runs.
//!
//! Loading bridges onto the existing environment-variable transport:
//! [`CampaignConfig::bootstrap`] reads the file named by
//! `CATALOG_CONFIG`, validates it, refuses to run if the environment
//! already carries conflicting `CATALOG_*` settings, and then exports
//! the file's values into the environment before any thread spawns.
//! Call sites keep reading the variables they always read; the file is
//! the single place a human writes them.

use std::collections::BTreeMap;
use std::fmt::Write as _;

use serde::Deserialize;

/// The coordination channels. Every field is required: a campaign
/// file that does not mention a channel does not parse, which is the
/// entire point.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Channels {
    /// The shared recognition screen: chains offer validated minima
    /// and a trial descending into any chain's known basin takes the
    /// stored minimum instead of finishing the descent
    /// (`CATALOG_SHARED_SCREEN`).
    pub shared_screen: bool,
    /// Continuous shared bias: the quench feels the ensemble's visited
    /// cloud at every step rather than only at steering decisions
    /// (`CATALOG_SHARED_BIAS`). Distinct from the always-on reference
    /// pullback that rec+catalog mode applies regardless.
    pub shared_bias: bool,
    /// Entropic occupancy bias on selection (`CATALOG_ENTROPIC_BIAS`).
    pub entropic_bias: bool,
    /// Histogram screen on arrivals (`CATALOG_HISTO_SCREEN`).
    pub histo_screen: bool,
    /// The per-chain seam ladder: banked doorway states with
    /// epsilon-greedy restarts after stagnation (`CATALOG_SEAM_LADDER`).
    pub seam_ladder: bool,
    /// The ensemble frontier exchange: raw doorway states shared
    /// through the coordinator so every chain's ladder holds the
    /// union's progress (`CATALOG_FRONTIER_EXCHANGE`). Requires the
    /// seam ladder, which is where arrivals land.
    pub frontier_exchange: bool,
}

/// The ensemble shape. Required: a campaign that does not say how many
/// chains it runs is not a campaign.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ensemble {
    /// Chain count (`CATALOG_REPLICAS`).
    pub replicas: u32,
    /// Wave width for selection sync (`CATALOG_WAVE`).
    pub wave: u32,
    /// Hops per slice between waves (`CATALOG_SLICE`).
    pub slice: u32,
    /// Hop cap per chain (`CATALOG_MAX_HOPS`).
    pub max_hops: u64,
    /// Census interval in evaluations (`CATALOG_POPULATION_INTERVAL`).
    pub population_interval: u64,
}

/// A validated campaign definition.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CampaignConfig {
    /// Campaign name (`CATALOG_CAMPAIGN`); the output namespace.
    pub campaign: String,
    /// The coordination channels, all explicit.
    pub channels: Channels,
    /// The ensemble shape.
    pub ensemble: Ensemble,
    /// Everything else the run wants set, verbatim `CATALOG_*` (or
    /// other) variables. Optional, but still schema'd: keys must be
    /// SCREAMING_SNAKE_CASE, values are strings.
    #[serde(default)]
    pub extra: BTreeMap<String, String>,
}

/// A loading failure. The message is the user interface: it names the
/// file, the key, and what to do.
#[derive(Debug)]
pub struct CampaignError(pub String);

impl std::fmt::Display for CampaignError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CampaignError {}

impl CampaignConfig {
    /// Parse and validate a campaign file's contents.
    pub fn parse(text: &str) -> Result<Self, CampaignError> {
        let cfg: CampaignConfig = toml::from_str(text)
            .map_err(|e| CampaignError(format!("campaign file rejected: {e}")))?;
        for key in cfg.extra.keys() {
            let ok = !key.is_empty()
                && key
                    .chars()
                    .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_');
            if !ok {
                return Err(CampaignError(format!(
                    "extra key {key:?} is not an environment variable name \
                     (SCREAMING_SNAKE_CASE)"
                )));
            }
        }
        if cfg.campaign.is_empty() {
            return Err(CampaignError("campaign name is empty".into()));
        }
        if cfg.channels.frontier_exchange && !cfg.channels.seam_ladder {
            return Err(CampaignError(
                "frontier_exchange requires seam_ladder: arrivals land in \
                 the seam bank, and without it the channel is dead mail"
                    .into(),
            ));
        }
        Ok(cfg)
    }

    /// The environment this campaign resolves to, as pairs. Boolean
    /// channels export "1" when on and stay unset when off, matching
    /// how every existing read site tests them; the banner still names
    /// the off channels explicitly.
    pub fn env_pairs(&self) -> Vec<(String, String)> {
        let mut pairs: Vec<(String, String)> = Vec::new();
        pairs.push(("CATALOG_CAMPAIGN".into(), self.campaign.clone()));
        pairs.push((
            "CATALOG_REPLICAS".into(),
            self.ensemble.replicas.to_string(),
        ));
        pairs.push(("CATALOG_WAVE".into(), self.ensemble.wave.to_string()));
        pairs.push(("CATALOG_SLICE".into(), self.ensemble.slice.to_string()));
        pairs.push((
            "CATALOG_MAX_HOPS".into(),
            self.ensemble.max_hops.to_string(),
        ));
        pairs.push((
            "CATALOG_POPULATION_INTERVAL".into(),
            self.ensemble.population_interval.to_string(),
        ));
        for (name, on) in self.channel_states() {
            if on {
                pairs.push((name.to_string(), "1".into()));
            }
        }
        for (k, v) in &self.extra {
            pairs.push((k.clone(), v.clone()));
        }
        pairs
    }

    /// Resolve the campaign to tab-separated environment records for a
    /// process launcher. Values containing record delimiters are rejected so
    /// a shell can read one validated name/value pair per line without eval.
    pub fn launcher_environment(&self) -> Result<String, CampaignError> {
        let mut output = String::new();
        for (name, value) in self.env_pairs() {
            if name
                .chars()
                .chain(value.chars())
                .any(|character| matches!(character, '\t' | '\n' | '\r' | '\0'))
            {
                return Err(CampaignError(format!(
                    "campaign environment value {name} contains a record delimiter"
                )));
            }
            let _ = writeln!(output, "{name}\t{value}");
        }
        Ok(output)
    }

    /// Channel names and their explicit states.
    pub fn channel_states(&self) -> [(&'static str, bool); 6] {
        [
            ("CATALOG_SHARED_SCREEN", self.channels.shared_screen),
            ("CATALOG_SHARED_BIAS", self.channels.shared_bias),
            ("CATALOG_ENTROPIC_BIAS", self.channels.entropic_bias),
            ("CATALOG_HISTO_SCREEN", self.channels.histo_screen),
            ("CATALOG_SEAM_LADDER", self.channels.seam_ladder),
            ("CATALOG_FRONTIER_EXCHANGE", self.channels.frontier_exchange),
        ]
    }

    /// The launch banner: one line per channel, every state explicit.
    /// Printed by whoever bootstraps, so every log opens with the
    /// channel table and a dark channel is visible in the first
    /// screenful, not at hour three.
    pub fn banner(&self) -> String {
        let mut out = String::new();
        let _ = writeln!(out, "campaign {}", self.campaign);
        let _ = writeln!(
            out,
            "  ensemble: replicas {} wave {} slice {} max_hops {} census {}",
            self.ensemble.replicas,
            self.ensemble.wave,
            self.ensemble.slice,
            self.ensemble.max_hops,
            self.ensemble.population_interval
        );
        for (name, on) in self.channel_states() {
            let _ = writeln!(out, "  channel {name}: {}", if on { "ON" } else { "off" });
        }
        for (k, v) in &self.extra {
            let _ = writeln!(out, "  extra {k}={v}");
        }
        out
    }

    /// Environment variables this config would export that are
    /// already set to a different value: the two-sources-of-truth
    /// hazard, reported instead of resolved.
    pub fn collisions(&self) -> Vec<String> {
        let mut out = Vec::new();
        for (k, v) in self.env_pairs() {
            if let Ok(existing) = std::env::var(&k) {
                if existing != v {
                    out.push(format!(
                        "{k} is {existing:?} in the environment but {v:?} in the campaign file"
                    ));
                }
            }
        }
        // A channel the file turns OFF that the environment turns on
        // is also a collision, even though off exports nothing.
        for (name, on) in self.channel_states() {
            if !on {
                if let Ok(existing) = std::env::var(name) {
                    if existing == "1" {
                        out.push(format!(
                            "{name} is on in the environment but off in the campaign file"
                        ));
                    }
                }
            }
        }
        out
    }

    /// Load the campaign named by `CATALOG_CONFIG`, validate it,
    /// refuse on collisions, export it into the process environment,
    /// and return it (the caller prints the banner). Returns `None`
    /// when `CATALOG_CONFIG` is unset: legacy env-var operation.
    ///
    /// # Safety contract
    /// Must run before any thread is spawned; it writes the process
    /// environment.
    pub fn bootstrap() -> Result<Option<Self>, CampaignError> {
        let path = match std::env::var("CATALOG_CONFIG") {
            Ok(p) if !p.is_empty() => p,
            _ => return Ok(None),
        };
        let text = std::fs::read_to_string(&path)
            .map_err(|e| CampaignError(format!("cannot read campaign file {path}: {e}")))?;
        let cfg = Self::parse(&text)?;
        let collisions = cfg.collisions();
        if !collisions.is_empty() {
            return Err(CampaignError(format!(
                "campaign file {path} conflicts with the environment; \
                 unset the variables or fix the file:\n  {}",
                collisions.join("\n  ")
            )));
        }
        for (k, v) in cfg.env_pairs() {
            // Sound per the bootstrap contract: single-threaded, at
            // process start, before any spawn.
            unsafe { std::env::set_var(&k, &v) };
        }
        Ok(Some(cfg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GOOD: &str = r#"
campaign = "lj75-occ-c48"

[channels]
shared_screen = true
shared_bias = true
entropic_bias = false
histo_screen = false
seam_ladder = true
frontier_exchange = true

[ensemble]
replicas = 48
wave = 48
slice = 500
max_hops = 60000
population_interval = 50000

[extra]
SEED_OFFSET_BASE = "9900000"
"#;

    #[test]
    fn a_complete_campaign_parses_and_banners_every_channel() {
        let cfg = CampaignConfig::parse(GOOD).unwrap();
        let banner = cfg.banner();
        assert!(banner.contains("CATALOG_SHARED_SCREEN: ON"));
        assert!(banner.contains("CATALOG_ENTROPIC_BIAS: off"));
        let pairs = cfg.env_pairs();
        assert!(
            pairs
                .iter()
                .any(|(k, v)| k == "CATALOG_SHARED_SCREEN" && v == "1")
        );
        assert!(!pairs.iter().any(|(k, _)| k == "CATALOG_ENTROPIC_BIAS"));
        assert!(
            pairs
                .iter()
                .any(|(k, v)| k == "SEED_OFFSET_BASE" && v == "9900000")
        );
    }

    #[test]
    fn a_missing_channel_is_a_parse_error_not_a_silent_default() {
        let text = GOOD.replace("shared_screen = true\n", "");
        let err = CampaignConfig::parse(&text).unwrap_err();
        assert!(err.0.contains("shared_screen"), "{}", err.0);
    }

    #[test]
    fn an_unknown_key_is_rejected() {
        let text = GOOD.replace("slice = 500", "slice = 500\nslise = 400");
        assert!(CampaignConfig::parse(&text).is_err());
    }

    #[test]
    fn extra_keys_must_look_like_environment_variables() {
        let text = GOOD.replace("SEED_OFFSET_BASE", "seed_offset_base");
        assert!(CampaignConfig::parse(&text).is_err());
    }

    #[test]
    fn launcher_environment_uses_the_canonical_resolved_values() {
        let cfg = CampaignConfig::parse(GOOD).unwrap();
        let launcher = cfg.launcher_environment().unwrap();
        let lines = launcher.lines().collect::<Vec<_>>();
        assert!(lines.contains(&"CATALOG_CAMPAIGN\tlj75-occ-c48"));
        assert!(lines.contains(&"CATALOG_REPLICAS\t48"));
        assert!(lines.contains(&"CATALOG_WAVE\t48"));
        assert!(lines.contains(&"CATALOG_SHARED_SCREEN\t1"));
        assert!(lines.contains(&"SEED_OFFSET_BASE\t9900000"));
        assert!(
            !lines
                .iter()
                .any(|line| line.contains("CATALOG_ENTROPIC_BIAS"))
        );
    }

    #[test]
    fn launcher_environment_rejects_record_delimiters() {
        let mut cfg = CampaignConfig::parse(GOOD).unwrap();
        cfg.extra
            .insert("BAD_VALUE".to_owned(), "first\nsecond".to_owned());
        let error = cfg.launcher_environment().unwrap_err();
        assert!(error.0.contains("BAD_VALUE"), "{}", error.0);
    }
}
