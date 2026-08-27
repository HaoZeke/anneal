//! Resolve one declarative campaign into launcher environment records.
//!
//!     campaign_env CAMPAIGN.toml

use anneal_core::campaign::CampaignConfig;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: campaign_env CAMPAIGN.toml")?;
    let text = std::fs::read_to_string(&path)?;
    let campaign = CampaignConfig::parse(&text)?;
    print!("{}", campaign.launcher_environment()?);
    Ok(())
}
