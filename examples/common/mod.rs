// Each example binary compiles this tree and calls only the helpers it needs.
#[allow(dead_code)]
pub(crate) mod efficiency;
#[allow(dead_code)]
pub(crate) mod ensemble_report;
#[cfg(feature = "rgpot-ex")]
pub(crate) mod rgpot_eindir;
#[allow(dead_code)]
pub(crate) mod slab;
