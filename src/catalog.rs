//! Search catalogs with explicit event and descriptor-basin identities.

pub mod basin;
pub mod event;
pub mod signature;

pub use basin::BasinCatalog;
pub use event::{Event, EventCatalog, TopologyRecord};
pub use signature::{DescriptorSignature, EngineSignature, SignatureDigest, SystemSignature};

/// Compatibility name for the local-topology event catalog.
#[deprecated(since = "0.9.0", note = "use EventCatalog")]
pub type Catalog = EventCatalog;
