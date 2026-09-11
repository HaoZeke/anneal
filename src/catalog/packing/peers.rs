//! Thread-local live occupancy, distinct from archived packing history.

use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

thread_local! {
    static PEERS: RefCell<Option<Vec<Vec<f64>>>> = const { RefCell::new(None) };
}

/// A chain's live-peer view, restored when the chain leaves this scope.
///
/// An enabled empty view means no peers are known. A disabled view leaves
/// history-driven proposals available without treating history as a census.
/// The scope belongs to its creating thread, like the packing archive.
pub struct PackingPeerScope {
    previous: Option<Vec<Vec<f64>>>,
    thread: PhantomData<Rc<()>>,
}

impl PackingPeerScope {
    /// Start with an empty live view when enabled, or with no live view.
    pub fn new(enabled: bool) -> Self {
        Self {
            previous: PEERS.with(|slot| slot.replace(enabled.then(Vec::new))),
            thread: PhantomData,
        }
    }
}

impl Drop for PackingPeerScope {
    fn drop(&mut self) {
        PEERS.with(|slot| *slot.borrow_mut() = self.previous.take());
    }
}

/// Replace the latest positions of the admitted peers without changing history.
///
/// Each entry represents one peer. Distinct peers at identical coordinates
/// retain their multiplicity. The transport owns producer identity and which
/// peers the topology admits; an empty vector clears live occupancy.
pub fn set_packing_peers(peers: Vec<Vec<f64>>) {
    PEERS.with(|slot| *slot.borrow_mut() = Some(peers));
}

/// Latest peers within the current state's packing confidence bound.
///
/// `None` means no live view is installed; `Some([])` means an installed view
/// has no nearby peers. Classification uses the proposal's occupied geometry,
/// so a checkpoint's nearby flag cannot push a chain out of a different family.
pub fn nearby_packing_peers(here: &[f64]) -> Option<Vec<Vec<f64>>> {
    PEERS.with(|slot| {
        slot.borrow().as_ref().map(|peers| {
            peers
                .iter()
                .filter(|peer| peer.iter().all(|value| value.is_finite()))
                .filter(|peer| super::nearby_packing(here, peer))
                .cloned()
                .collect()
        })
    })
}
