//! Caller-owned stop flag for the thread that runs a search.
//!
//! A harness stops evaluation by raising an [`AtomicBool`]. Binding that
//! flag here makes every driver on this thread see it. A [`crate::methods::cluster_hopping::Ledger`]
//! also carries the flag, because a worker thread does not inherit
//! thread-local state.

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

thread_local! {
    static BOUND: RefCell<Option<Arc<AtomicBool>>> = const { RefCell::new(None) };
}

/// Installs `flag` as this thread's caller stop until dropped.
///
/// Dropping the guard restores whatever was bound before, including the
/// absence of a flag.
pub struct Bind {
    previous: Option<Arc<AtomicBool>>,
}

impl Bind {
    /// Binds `flag` for the rest of this thread's search call.
    pub fn new(flag: Arc<AtomicBool>) -> Self {
        let previous = BOUND.with(|slot| slot.replace(Some(flag)));
        Self { previous }
    }
}

impl Drop for Bind {
    fn drop(&mut self) {
        let previous = self.previous.take();
        BOUND.with(|slot| {
            *slot.borrow_mut() = previous;
        });
    }
}

/// The flag bound on this thread, if a search installed one.
pub fn bound() -> Option<Arc<AtomicBool>> {
    BOUND.with(|slot| slot.borrow().clone())
}

/// Whether the bound caller stop is raised.
///
/// False when nothing is bound. A down flag does not change control flow
/// beyond this load.
#[inline]
pub fn raised() -> bool {
    BOUND.with(|slot| {
        slot.borrow()
            .as_ref()
            .is_some_and(|flag| flag.load(Ordering::Relaxed))
    })
}
