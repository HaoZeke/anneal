//! Calls into the packing classifiers and builders, recorded per thread.
//!
//! Test builds only. Each classifier or builder that recognises or
//! imposes a named packing (a fivefold axis, an icosahedral share, the
//! 555/421/422 motif fractions, a twin plane, a lattice or a cut) notes its
//! name on entry, so a test can ask what a whole search called.

use std::cell::RefCell;

thread_local! {
    static CALLS: RefCell<Vec<&'static str>> = const { RefCell::new(Vec::new()) };
}

pub(crate) fn note(name: &'static str) {
    CALLS.with(|calls| {
        let mut calls = calls.borrow_mut();
        if !calls.contains(&name) {
            calls.push(name);
        }
    });
}

/// Names noted since the last take, each once.
pub(crate) fn take() -> Vec<&'static str> {
    CALLS.with(|calls| std::mem::take(&mut *calls.borrow_mut()))
}
