//! One weak-scaling cell, and the parallel-tempering worker process.
//!
//! Usage: `weak_scale <pt|population|dmc> <workers> <seed>`

fn main() {
    anneal_core::weak_scale::entry();
}
