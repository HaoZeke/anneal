use ndarray::{Array1, ArrayView1};

/// Stand-in for the packing accounts.
pub struct PackingBank;

impl PackingBank {
    /// Empty accounts.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Records nothing.
    pub fn offer(&mut self, _energy: f64, _state: ArrayView1<f64>, _cutoff: f64) {}

    /// No other packing is stored yet.
    pub fn better_other(
        &self,
        _here: ArrayView1<f64>,
        _energy: f64,
        _cutoff: f64,
    ) -> Option<(f64, Array1<f64>)> {
        None
    }
}
