//! Cluster pair calls loaded from rgpot.
//!
//! `RGPOT_LIB` is the path of `librgpot`. The symbols are
//! `rgpot_lj_gradient`, `rgpot_lj_pair`, `rgpot_lj_atom_energies`,
//! `rgpot_lj_partial_gradient`, `rgpot_lj_arithmetic_gradient`,
//! `rgpot_morse_gradient`, `rgpot_morse_pair`, `rgpot_binary_lj_gradient`,
//! and `rgpot_tip4p_sites`. Gradients are derivatives of the energy.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int, c_void};
use std::sync::OnceLock;

use ndarray::{Array1, ArrayView1};

unsafe extern "C" {
    fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    fn dlerror() -> *mut c_char;
}

type LjGrad = unsafe extern "C" fn(f64, f64, f64, i32, *const f64, *mut f64, *mut f64) -> i32;
type LjPair = unsafe extern "C" fn(f64, f64, f64, *mut f64, *mut f64, *mut f64) -> i32;
type LjAtoms = unsafe extern "C" fn(f64, f64, i32, *const f64, *mut f64) -> i32;
type LjPartial = unsafe extern "C" fn(f64, f64, i32, *const f64, i32, *const i32, *mut f64) -> i32;
type LjArith = unsafe extern "C" fn(*const f64, f64, i32, *const f64, *mut f64, *mut f64) -> i32;
type MorseGrad =
    unsafe extern "C" fn(f64, f64, f64, f64, i32, *const f64, *mut f64, *mut f64) -> i32;
type MorsePair = unsafe extern "C" fn(f64, f64, f64, f64, *mut f64, *mut f64, *mut f64) -> i32;
type Binary = unsafe extern "C" fn(
    i32,
    *const f64,
    *const f64,
    f64,
    i32,
    *const f64,
    *mut f64,
    *mut f64,
) -> i32;
type Tip4p = unsafe extern "C" fn(*const Tip4pC, i32, *const f64, *mut f64, *mut f64) -> i32;

#[repr(C)]
struct Tip4pC {
    c6: f64,
    c12: f64,
    epsilon: f64,
    sigma: f64,
    use_epsilon: i32,
    coulomb: f64,
    q_h: f64,
    q_m: f64,
}

struct Api {
    lj_gradient: LjGrad,
    lj_pair: LjPair,
    lj_atoms: LjAtoms,
    lj_partial: LjPartial,
    lj_arithmetic: LjArith,
    morse_gradient: MorseGrad,
    morse_pair: MorsePair,
    binary: Binary,
    tip4p: Tip4p,
}

// The search calls the loaded symbols from more than one thread.
unsafe impl Send for Api {}
unsafe impl Sync for Api {}

fn api() -> &'static Api {
    static API: OnceLock<Api> = OnceLock::new();
    API.get_or_init(|| {
        let path = std::env::var("RGPOT_LIB").unwrap_or_else(|_| {
            panic!("set RGPOT_LIB to librgpot");
        });
        let c_path = CString::new(path.clone()).expect("library path");
        let handle = unsafe { dlopen(c_path.as_ptr(), 2) };
        if handle.is_null() {
            let err = unsafe { dlerror() };
            let msg = if err.is_null() {
                "dlopen failed".to_string()
            } else {
                unsafe { CStr::from_ptr(err) }
                    .to_string_lossy()
                    .into_owned()
            };
            panic!("cannot load {path}: {msg}");
        }
        Api {
            lj_gradient: load(handle, "rgpot_lj_gradient"),
            lj_pair: load(handle, "rgpot_lj_pair"),
            lj_atoms: load(handle, "rgpot_lj_atom_energies"),
            lj_partial: load(handle, "rgpot_lj_partial_gradient"),
            lj_arithmetic: load(handle, "rgpot_lj_arithmetic_gradient"),
            morse_gradient: load(handle, "rgpot_morse_gradient"),
            morse_pair: load(handle, "rgpot_morse_pair"),
            binary: load(handle, "rgpot_binary_lj_gradient"),
            tip4p: load(handle, "rgpot_tip4p_sites"),
        }
    })
}

fn load<T>(handle: *mut c_void, name: &str) -> T {
    let symbol = CString::new(name).unwrap();
    let raw = unsafe { dlsym(handle, symbol.as_ptr()) };
    if raw.is_null() {
        panic!("librgpot has no {name}");
    }
    unsafe { std::mem::transmute_copy(&raw) }
}

fn with_coords<R>(x: ArrayView1<f64>, body: impl FnOnce(*const f64, usize) -> R) -> R {
    if let Some(slice) = x.as_slice() {
        body(slice.as_ptr(), slice.len())
    } else {
        let owned = x.to_vec();
        body(owned.as_ptr(), owned.len())
    }
}

fn gradient_from(raw: Vec<f64>) -> Array1<f64> {
    Array1::from_vec(raw)
}

/// Reduced Lennard-Jones, `epsilon = 1`, `sigma = 1`, no cutoff.
pub fn reduced_lennard_jones(x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    lennard_jones(1.0, 1.0, 0.0, x)
}

/// Unshifted Lennard-Jones. `cutoff <= 0` keeps every pair.
pub fn lennard_jones(
    epsilon: f64,
    sigma: f64,
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_coords(x, |ptr, len| {
        let n = (len / 3) as i32;
        let mut energy = 0.0;
        let mut gradient = vec![0.0; len];
        let rc = unsafe {
            (api().lj_gradient)(
                epsilon,
                sigma,
                cutoff,
                n,
                ptr,
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_lj_gradient failed");
        (energy, gradient_from(gradient))
    })
}

/// Energy, `(dV/dr) / r`, and `d2V/dr2` of one Lennard-Jones pair.
pub fn lj_pair_terms(epsilon: f64, sigma: f64, r2: f64) -> (f64, f64, f64) {
    let mut energy = 0.0;
    let mut coef = 0.0;
    let mut curv = 0.0;
    let rc = unsafe { (api().lj_pair)(epsilon, sigma, r2, &mut energy, &mut coef, &mut curv) };
    assert_eq!(rc, 0, "rgpot_lj_pair failed");
    (energy, coef, curv)
}

/// Pair energy of each atom. The total energy is half the sum.
pub fn lennard_jones_atom_energies(epsilon: f64, sigma: f64, x: ArrayView1<f64>) -> Array1<f64> {
    with_coords(x, |ptr, len| {
        let n = (len / 3) as i32;
        let mut atoms = vec![0.0; n as usize];
        let rc = unsafe { (api().lj_atoms)(epsilon, sigma, n, ptr, atoms.as_mut_ptr()) };
        assert_eq!(rc, 0, "rgpot_lj_atom_energies failed");
        Array1::from_vec(atoms)
    })
}

/// Gradient rows of `moved` only.
pub fn lennard_jones_partial(x: ArrayView1<f64>, moved: &[usize]) -> Array1<f64> {
    let moved: Vec<i32> = moved.iter().map(|index| *index as i32).collect();
    with_coords(x, |ptr, len| {
        let n = (len / 3) as i32;
        let mut gradient = vec![0.0; len];
        let rc = unsafe {
            (api().lj_partial)(
                1.0,
                1.0,
                n,
                ptr,
                moved.len() as i32,
                moved.as_ptr(),
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_lj_partial_gradient failed");
        gradient_from(gradient)
    })
}

/// Arithmetic-mean diameters, one `sigma` per atom.
pub fn lennard_jones_arithmetic(
    sigma: &[f64],
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_coords(x, |ptr, len| {
        let n = (len / 3) as i32;
        assert_eq!(sigma.len(), n as usize, "one sigma per atom");
        let mut energy = 0.0;
        let mut gradient = vec![0.0; len];
        let rc = unsafe {
            (api().lj_arithmetic)(
                sigma.as_ptr(),
                cutoff,
                n,
                ptr,
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_lj_arithmetic_gradient failed");
        (energy, gradient_from(gradient))
    })
}

/// Doye and Wales Morse at `epsilon = 1`, `re = 1`, no cutoff.
pub fn morse(rho: f64, x: ArrayView1<f64>) -> (f64, Array1<f64>) {
    morse_cluster(1.0, rho, 1.0, 0.0, x)
}

/// Unshifted Morse. `cutoff <= 0` keeps every pair.
pub fn morse_cluster(
    epsilon: f64,
    rho: f64,
    re: f64,
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_coords(x, |ptr, len| {
        let n = (len / 3) as i32;
        let mut energy = 0.0;
        let mut gradient = vec![0.0; len];
        let rc = unsafe {
            (api().morse_gradient)(
                epsilon,
                rho,
                re,
                cutoff,
                n,
                ptr,
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_morse_gradient failed");
        (energy, gradient_from(gradient))
    })
}

/// Energy, `(dV/dr) / r`, and `d2V/dr2` of one Morse pair.
pub fn morse_pair_terms(epsilon: f64, rho: f64, re: f64, r2: f64) -> (f64, f64, f64) {
    let mut energy = 0.0;
    let mut coef = 0.0;
    let mut curv = 0.0;
    let rc = unsafe { (api().morse_pair)(epsilon, rho, re, r2, &mut energy, &mut coef, &mut curv) };
    assert_eq!(rc, 0, "rgpot_morse_pair failed");
    (energy, coef, curv)
}

/// Binary Lennard-Jones. Species A is the first `n_a` atoms.
pub fn binary_lennard_jones(
    n_a: usize,
    epsilon: [f64; 3],
    sigma: [f64; 3],
    cutoff: f64,
    x: ArrayView1<f64>,
) -> (f64, Array1<f64>) {
    with_coords(x, |ptr, len| {
        let n = (len / 3) as i32;
        let mut energy = 0.0;
        let mut gradient = vec![0.0; len];
        let rc = unsafe {
            (api().binary)(
                n_a as i32,
                epsilon.as_ptr(),
                sigma.as_ptr(),
                cutoff,
                n,
                ptr,
                &mut energy,
                gradient.as_mut_ptr(),
            )
        };
        assert_eq!(rc, 0, "rgpot_binary_lj_gradient failed");
        (energy, gradient_from(gradient))
    })
}

/// Parameters of one TIP4P site evaluation.
#[derive(Debug, Clone, Copy)]
pub struct Tip4pSites {
    /// Attractive coefficient. Used when `use_epsilon` is false.
    pub c6: f64,
    /// Repulsive coefficient. Used when `use_epsilon` is false.
    pub c12: f64,
    /// Well depth. Used when `use_epsilon` is true.
    pub epsilon: f64,
    /// Diameter. Used when `use_epsilon` is true.
    pub sigma: f64,
    /// Select the epsilon and sigma oxygen term.
    pub use_epsilon: bool,
    /// Coulomb prefactor.
    pub coulomb: f64,
    /// Hydrogen charge.
    pub q_h: f64,
    /// M-site charge.
    pub q_m: f64,
}

/// TIP4P site energy and gradient. `sites` is molecule-major O, H, H, M.
pub fn tip4p_site_gradient(cfg: Tip4pSites, sites: &[f64]) -> (f64, Vec<f64>) {
    assert!(
        sites.len().is_multiple_of(12) && !sites.is_empty(),
        "TIP4P sites are 12 coordinates per molecule"
    );
    let c = Tip4pC {
        c6: cfg.c6,
        c12: cfg.c12,
        epsilon: cfg.epsilon,
        sigma: cfg.sigma,
        use_epsilon: i32::from(cfg.use_epsilon),
        coulomb: cfg.coulomb,
        q_h: cfg.q_h,
        q_m: cfg.q_m,
    };
    let n = (sites.len() / 12) as i32;
    let mut energy = 0.0;
    let mut gradient = vec![0.0; sites.len()];
    let rc = unsafe { (api().tip4p)(&c, n, sites.as_ptr(), &mut energy, gradient.as_mut_ptr()) };
    assert_eq!(rc, 0, "rgpot_tip4p_sites failed");
    (energy, gradient)
}
