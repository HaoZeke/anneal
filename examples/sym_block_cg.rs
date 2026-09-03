//! What the symmetry block-diagonalisation costs and what it buys.
//!
//! The depth solve is conjugate gradients on the model Hessian. Where a
//! structure carries a point group the operator commutes with it, so the solve
//! splits into a totally symmetric block and its complement, and each block
//! sees a shorter spectrum than the whole. The question is whether the shorter
//! spectrum repays running two Krylov spaces instead of one.
//!
//! Prints, per structure: the order of the site group, the dimensions of the
//! two blocks, the condition number on each, the operator applications each
//! solve costs, and the relative disagreement between the plain depth and the
//! sum of the blocks.

use anneal_core::sym_hessian::{
    block_depth, conditioning, isotypic_bases, restricted_condition, dense_model, site_symmetries,
    split_depth, SiteSymmetry,
};
use anneal_core::symmetrise::{detect_all, generate_group};
use ndarray::{Array1, ArrayView1};

/// Deterministic uniform stream, so the example pulls in no extra dependency
/// and every structure sees the same gradient sequence.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Twelve vertices of a regular icosahedron, Ih.
fn icosahedron() -> Array1<f64> {
    let phi = (1.0 + 5.0_f64.sqrt()) / 2.0;
    let mut pts: Vec<[f64; 3]> = Vec::new();
    for &s1 in &[1.0, -1.0] {
        for &s2 in &[1.0, -1.0] {
            pts.push([0.0, s1, s2 * phi]);
            pts.push([s1, s2 * phi, 0.0]);
            pts.push([s2 * phi, 0.0, s1]);
        }
    }
    let mut x = Array1::zeros(3 * pts.len());
    for (i, p) in pts.iter().enumerate() {
        for k in 0..3 {
            x[3 * i + k] = p[k] / 2.0;
        }
    }
    x
}

/// The twenty-four vertices of a truncated octahedron, Oh: every permutation
/// of `(0, +-1, +-2)`.
fn truncated_octahedron() -> Array1<f64> {
    let mut pts: Vec<[f64; 3]> = Vec::new();
    for &a in &[1.0f64, -1.0] {
        for &b in &[2.0f64, -2.0] {
            pts.push([0.0, a, b]);
            pts.push([0.0, b, a]);
            pts.push([a, 0.0, b]);
            pts.push([b, 0.0, a]);
            pts.push([a, b, 0.0]);
            pts.push([b, a, 0.0]);
        }
    }
    let mut x = Array1::zeros(3 * pts.len());
    for (i, p) in pts.iter().enumerate() {
        for k in 0..3 {
            x[3 * i + k] = p[k];
        }
    }
    x
}

/// A cuboctahedron, Oh at thirteen points once the centre is added: the
/// close-packed thirteen-point shell, against the icosahedral one.
fn centred_cuboctahedron() -> Array1<f64> {
    let mut pts: Vec<[f64; 3]> = vec![[0.0, 0.0, 0.0]];
    for &a in &[1.0f64, -1.0] {
        for &b in &[1.0f64, -1.0] {
            pts.push([a, b, 0.0]);
            pts.push([a, 0.0, b]);
            pts.push([0.0, a, b]);
        }
    }
    let mut x = Array1::zeros(3 * pts.len());
    for (i, p) in pts.iter().enumerate() {
        for k in 0..3 {
            x[3 * i + k] = p[k];
        }
    }
    x
}

/// A blob with no symmetry: the control, where the split must do nothing.
fn asymmetric(n: usize, seed: u64) -> Array1<f64> {
    let mut rng = Rng(seed);
    Array1::from((0..3 * n).map(|_| rng.next() * 5.0).collect::<Vec<_>>())
}

fn group_of(x: ArrayView1<f64>, n: usize) -> Vec<SiteSymmetry> {
    let cands = detect_all(x, n, &[2, 3, 4, 5], 0.05);
    let group = generate_group(&cands, 200);
    site_symmetries(x, n, &group, 1e-6)
}

fn main() {
    let cases: Vec<(&str, Array1<f64>)> = vec![
        ("icosahedron (Ih)", icosahedron()),
        ("cuboctahedron + centre (Oh)", centred_cuboctahedron()),
        ("truncated octahedron (Oh)", truncated_octahedron()),
        ("random blob, 12 (C1)", asymmetric(12, 20260806)),
        ("random blob, 24 (C1)", asymmetric(24, 991)),
    ];

    // Two tolerances, because an iteration count is only meaningful against the
    // accuracy it bought and the ratio can move with it.
    for &tol in &[1e-6, 1e-10] {
        println!("\nrelative residual tolerance {tol:e}");
        println!(
            "{:<30} {:>5} {:>7} {:>7} {:>9} {:>9} {:>6} {:>6} {:>6} {:>7} {:>10}",
            "structure",
            "|G|",
            "dim A1",
            "dim all",
            "cond A1",
            "cond all",
            "plain",
            "sym",
            "rest",
            "ratio",
            "split err"
        );
        for (name, x) in &cases {
            let n = x.len() / 3;
            let sym = group_of(x.view(), n);
            let cond = conditioning(x.view(), n, &sym);
            // Averaged over ten gradients: a single right-hand side can land
            // favourably in the Krylov space and the count is what is being
            // compared.
            let mut rng = Rng(0xC0FFEE);
            let (mut plain, mut s, mut r, mut err) = (0.0, 0.0, 0.0, 0.0f64);
            let trials = 10;
            for _ in 0..trials {
                let g = Array1::from((0..3 * n).map(|_| rng.next()).collect::<Vec<_>>());
                let rep = split_depth(x.view(), n, g.view(), &sym, tol, 500);
                plain += rep.plain.iterations as f64;
                s += rep.symmetric.iterations as f64;
                r += rep.remainder.iterations as f64;
                err = err.max(rep.depth_error);
            }
            let t = trials as f64;
            let (plain, s, r) = (plain / t, s / t, r / t);
            let (c_sym, c_full, d_sym, d_full) = match cond {
                Some(c) => (c.symmetric, c.full, c.symmetric_dim, c.full_dim),
                None => (f64::NAN, f64::NAN, 0, 3 * n - 3),
            };
            println!(
                "{name:<30} {:>5} {d_sym:>7} {d_full:>7} {c_sym:>9.2} {c_full:>9.2} \
                 {plain:>6.1} {s:>6.1} {r:>6.1} {:>7.2} {err:>10.2e}",
                sym.len(),
                (s + r) / plain.max(1.0)
            );
        }
    }
    println!(
        "\nratio above one means the A1 split costs more operator applications \
         than the single solve it replaces."
    );

    // The whole block-diagonalisation, not only its totally symmetric piece.
    // A1 is one isotypic component of several, and a projection onto it alone
    // leaves everything else in one undecomposed block, so the count above says
    // nothing about what a full decomposition would cost.
    println!("\nfull isotypic decomposition, tolerance 1e-10");
    println!(
        "{:<30} {:>7} {:>22} {:>9} {:>7} {:>7} {:>7} {:>10}",
        "structure", "blocks", "dims", "worst cond", "plain", "sum", "worst", "split err"
    );
    for (name, x) in &cases {
        let n = x.len() / 3;
        let sym = group_of(x.view(), n);
        let bases = isotypic_bases(&sym, n, 0x5EED);
        let h = dense_model(x.view(), n);
        let worst_cond = bases
            .iter()
            .filter_map(|b| restricted_condition(h.view(), b.view()))
            .fold(0.0f64, f64::max);
        let mut rng = Rng(0xC0FFEE);
        let (mut plain, mut sum, mut worst, mut err) = (0.0, 0.0, 0.0, 0.0f64);
        let trials = 10;
        for _ in 0..trials {
            let g = Array1::from((0..3 * n).map(|_| rng.next()).collect::<Vec<_>>());
            let rep = block_depth(x.view(), n, g.view(), &bases, 1e-10, 500);
            plain += rep.plain.iterations as f64;
            sum += rep.blocks.iter().map(|b| b.iterations).sum::<usize>() as f64;
            worst += rep.critical_path() as f64;
            err = err.max(rep.depth_error);
        }
        let t = trials as f64;
        let mut dims: Vec<usize> = bases.iter().map(|b| b.ncols()).collect();
        dims.sort_unstable();
        let shown = if dims.len() > 6 {
            format!("{:?}..", &dims[dims.len() - 5..])
        } else {
            format!("{dims:?}")
        };
        println!(
            "{name:<30} {:>7} {shown:>22} {worst_cond:>9.2} {:>7.1} {:>7.1} {:>7.1} {err:>10.2e}",
            bases.len(),
            plain / t,
            sum / t,
            worst / t
        );
    }
    println!(
        "\n`sum` is what one core pays, `worst` what a core-per-block machine \
         waits for; both against `plain`."
    );
}
