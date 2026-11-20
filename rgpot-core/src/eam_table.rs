// MIT License
// Copyright 2023--present rgpot developers

//! Tabulated embedded atom method (EAM) potentials read from DYNAMO tables.
//!
//! This is the evaluation of `CppCore/rgpot/EAMTable/EAMTablePot.cc` for a
//! host that links `rgpot-core` and nothing else: `rgpot-core` never links
//! the C++ kernels, so the table reader, the cubic tables and the pair sums
//! follow that file step by step, and both are checked against the same
//! LAMMPS runs in `CppCore/tests/data/eam_table`.
//!
//! The three layouts are those of the `pair_eam` page of LAMMPS:
//! [`EamTableFormat::Funcfl`] is `pair_style eam` (one element),
//! [`EamTableFormat::Setfl`] is `eam/alloy` and
//! [`EamTableFormat::FinnisSinclair`] is `eam/fs`. The energy is
//!
//! ```text
//! E = sum_i F_a(rho_i) + sum_{i<j} phi_ab(r_ij)
//! rho_i = sum_{j != i} rho_ab(r_ij)
//! ```
//!
//! with `a` the element of atom `i` and `b` that of atom `j`. Lengths are in
//! Angstrom and energies in eV, the units of the LAMMPS `metal` files.
//! Forces are minus the gradient of the energy. An atom picks its element by
//! atomic number, matched against the first field of each element line of the
//! file.
//!
//! The tables go through the steps of `PairEAM` in LAMMPS: a funcfl array is
//! resampled by four-point Lagrange interpolation onto one point fewer than
//! the file lists, the pair term of a funcfl file is `27.2 * 0.529 * Z(r)^2`,
//! each array becomes the seven-coefficient cubic of `PairEAM::interpolate`,
//! and the embedding energy is extended linearly above `rhomax`.
//!
//! A pair counts when `r` is below the cutoff of the file. A periodic
//! evaluation keeps the nearest image of each partner, so it needs the cutoff
//! under half of every face spacing of the cell, and refuses a smaller cell.
//! The pair search visits every pair, which costs the square of the atom
//! count and suits the cells of a few hundred atoms that a saddle search
//! evaluates.
//!
//! Failures are [`EamTableError`]. Through the C ABI,
//! `rgpot_potential_new_eam_table` returns NULL with the message in
//! `rgpot_last_error`, and an evaluation that fails returns a status.

use std::ffi::c_void;
use std::fmt;
use std::fs;
use std::os::raw::c_int;
use std::path::Path;

use crate::potential::PotentialImpl;
use crate::status::rgpot_status_t;
use crate::tensor::create_owned_f64_tensor;

/// Format code of the C ABI: choose from the suffix of the file name.
/// `.eam.alloy` and `.setfl` are setfl, `.eam.fs` is Finnis-Sinclair and
/// `.eam` is funcfl.
pub const RGPOT_EAM_TABLE_AUTO: c_int = 0;
/// Format code of the C ABI: funcfl, `pair_style eam`, one element.
pub const RGPOT_EAM_TABLE_FUNCFL: c_int = 1;
/// Format code of the C ABI: setfl, `pair_style eam/alloy`.
pub const RGPOT_EAM_TABLE_SETFL: c_int = 2;
/// Format code of the C ABI: Finnis-Sinclair, `pair_style eam/fs`.
pub const RGPOT_EAM_TABLE_FINNIS_SINCLAIR: c_int = 3;

/// Hartree in eV times Bohr in Angstrom, as `pair_eam.cpp` writes it.
const HARTREE_BOHR: f64 = 27.2 * 0.529;

/// Coefficients of one cubic of an interpolation table.
const COEFFICIENTS: usize = 7;

/// File layout of a table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EamTableFormat {
    /// Choose from the suffix of the file name; see [`RGPOT_EAM_TABLE_AUTO`].
    Auto,
    /// `pair_style eam`: one element with F(rho), Z(r) and rho(r).
    Funcfl,
    /// `pair_style eam/alloy`: F_a(rho), rho_b(r) and r phi_ab(r).
    Setfl,
    /// `pair_style eam/fs`: F_a(rho), rho_ab(r) and r phi_ab(r).
    FinnisSinclair,
}

impl EamTableFormat {
    /// The format of a C ABI format code (`RGPOT_EAM_TABLE_*`).
    pub fn from_code(code: c_int) -> Result<Self, EamTableError> {
        match code {
            RGPOT_EAM_TABLE_AUTO => Ok(Self::Auto),
            RGPOT_EAM_TABLE_FUNCFL => Ok(Self::Funcfl),
            RGPOT_EAM_TABLE_SETFL => Ok(Self::Setfl),
            RGPOT_EAM_TABLE_FINNIS_SINCLAIR => Ok(Self::FinnisSinclair),
            other => Err(EamTableError::Invalid(format!(
                "format code {other} is not one of RGPOT_EAM_TABLE_AUTO, _FUNCFL, _SETFL or _FINNIS_SINCLAIR"
            ))),
        }
    }

    fn layout(self, path: &Path) -> Result<Layout, EamTableError> {
        match self {
            Self::Funcfl => Ok(Layout::Funcfl),
            Self::Setfl => Ok(Layout::Setfl),
            Self::FinnisSinclair => Ok(Layout::FinnisSinclair),
            Self::Auto => {
                let name = path.to_string_lossy().to_ascii_lowercase();
                if name.ends_with(".eam.alloy") || name.ends_with(".setfl") {
                    Ok(Layout::Setfl)
                } else if name.ends_with(".eam.fs") {
                    Ok(Layout::FinnisSinclair)
                } else if name.ends_with(".eam") {
                    Ok(Layout::Funcfl)
                } else {
                    Err(EamTableError::Invalid(format!(
                        "cannot tell the table format of {} from its suffix; give funcfl, setfl or Finnis-Sinclair",
                        path.display()
                    )))
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Layout {
    Funcfl,
    Setfl,
    FinnisSinclair,
}

/// Why a table or an evaluation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EamTableError {
    /// An argument is not usable: a NULL or empty path, a format code that
    /// is none of the constants, a format that cannot be chosen.
    Invalid(String),
    /// The table file could not be read.
    Open {
        /// The path that was opened.
        path: String,
        /// The reason the operating system gave.
        reason: String,
    },
    /// The table file is not a table of the format asked for.
    Malformed(String),
    /// The geometry cannot be evaluated: a wrong array length, a
    /// non-finite coordinate, an atomic number the table lacks, or a cell
    /// that is too small or has no volume.
    Geometry(String),
    /// The evaluation gave a non-finite energy or force.
    NonFinite,
}

impl EamTableError {
    /// The status the C ABI returns for this failure.
    pub fn status(&self) -> rgpot_status_t {
        match self {
            Self::NonFinite => rgpot_status_t::RGPOT_INTERNAL_ERROR,
            _ => rgpot_status_t::RGPOT_INVALID_PARAMETER,
        }
    }
}

impl fmt::Display for EamTableError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(why) => write!(f, "EAM table: {why}"),
            Self::Open { path, reason } => write!(f, "EAM table: cannot open {path}: {reason}"),
            Self::Malformed(why) => write!(f, "EAM table: {why}"),
            Self::Geometry(why) => write!(f, "EAM table: {why}"),
            Self::NonFinite => write!(f, "EAM table: the evaluation gave a non-finite value"),
        }
    }
}

impl std::error::Error for EamTableError {}

fn malformed<T>(why: impl Into<String>) -> Result<T, EamTableError> {
    Err(EamTableError::Malformed(why.into()))
}

fn geometry<T>(why: impl Into<String>) -> Result<T, EamTableError> {
    Err(EamTableError::Geometry(why.into()))
}

/// Tabulated functions after the LAMMPS `file2array` step. Entry `m` of an
/// array sits at `m * dr` or `m * drho`.
#[derive(Debug, Clone)]
pub struct EamTable {
    nrho: usize,
    drho: f64,
    nr: usize,
    dr: f64,
    cutoff: f64,
    rhomax: f64,
    /// Atomic number of each element.
    z: Vec<i32>,
    /// F_a, one per element.
    frho: Vec<Vec<f64>>,
    /// rho_ab at index `b * n + a`, the density element b puts at an a site.
    rhor: Vec<Vec<f64>>,
    /// r phi_ab at index `a (a + 1) / 2 + b` for `a >= b`.
    z2r: Vec<Vec<f64>>,
}

impl EamTable {
    /// Read the table file at `path`.
    pub fn open(path: &Path, format: EamTableFormat) -> Result<Self, EamTableError> {
        let layout = format.layout(path)?;
        let text = fs::read_to_string(path).map_err(|why| EamTableError::Open {
            path: path.display().to_string(),
            reason: why.to_string(),
        })?;
        Self::parse(&text, layout)
    }

    /// Read a table from text in the given layout. [`EamTableFormat::Auto`]
    /// has no file name to read, so it is refused.
    pub fn from_text(text: &str, format: EamTableFormat) -> Result<Self, EamTableError> {
        match format {
            EamTableFormat::Auto => Err(EamTableError::Invalid(
                "a table read from text needs funcfl, setfl or Finnis-Sinclair".into(),
            )),
            EamTableFormat::Funcfl => Self::parse(text, Layout::Funcfl),
            EamTableFormat::Setfl => Self::parse(text, Layout::Setfl),
            EamTableFormat::FinnisSinclair => Self::parse(text, Layout::FinnisSinclair),
        }
    }

    /// Radial cutoff of the table, in Angstrom.
    pub fn cutoff(&self) -> f64 {
        self.cutoff
    }

    /// Atomic number of each element of the table, in file order.
    pub fn atomic_numbers(&self) -> &[i32] {
        &self.z
    }

    fn parse(text: &str, layout: Layout) -> Result<Self, EamTableError> {
        let mut input = Scanner::new(text);
        if layout == Layout::Funcfl {
            input.skip_line();
            let z = input.element_line()?;
            let grid = input.grid_line()?;
            let f = input.block(grid.nrho)?;
            let zr = input.block(grid.nr)?;
            let rho = input.block(grid.nr)?;
            let rhomax = (grid.nrho - 1) as f64 * grid.drho;
            let nrho = (rhomax / grid.drho).round() as usize;
            let nr = ((grid.nr - 1) as f64 * grid.dr / grid.dr).round() as usize;
            let mut z2 = resample(&zr, grid.dr, nr, grid.dr);
            for v in &mut z2 {
                *v = HARTREE_BOHR * *v * *v;
            }
            return Ok(Self {
                nrho,
                drho: grid.drho,
                nr,
                dr: grid.dr,
                cutoff: grid.cutoff,
                rhomax,
                z: vec![z],
                frho: vec![resample(&f, grid.drho, nrho, grid.drho)],
                rhor: vec![resample(&rho, grid.dr, nr, grid.dr)],
                z2r: vec![z2],
            });
        }

        for _ in 0..3 {
            input.skip_line();
        }
        let nel: i64 = input.number("the element count")?;
        input.skip_line();
        if nel < 1 {
            return malformed("unreadable element count");
        }
        let nel = nel as usize;
        let grid = input.grid_line()?;
        let per_element = if layout == Layout::FinnisSinclair {
            nel
        } else {
            1
        };
        let mut z = Vec::new();
        let mut frho = Vec::new();
        let mut dens = Vec::new();
        for _ in 0..nel {
            z.push(input.element_line()?);
            frho.push(input.block(grid.nrho)?);
            for _ in 0..per_element {
                dens.push(input.block(grid.nr)?);
            }
        }
        let mut z2r = Vec::new();
        for a in 0..nel {
            for _ in 0..=a {
                z2r.push(input.block(grid.nr)?);
            }
        }
        let mut rhor = Vec::new();
        for b in 0..nel {
            for a in 0..nel {
                rhor.push(dens[b * per_element + a % per_element].clone());
            }
        }
        Ok(Self {
            nrho: grid.nrho,
            drho: grid.drho,
            nr: grid.nr,
            dr: grid.dr,
            cutoff: grid.cutoff,
            rhomax: (grid.nrho - 1) as f64 * grid.drho,
            z,
            frho,
            rhor,
            z2r,
        })
    }
}

struct Grid {
    nrho: usize,
    drho: f64,
    nr: usize,
    dr: f64,
    cutoff: f64,
}

/// Whitespace separated reader that also knows lines, as the stream the
/// LAMMPS reader gives `>>` and `getline`.
struct Scanner<'a> {
    rest: &'a str,
}

impl<'a> Scanner<'a> {
    fn new(text: &'a str) -> Self {
        Self { rest: text }
    }

    /// Drop the rest of the current line, newline included.
    fn skip_line(&mut self) {
        match self.rest.find('\n') {
            Some(at) => self.rest = &self.rest[at + 1..],
            None => self.rest = "",
        }
    }

    fn token(&mut self) -> Option<&'a str> {
        let text = self.rest.trim_start();
        if text.is_empty() {
            self.rest = text;
            return None;
        }
        let (token, rest) = match text.find(char::is_whitespace) {
            Some(end) => text.split_at(end),
            None => (text, ""),
        };
        self.rest = rest;
        Some(token)
    }

    fn number<T: std::str::FromStr>(&mut self, what: &str) -> Result<T, EamTableError> {
        match self.token().map(str::parse::<T>) {
            Some(Ok(value)) => Ok(value),
            _ => malformed(format!("unreadable {what}")),
        }
    }

    /// `z mass ...` and the rest of the line. Gives the atomic number.
    fn element_line(&mut self) -> Result<i32, EamTableError> {
        let z: i32 = self.number("element line")?;
        let _mass: f64 = self.number("element line")?;
        self.skip_line();
        Ok(z)
    }

    /// `nrho drho nr dr cutoff` and the rest of the line.
    fn grid_line(&mut self) -> Result<Grid, EamTableError> {
        let nrho: i64 = self.number("grid line")?;
        let drho: f64 = self.number("grid line")?;
        let nr: i64 = self.number("grid line")?;
        let dr: f64 = self.number("grid line")?;
        let cutoff: f64 = self.number("grid line")?;
        self.skip_line();
        let positive = |v: f64| v > 0.0 && v.is_finite();
        if nrho < 5 || nr < 5 || !positive(drho) || !positive(dr) || !positive(cutoff) {
            return malformed("unreadable grid line");
        }
        Ok(Grid {
            nrho: nrho as usize,
            drho,
            nr: nr as usize,
            dr,
            cutoff,
        })
    }

    /// `n` numbers. The vector grows as numbers arrive, so a count the file
    /// does not back costs nothing.
    fn block(&mut self, n: usize) -> Result<Vec<f64>, EamTableError> {
        let mut values = Vec::new();
        for _ in 0..n {
            let v: f64 = match self.token().map(str::parse::<f64>) {
                Some(Ok(v)) if v.is_finite() => v,
                _ => return malformed("short or unreadable array"),
            };
            values.push(v);
        }
        Ok(values)
    }
}

/// Four-point Lagrange resampling of `f` (spacing `fd`) onto `n` points of
/// spacing `d`, the funcfl step of `PairEAM::file2array`.
fn resample(f: &[f64], fd: f64, n: usize, d: f64) -> Vec<f64> {
    let sixth = 1.0 / 6.0;
    let nf = f.len() as i64;
    let mut out = Vec::with_capacity(n);
    for m in 1..=n {
        let p = (m - 1) as f64 * d / fd + 1.0;
        let k = (p as i64).min(nf - 2).max(2);
        let p = (p - k as f64).min(2.0);
        let c1 = -sixth * p * (p - 1.0) * (p - 2.0);
        let c2 = 0.5 * (p * p - 1.0) * (p - 2.0);
        let c3 = -0.5 * p * (p + 1.0) * (p - 2.0);
        let c4 = sixth * p * (p * p - 1.0);
        let k = k as usize;
        out.push(c1 * f[k - 2] + c2 * f[k - 1] + c3 * f[k] + c4 * f[k + 1]);
    }
    out
}

/// `PairEAM::interpolate`. Knot `m` holds the cubic on `[m, m + 1]` in the
/// local coordinate `p`: `c[3..=6]` for the value and `c[0..=2]` for its
/// derivative in `x`.
fn spline(f: &[f64], delta: f64) -> Vec<f64> {
    let n = f.len();
    let mut s = vec![0.0; COEFFICIENTS * n];
    let at = |m: usize, k: usize| COEFFICIENTS * (m - 1) + k;
    for m in 1..=n {
        s[at(m, 6)] = f[m - 1];
    }
    s[at(1, 5)] = s[at(2, 6)] - s[at(1, 6)];
    s[at(2, 5)] = 0.5 * (s[at(3, 6)] - s[at(1, 6)]);
    s[at(n - 1, 5)] = 0.5 * (s[at(n, 6)] - s[at(n - 2, 6)]);
    s[at(n, 5)] = s[at(n, 6)] - s[at(n - 1, 6)];
    for m in 3..n - 1 {
        s[at(m, 5)] = ((s[at(m - 2, 6)] - s[at(m + 2, 6)])
            + 8.0 * (s[at(m + 1, 6)] - s[at(m - 1, 6)]))
            / 12.0;
    }
    for m in 1..n {
        s[at(m, 4)] = 3.0 * (s[at(m + 1, 6)] - s[at(m, 6)]) - 2.0 * s[at(m, 5)] - s[at(m + 1, 5)];
        s[at(m, 3)] = s[at(m, 5)] + s[at(m + 1, 5)] - 2.0 * (s[at(m + 1, 6)] - s[at(m, 6)]);
    }
    s[at(n, 4)] = 0.0;
    s[at(n, 3)] = 0.0;
    for m in 1..=n {
        s[at(m, 2)] = s[at(m, 5)] / delta;
        s[at(m, 1)] = 2.0 * s[at(m, 4)] / delta;
        s[at(m, 0)] = 3.0 * s[at(m, 3)] / delta;
    }
    s
}

/// Where `x` falls in a table of `n` knots `1 / rdx` apart: the offset of the
/// coefficients of its cubic and the local coordinate in it.
#[derive(Clone, Copy)]
struct Place {
    offset: usize,
    p: f64,
}

fn place(x: f64, rdx: f64, n: usize) -> Place {
    let q = x * rdx + 1.0;
    let m = (q as i64).min(n as i64 - 1).max(1);
    Place {
        offset: COEFFICIENTS * (m as usize - 1),
        p: (q - m as f64).min(1.0),
    }
}

/// The cubic of a table at a [`Place`].
struct Knot<'a> {
    c: &'a [f64],
    p: f64,
}

impl<'a> Knot<'a> {
    fn new(table: &'a [f64], at: Place) -> Self {
        Self {
            c: &table[at.offset..at.offset + COEFFICIENTS],
            p: at.p,
        }
    }

    fn value(&self) -> f64 {
        let c = self.c;
        ((c[3] * self.p + c[4]) * self.p + c[5]) * self.p + c[6]
    }

    fn slope(&self) -> f64 {
        let c = self.c;
        (c[0] * self.p + c[1]) * self.p + c[2]
    }
}

/// A table with its cubics, and the evaluation of a geometry.
///
/// The potential is immutable once built, so one instance serves any number
/// of threads.
#[derive(Debug, Clone)]
pub struct EamTablePotential {
    table: EamTable,
    frho: Vec<Vec<f64>>,
    rhor: Vec<Vec<f64>>,
    z2r: Vec<Vec<f64>>,
    periodic: bool,
}

/// Energy in eV and forces in eV/Angstrom of one geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct EamEvaluation {
    /// Potential energy.
    pub energy: f64,
    /// Minus the gradient of the energy, `3 * n_atoms` values.
    pub forces: Vec<f64>,
}

struct Pair {
    i: usize,
    j: usize,
    d: [f64; 3],
    r: f64,
}

impl EamTablePotential {
    /// Read the table at `path` and build the potential. With `periodic`
    /// set, an evaluation keeps the nearest image of each partner and needs
    /// a cell; otherwise the cell is ignored.
    pub fn open(
        path: &Path,
        format: EamTableFormat,
        periodic: bool,
    ) -> Result<Self, EamTableError> {
        Self::new(EamTable::open(path, format)?, periodic)
    }

    /// Build the potential of a table already read.
    pub fn new(table: EamTable, periodic: bool) -> Result<Self, EamTableError> {
        let n = table.z.len();
        if n == 0
            || table.frho.len() != n
            || table.rhor.len() != n * n
            || table.z2r.len() != n * (n + 1) / 2
        {
            return malformed("array count does not match");
        }
        if let Some(z) = table.z.iter().find(|&&z| z < 0) {
            return malformed(format!("negative atomic number {z}"));
        }
        let min_knots = 4;
        let sized = |arrays: &[Vec<f64>], len: usize| {
            arrays
                .iter()
                .all(|a| a.len() == len && a.len() >= min_knots)
        };
        if !sized(&table.frho, table.nrho)
            || !sized(&table.rhor, table.nr)
            || !sized(&table.z2r, table.nr)
        {
            return malformed("array length does not match");
        }
        let cubics = |arrays: &[Vec<f64>], delta: f64| -> Vec<Vec<f64>> {
            arrays.iter().map(|a| spline(a, delta)).collect()
        };
        Ok(Self {
            frho: cubics(&table.frho, table.drho),
            rhor: cubics(&table.rhor, table.dr),
            z2r: cubics(&table.z2r, table.dr),
            table,
            periodic,
        })
    }

    /// The table this potential evaluates.
    pub fn table(&self) -> &EamTable {
        &self.table
    }

    /// Whether an evaluation keeps the nearest image of each partner.
    pub fn periodic(&self) -> bool {
        self.periodic
    }

    fn element_of(&self, z: i32) -> Option<usize> {
        self.table.z.iter().position(|&known| known == z)
    }

    /// F_a(rho) with the linear extension above `rhomax`, and F'_a(rho).
    fn embedding(&self, a: usize, rho: f64) -> (f64, f64) {
        let place = place(rho, 1.0 / self.table.drho, self.table.nrho);
        let knot = Knot::new(&self.frho[a], place);
        let slope = knot.slope();
        let mut energy = knot.value();
        if rho > self.table.rhomax {
            energy += slope * (rho - self.table.rhomax);
        }
        (energy, slope)
    }

    /// The radial tables of a pair of an a atom i and a b atom j: the
    /// density j puts at i, the density i puts at j, and r phi_ab.
    fn couple(&self, a: usize, b: usize) -> (&[f64], &[f64], &[f64]) {
        let n = self.table.z.len();
        let hi = a.max(b);
        (
            &self.rhor[b * n + a],
            &self.rhor[a * n + b],
            &self.z2r[hi * (hi + 1) / 2 + a.min(b)],
        )
    }

    /// The pair list within the cutoff, each pair once, `d = r_i - r_j`.
    fn pairs(&self, positions: &[f64], cell: &[f64; 9]) -> Result<Vec<Pair>, EamTableError> {
        let n = positions.len() / 3;
        let cutoff = self.table.cutoff;
        let cut2 = cutoff * cutoff;
        let inverse = if self.periodic {
            Some(self.cell_inverse(cell)?)
        } else {
            None
        };
        let mut pairs = Vec::new();
        for i in 0..n {
            for j in i + 1..n {
                let mut d = [0.0; 3];
                for (k, v) in d.iter_mut().enumerate() {
                    *v = positions[3 * i + k] - positions[3 * j + k];
                }
                if let Some(inverse) = &inverse {
                    d = nearest_image(d, cell, inverse);
                }
                let r2 = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if r2 > 0.0 && r2 < cut2 {
                    pairs.push(Pair {
                        i,
                        j,
                        d,
                        r: r2.sqrt(),
                    });
                }
            }
        }
        Ok(pairs)
    }

    /// The inverse of the cell, after the check that one image of each
    /// partner is all there is within the cutoff.
    fn cell_inverse(&self, cell: &[f64; 9]) -> Result<[f64; 9], EamTableError> {
        let row = |k: usize| [cell[3 * k], cell[3 * k + 1], cell[3 * k + 2]];
        let cross = |u: [f64; 3], v: [f64; 3]| {
            [
                u[1] * v[2] - u[2] * v[1],
                u[2] * v[0] - u[0] * v[2],
                u[0] * v[1] - u[1] * v[0],
            ]
        };
        let normals: Vec<[f64; 3]> = (0..3)
            .map(|k| cross(row((k + 1) % 3), row((k + 2) % 3)))
            .collect();
        let a = row(0);
        let det = a[0] * normals[0][0] + a[1] * normals[0][1] + a[2] * normals[0][2];
        let volume = det.abs();
        if volume <= 0.0 || !volume.is_finite() {
            return geometry("periodic evaluation needs a cell");
        }
        for normal in &normals {
            let area =
                (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
            if volume / area <= 2.0 * self.table.cutoff {
                return geometry("cutoff exceeds half a face spacing");
            }
        }
        // Columns of the inverse are the cross products of the other rows
        // over the determinant: inverse[m][k] = normals[k][m] / det.
        let mut inverse = [0.0; 9];
        for m in 0..3 {
            for k in 0..3 {
                inverse[3 * m + k] = normals[k][m] / det;
            }
        }
        Ok(inverse)
    }

    /// Energy and forces of one geometry.
    ///
    /// `positions` holds `3 * n` Angstrom values, `atomic_numbers` holds `n`
    /// values and `cell` is the lattice vectors as rows. The cell is ignored
    /// for a potential that is not periodic.
    pub fn force(
        &self,
        positions: &[f64],
        atomic_numbers: &[i32],
        cell: &[f64; 9],
    ) -> Result<EamEvaluation, EamTableError> {
        let n = atomic_numbers.len();
        if positions.len() != 3 * n {
            return geometry(format!(
                "{} coordinates for {n} atoms; expected {}",
                positions.len(),
                3 * n
            ));
        }
        if let Some(at) = positions.iter().position(|v| !v.is_finite()) {
            return geometry(format!("non-finite coordinate at element {at}"));
        }
        if self.periodic && cell.iter().any(|v| !v.is_finite()) {
            return geometry("non-finite cell");
        }
        let mut element = Vec::with_capacity(n);
        for &z in atomic_numbers {
            match self.element_of(z) {
                Some(a) => element.push(a),
                None => return geometry(format!("no element for atomic number {z}")),
            }
        }

        let pairs = self.pairs(positions, cell)?;
        let mut rho = vec![0.0; n];
        for pair in &pairs {
            let (at_i, at_j, _) = self.couple(element[pair.i], element[pair.j]);
            let at = place(pair.r, 1.0 / self.table.dr, self.table.nr);
            rho[pair.i] += Knot::new(at_i, at).value();
            rho[pair.j] += Knot::new(at_j, at).value();
        }

        let mut energy = 0.0;
        let mut fprime = vec![0.0; n];
        for i in 0..n {
            let (embed, slope) = self.embedding(element[i], rho[i]);
            energy += embed;
            fprime[i] = slope;
        }

        let mut forces = vec![0.0; 3 * n];
        for pair in &pairs {
            let (at_i, at_j, z2) = self.couple(element[pair.i], element[pair.j]);
            let at = place(pair.r, 1.0 / self.table.dr, self.table.nr);
            let rhoip = Knot::new(at_j, at).slope();
            let rhojp = Knot::new(at_i, at).slope();
            let z2 = Knot::new(z2, at);
            let recip = 1.0 / pair.r;
            let phi = z2.value() * recip;
            let phip = z2.slope() * recip - phi * recip;
            let fscale = -(fprime[pair.i] * rhojp + fprime[pair.j] * rhoip + phip) * recip;
            energy += phi;
            for k in 0..3 {
                forces[3 * pair.i + k] += fscale * pair.d[k];
                forces[3 * pair.j + k] -= fscale * pair.d[k];
            }
        }

        if !energy.is_finite() || forces.iter().any(|f| !f.is_finite()) {
            return Err(EamTableError::NonFinite);
        }
        Ok(EamEvaluation { energy, forces })
    }

    /// The potential as a [`PotentialImpl`], the handle the rest of the crate
    /// evaluates through. The handle owns the potential.
    pub fn into_potential(self) -> PotentialImpl {
        PotentialImpl::new(
            eam_table_callback,
            Box::into_raw(Box::new(self)).cast::<c_void>(),
            Some(drop_eam_table),
        )
    }
}

/// The nearest image of the displacement `d` in the cell of lattice vectors
/// `cell` (rows) with inverse `inverse`: the fractional coordinates are
/// rounded to the nearest whole cell.
fn nearest_image(d: [f64; 3], cell: &[f64; 9], inverse: &[f64; 9]) -> [f64; 3] {
    let mut s = [0.0; 3];
    for (k, v) in s.iter_mut().enumerate() {
        let fractional = d[0] * inverse[k] + d[1] * inverse[3 + k] + d[2] * inverse[6 + k];
        *v = fractional - fractional.round();
    }
    let mut out = [0.0; 3];
    for (k, v) in out.iter_mut().enumerate() {
        *v = s[0] * cell[k] + s[1] * cell[3 + k] + s[2] * cell[6 + k];
    }
    out
}

/// Drops the `Box<T>` behind `user_data`. A panic in the drop is caught and
/// recorded in the last-error slot: the entry points that return a status
/// have the same guard, and a free function has no status to carry it.
///
/// # Safety
///
/// `user_data` is null, or it came from `Box::<T>::into_raw` and has not
/// been freed.
unsafe fn drop_boxed<T>(user_data: *mut c_void) {
    use crate::status::catch_unwind;

    if user_data.is_null() {
        return;
    }
    let _ = catch_unwind(std::panic::AssertUnwindSafe(|| {
        drop(unsafe { Box::from_raw(user_data.cast::<T>()) });
        rgpot_status_t::RGPOT_SUCCESS
    }));
}

/// Frees the potential of [`EamTablePotential::into_potential`] and of
/// `rgpot_potential_new_eam_table`.
pub(crate) unsafe extern "C" fn drop_eam_table(user_data: *mut c_void) {
    unsafe { drop_boxed::<EamTablePotential>(user_data) };
}

/// The `PotentialCallback` of a handle made from an [`EamTablePotential`].
pub(crate) unsafe extern "C" fn eam_table_callback(
    user_data: *mut c_void,
    input: *const crate::types::rgpot_force_input_t,
    output: *mut crate::types::rgpot_force_out_t,
) -> rgpot_status_t {
    use crate::status::{catch_unwind, set_last_error};
    use crate::tensor::{copy_f64_tensor, copy_i32_tensor};

    catch_unwind(std::panic::AssertUnwindSafe(|| {
        if user_data.is_null() || input.is_null() || output.is_null() {
            set_last_error("EAM table potential: null user data, input or output");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        }
        let potential = unsafe { &*user_data.cast::<EamTablePotential>() };
        let input = unsafe { &*input };
        let Some(n_atoms) = (unsafe { input.n_atoms() }) else {
            set_last_error("EAM table potential: positions tensor has no shape");
            return rgpot_status_t::RGPOT_INVALID_PARAMETER;
        };
        let positions = copy_f64_tensor(input.positions, "positions", 3 * n_atoms, true);
        let numbers = copy_i32_tensor(input.atomic_numbers, "atomic_numbers", n_atoms);
        let cell = if potential.periodic {
            copy_f64_tensor(input.box_matrix, "box_matrix", 9, true)
        } else {
            Ok(vec![0.0; 9])
        };
        let (positions, numbers, cell) = match (positions, numbers, cell) {
            (Ok(positions), Ok(numbers), Ok(cell)) => (positions, numbers, cell),
            (Err(why), _, _) | (_, Err(why), _) | (_, _, Err(why)) => {
                set_last_error(&why);
                return rgpot_status_t::RGPOT_INVALID_PARAMETER;
            }
        };
        let mut matrix = [0.0; 9];
        matrix.copy_from_slice(&cell);
        match potential.force(&positions, &numbers, &matrix) {
            Ok(answer) => {
                let forces = create_owned_f64_tensor(answer.forces, vec![n_atoms as i64, 3]);
                unsafe {
                    (*output).forces = forces;
                    (*output).energy = answer.energy;
                    (*output).variance = 0.0;
                }
                rgpot_status_t::RGPOT_SUCCESS
            }
            Err(why) => {
                set_last_error(&why.to_string());
                why.status()
            }
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two-element eam/fs table of constants: F_A = -rho, F_B = -2 rho, no
    /// pair term, rho_AA = 1, rho_BA = 3, rho_AB = 5, rho_BB = 7, with
    /// rho_ab the density a b neighbor puts at an a site. The same table as
    /// `two_element_fs` of `CppCore/tests/EAMTablePotTest.cc`.
    fn two_element_fs() -> String {
        let row = |f: &dyn Fn(usize) -> f64| {
            let mut line: String = (0..11).map(|m| format!("{} ", f(m))).collect();
            line.push('\n');
            line
        };
        let mut s = String::from("c\nc\nc\n2 A B\n11 1.0 11 0.5 4.0\n");
        s += "13 1.0 1.0 fcc\n";
        s += &row(&|m| -(m as f64));
        s += &row(&|_| 1.0);
        s += &row(&|_| 3.0);
        s += "28 1.0 1.0 fcc\n";
        s += &row(&|m| -2.0 * m as f64);
        s += &row(&|_| 5.0);
        s += &row(&|_| 7.0);
        for _ in 0..3 {
            s += &row(&|_| 0.0);
        }
        s
    }

    fn dimer(pot: &EamTablePotential, z: [i32; 2]) -> f64 {
        let positions = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        pot.force(&positions, &z, &[0.0; 9]).unwrap().energy
    }

    #[test]
    fn finnis_sinclair_density_blocks_follow_the_lammps_order() {
        let table = EamTable::from_text(&two_element_fs(), EamTableFormat::FinnisSinclair).unwrap();
        assert_eq!(table.atomic_numbers(), &[13, 28]);
        let pot = EamTablePotential::new(table, false).unwrap();
        assert_eq!(dimer(&pot, [13, 13]), -2.0);
        assert_eq!(dimer(&pot, [13, 28]), -5.0 - 2.0 * 3.0);
        assert_eq!(dimer(&pot, [28, 28]), -28.0);
    }

    #[test]
    fn a_format_code_outside_the_constants_is_refused() {
        assert!(matches!(
            EamTableFormat::from_code(9),
            Err(EamTableError::Invalid(_))
        ));
        assert_eq!(
            EamTableFormat::from_code(RGPOT_EAM_TABLE_SETFL),
            Ok(EamTableFormat::Setfl)
        );
    }

    #[test]
    fn auto_reads_the_format_from_the_suffix() {
        let pick = |name: &str| EamTableFormat::Auto.layout(Path::new(name));
        assert_eq!(pick("Cu_u3.eam"), Ok(Layout::Funcfl));
        assert_eq!(pick("CuNi.EAM.ALLOY"), Ok(Layout::Setfl));
        assert_eq!(pick("x.setfl"), Ok(Layout::Setfl));
        assert_eq!(pick("x.eam.fs"), Ok(Layout::FinnisSinclair));
        assert!(matches!(pick("x.dat"), Err(EamTableError::Invalid(_))));
    }

    #[test]
    fn a_short_array_is_malformed() {
        let mut text = two_element_fs();
        text.truncate(text.len() - 20);
        assert!(matches!(
            EamTable::from_text(&text, EamTableFormat::FinnisSinclair),
            Err(EamTableError::Malformed(_))
        ));
    }

    #[test]
    fn a_grid_count_the_file_does_not_back_is_malformed_without_allocating_it() {
        let text = "c\nc\nc\n1 A\n2000000000 1.0 11 0.5 4.0\n13 1.0 1.0 fcc\n1.0 2.0\n";
        assert!(matches!(
            EamTable::from_text(text, EamTableFormat::Setfl),
            Err(EamTableError::Malformed(_))
        ));
    }

    #[test]
    fn an_unknown_atomic_number_is_a_geometry_error() {
        let table = EamTable::from_text(&two_element_fs(), EamTableFormat::FinnisSinclair).unwrap();
        let pot = EamTablePotential::new(table, false).unwrap();
        let positions = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let err = pot.force(&positions, &[13, 29], &[0.0; 9]).unwrap_err();
        assert!(matches!(err, EamTableError::Geometry(_)), "{err}");
        assert!(err.to_string().contains("29"));
    }

    #[test]
    fn a_cell_under_twice_the_cutoff_is_refused() {
        let table = EamTable::from_text(&two_element_fs(), EamTableFormat::FinnisSinclair).unwrap();
        let pot = EamTablePotential::new(table, true).unwrap();
        let positions = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let small = [7.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 7.0];
        let err = pot.force(&positions, &[13, 13], &small).unwrap_err();
        assert!(err.to_string().contains("half a face spacing"), "{err}");
        let flat = [0.0; 9];
        let err = pot.force(&positions, &[13, 13], &flat).unwrap_err();
        assert!(err.to_string().contains("needs a cell"), "{err}");
    }

    struct PanicsOnDrop;

    impl Drop for PanicsOnDrop {
        fn drop(&mut self) {
            panic!("drop panicked");
        }
    }

    fn last_error() -> String {
        let ptr = unsafe { crate::status::rgpot_last_error() };
        unsafe { std::ffi::CStr::from_ptr(ptr) }
            .to_str()
            .unwrap()
            .to_owned()
    }

    #[test]
    fn a_panic_in_the_drop_is_caught_and_recorded() {
        crate::status::set_last_error("");
        let user_data = Box::into_raw(Box::new(PanicsOnDrop)).cast::<c_void>();
        unsafe { drop_boxed::<PanicsOnDrop>(user_data) };
        assert_eq!(last_error(), "drop panicked");
    }

    #[test]
    fn freeing_null_or_a_potential_does_not_touch_the_error() {
        crate::status::set_last_error("untouched");
        unsafe { drop_eam_table(std::ptr::null_mut()) };
        let table = EamTable::from_text(&two_element_fs(), EamTableFormat::FinnisSinclair).unwrap();
        let pot = EamTablePotential::new(table, false).unwrap();
        let user_data = Box::into_raw(Box::new(pot)).cast::<c_void>();
        unsafe { drop_eam_table(user_data) };
        assert_eq!(last_error(), "untouched");
    }

    #[test]
    fn the_nearest_image_folds_a_displacement_across_the_cell() {
        let table = EamTable::from_text(&two_element_fs(), EamTableFormat::FinnisSinclair).unwrap();
        let pot = EamTablePotential::new(table, true).unwrap();
        let cell = [10.0, 0.0, 0.0, 1.0, 10.0, 0.0, 0.0, 0.5, 10.0];
        let near = [0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let wrapped = [10.0, 0.0, 0.0, 12.0, 0.0, 0.0];
        let direct = pot.force(&near, &[13, 28], &cell).unwrap();
        let folded = pot.force(&wrapped, &[13, 28], &cell).unwrap();
        assert_eq!(direct.energy, folded.energy);
        // The second atom sits one cell vector from its image, not two.
        let across = [0.0, 0.0, 0.0, 9.0, 0.0, 0.0];
        let pairs = pot.pairs(&across, &cell).unwrap();
        assert_eq!(pairs.len(), 1);
        assert!((pairs[0].r - 1.0).abs() < f64::EPSILON * 16.0);
    }
}
