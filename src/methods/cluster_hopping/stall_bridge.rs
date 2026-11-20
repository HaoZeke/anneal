//! Stalled rigid-body archive walk.
//!
//! The loop is compiled in `anneal-bridge` and reached through function
//! pointers. The pointers are the helpers this crate already had, so the
//! loop adds no body for a molecular Tric step to inherit.

use ndarray::{Array1, ArrayView1};
use rand::rngs::StdRng;

use super::{ClusterMove, Config, Ledger};
use crate::allocate::FlooredThompson;
use anneal_bridge::Ops;

static OPS: Ops = Ops {
    offer,
    remaining,
    stopped,
    spent,
    best_energy,
    has_best,
    copy_best,
    cage_key: super::rigid_cage_key,
    recentre: super::recentre,
    identity: super::hop_is_identity,
    contain,
    has_geometry,
    reverses_cycle,
    water_sites,
    lab_sites,
    donations: super::cycles::donations,
    frame_rotation: super::cycles::frame_rotation,
    compose_rotation: crate::potentials::compose_rotation,
    stem_window,
    propose,
    relax: relax_point,
    record,
    reset_bandit,
    bandit_select,
    bandit_update,
    n_points,
    temperature,
    relax_steps,
    n_kernels,
};

/// Walk a stored cage after a rigid chain has stalled.
///
/// The hop calls this only when the rigid bridge is armed.
#[inline(never)]
#[cold]
pub(super) fn walk_armed_rigid_archive(
    cfg: &Config,
    ledger: &mut Ledger,
    relax: &mut (dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send),
    cage_archive: &mut crate::methods::stall_archive::StallArchive,
    kernels: &[ClusterMove],
    propose: ProposeStd,
    e: &mut f64,
    x: &mut Array1<f64>,
    hops: usize,
    cage_quantum: f64,
    orient_scale: f64,
    new_lowest: f64,
    archive_end: usize,
) {
    let mut ctx = Ctx {
        cfg,
        ledger,
        relax,
        cage_archive,
        kernels,
        propose,
        bandit: None,
    };
    if let Some((next_e, next_x)) = anneal_bridge::walk(
        &OPS,
        (&mut ctx as *mut Ctx<'_>).cast(),
        *e,
        x.view(),
        hops,
        cage_quantum,
        orient_scale,
        new_lowest,
        archive_end,
    ) {
        *e = next_e;
        *x = next_x;
    }
}

struct Ctx<'a> {
    cfg: &'a Config,
    ledger: &'a mut Ledger,
    relax: &'a mut (dyn FnMut(&mut Ledger, ArrayView1<f64>, usize) -> (f64, Array1<f64>) + Send),
    cage_archive: &'a mut crate::methods::stall_archive::StallArchive,
    kernels: &'a [ClusterMove],
    propose: ProposeStd,
    bandit: Option<FlooredThompson>,
}

type ProposeStd = fn(&ClusterMove, ArrayView1<f64>, f64, f64, &mut StdRng) -> Array1<f64>;

/// The walk calls one pointer at a time and does not hold the pointer
/// across that call, so the context is exclusively borrowed inside it.
#[cold]
fn ctx<'a>(raw: *mut u8) -> &'a mut Ctx<'a> {
    unsafe { &mut *raw.cast() }
}

#[cold]
fn offer(raw: *mut u8, energy: f64, x: ArrayView1<f64>, key: u64) {
    let ctx = ctx(raw);
    let put: fn(&mut crate::methods::stall_archive::StallArchive, f64, ArrayView1<f64>, u64) =
        crate::methods::stall_archive::StallArchive::offer_cage;
    call(put)(ctx.cage_archive, energy, x, key);
}

#[cold]
fn reverses_cycle(raw: *mut u8, arm: usize) -> bool {
    ctx(raw).kernels[arm].reverses_directed_cycle()
}

#[cold]
fn call<T>(f: T) -> T {
    std::hint::black_box(f)
}

#[cold]
fn copy_point(x: &Array1<f64>) -> Array1<f64> {
    call::<fn(&Array1<f64>) -> Array1<f64>>(Clone::clone)(x)
}

#[cold]
fn has_best(raw: *mut u8) -> bool {
    ctx(raw).ledger.best_state.is_some()
}

#[cold]
fn copy_best(raw: *mut u8) -> Array1<f64> {
    copy_point(ctx(raw).ledger.best_state.as_ref().unwrap())
}

#[cold]
fn remaining(raw: *mut u8) -> usize {
    call::<fn(&Ledger) -> usize>(Ledger::remaining)(ctx(raw).ledger)
}

#[cold]
fn stopped(raw: *mut u8) -> bool {
    call::<fn(&Ledger) -> bool>(Ledger::stopped)(ctx(raw).ledger)
}

#[cold]
fn spent(raw: *mut u8) -> usize {
    ctx(raw).ledger.spent()
}

#[cold]
fn best_energy(raw: *mut u8) -> f64 {
    ctx(raw).ledger.best
}

#[cold]
fn has_geometry(raw: *mut u8) -> bool {
    ctx(raw).cfg.rigid_body_geometry.is_some()
}

#[cold]
fn water_sites(raw: *mut u8) -> Option<(usize, [usize; 2])> {
    let geometry = ctx(raw).cfg.rigid_body_geometry.as_ref()?;
    let sites: fn(&crate::rigid_body::RigidBodyGeometry) -> Option<(usize, [usize; 2])> =
        super::cycles::water_sites;
    call(sites)(geometry)
}

#[cold]
fn lab_sites(raw: *mut u8, x: ArrayView1<f64>) -> Vec<Vec<[f64; 3]>> {
    let ctx = ctx(raw);
    let geometry = ctx.cfg.rigid_body_geometry.as_ref().unwrap();
    let n = ctx.cfg.n_points;
    let place: fn(
        &crate::rigid_body::RigidBodyGeometry,
        usize,
        ArrayView1<f64>,
    ) -> Vec<Vec<[f64; 3]>> = super::cycles::lab_sites;
    call(place)(geometry, n, x)
}

#[cold]
fn stem_window(raw: *mut u8) -> f64 {
    let Some(geometry) = ctx(raw).cfg.rigid_body_geometry.as_ref() else {
        return 0.0;
    };
    super::cycles::stem_window(geometry)
}

#[cold]
fn contain(raw: *mut u8, x: &mut Array1<f64>) {
    let ctx = ctx(raw);
    let put: fn(&Config, &mut Array1<f64>, usize) = super::contain_trial;
    put(ctx.cfg, x, ctx.cfg.n_points);
}

#[cold]
fn propose(
    raw: *mut u8,
    arm: usize,
    x: ArrayView1<f64>,
    temperature: f64,
    scale: f64,
    rng: &mut StdRng,
) -> Array1<f64> {
    let ctx = ctx(raw);
    (ctx.propose)(&ctx.kernels[arm], x, temperature, scale, rng)
}

#[cold]
fn relax_point(raw: *mut u8, x: ArrayView1<f64>, steps: usize) -> (f64, Array1<f64>) {
    let ctx = ctx(raw);
    (ctx.relax)(ctx.ledger, x, steps)
}

#[cold]
fn record(raw: *mut u8, energy: f64, x: ArrayView1<f64>) {
    let write: fn(&mut Ledger, f64, ArrayView1<f64>) = Ledger::record;
    write(ctx(raw).ledger, energy, x);
}

#[cold]
fn reset_bandit(raw: *mut u8, arms: usize) {
    let fresh = call::<fn(usize) -> FlooredThompson>(FlooredThompson::new)(arms);
    let old = ctx(raw).bandit.replace(fresh);
    let drop_old: fn(Option<FlooredThompson>) = std::mem::drop;
    call(drop_old)(old);
}

#[cold]
fn bandit_select(raw: *mut u8, rng: &mut StdRng) -> usize {
    let choose: fn(&mut FlooredThompson, &mut StdRng) -> usize = FlooredThompson::select;
    choose(ctx(raw).bandit.as_mut().expect("bandit"), rng)
}

#[cold]
fn bandit_update(raw: *mut u8, arm: usize, reward: bool) {
    let note: fn(&mut FlooredThompson, usize, bool) = FlooredThompson::update;
    note(ctx(raw).bandit.as_mut().expect("bandit"), arm, reward);
}

#[cold]
fn n_points(raw: *mut u8) -> usize {
    ctx(raw).cfg.n_points
}

#[cold]
fn temperature(raw: *mut u8) -> f64 {
    ctx(raw).cfg.temperature
}

#[cold]
fn relax_steps(raw: *mut u8) -> usize {
    ctx(raw).cfg.relax_steps
}

#[cold]
fn n_kernels(raw: *mut u8) -> usize {
    ctx(raw).kernels.len()
}
