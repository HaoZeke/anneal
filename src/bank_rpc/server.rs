//! Single-threaded acceptor; each client is a thread sharing the bank.

use std::io::Write;
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};

use capnp::message::{Builder, ReaderOptions};
use capnp::serialize;
use ndarray::{Array1, ArrayView1};

use crate::Bank_capnp::{bank_reply, bank_request};
use crate::funnel_bo::FunnelModel;
use crate::methods::bank::{Admission, Bank};

/// Hausdorff length below which two members are the same geometry.
const IRA_SAME: f64 = 0.05;
/// AS-KMC / MCAMC: well height at which the packing is a raised
/// superbasin (N_f = 20 deposits of 0.25). Intra-well hops are then
/// rate-scaled and the well is absorbing: sample will not start there.
const SUPERBASIN_HEIGHT: f64 = 5.0;

/// Packing merge for wells. Same number as recommended SOAP packing.
fn pack_merge() -> f64 {
    #[cfg(feature = "featomic")]
    {
        crate::featomic_hop::SOAP_PACK_MERGE
    }
    #[cfg(not(feature = "featomic"))]
    {
        0.10
    }
}

fn soap_l2(a: ArrayView1<f64>, b: ArrayView1<f64>) -> f64 {
    if a.len() != b.len() || a.is_empty() {
        return f64::INFINITY;
    }
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

struct Well {
    soap: Array1<f64>,
    height: f64,
}

struct Inner {
    bank: Bank,
    soaps: Vec<Array1<f64>>,
    wells: Vec<Well>,
    funnel: FunnelModel,
    seeded: usize,
    capacity: usize,
}

impl Inner {
    fn new(capacity: usize) -> Self {
        Self {
            bank: Bank::new(capacity, 1.0),
            soaps: Vec::new(),
            wells: Vec::new(),
            // Length 0.15 sits on the unit-SOAP packing gap (ico–Marks 0.163).
            funnel: FunnelModel::new(0.15, 20.0, 1e-2),
            seeded: 0,
            capacity,
        }
    }

    fn observe(&mut self, soap: ArrayView1<f64>, energy: f64) {
        if !soap.is_empty() {
            self.funnel.observe(soap, energy);
        }
    }

    fn offer(&mut self, energy: f64, coords: Array1<f64>, soap: Array1<f64>) -> (u16, f64) {
        let kind = if self.seeded < self.capacity {
            let merge = pack_merge();
            if let Some(i) = self
                .soaps
                .iter()
                .position(|s| soap_l2(soap.view(), s.view()) <= merge)
            {
                // Same packing: one first-bank slot, keep the deeper energy.
                if energy
                    < self
                        .bank
                        .members()
                        .get(i)
                        .map(|m| m.energy)
                        .unwrap_or(f64::INFINITY)
                {
                    self.bank.overwrite(i, coords.view(), energy);
                    self.observe(soap.view(), energy);
                    self.soaps[i] = soap;
                    1
                } else {
                    2
                }
            } else if self.bank.seed(coords.view(), energy) {
                self.observe(soap.view(), energy);
                self.soaps.push(soap);
                self.seeded += 1;
                if self.seeded == self.capacity {
                    self.refresh_dcut();
                }
                0
            } else {
                4
            }
        } else {
            let soaps = self.soaps.clone();
            let cand = soap.clone();
            let states: Vec<Array1<f64>> = self
                .bank
                .members()
                .iter()
                .map(|m| m.state.clone())
                .collect();
            let admission = self.bank.offer(coords.view(), energy, |p, q| {
                // IRA: same geometry under rotation/permutation (a length).
                // SOAP: Lee Dcut among distinct packings (unit mean-SOAP).
                // They are not substitutes. Mixing IRA lengths into a SOAP
                // Dcut makes every offer novel or every offer a duplicate.
                #[cfg(feature = "ira")]
                {
                    let n = p.len() / 3;
                    let ira = crate::shape::IraMetric {
                        kmax_factor: if n >= 55 { 2.5 } else { 1.8 },
                    };
                    let ira_d = ira.distance(p, q);
                    if ira_d.is_finite() && ira_d <= IRA_SAME {
                        return 0.0;
                    }
                }
                let _ = p;
                states
                    .iter()
                    .position(|s| s.len() == q.len() && s.iter().zip(q.iter()).all(|(a, b)| a == b))
                    .and_then(|i| soaps.get(i))
                    .filter(|s| !s.is_empty() && !cand.is_empty())
                    .map(|s| soap_l2(cand.view(), s.view()))
                    .unwrap_or(f64::INFINITY)
            });
            self.observe(cand.view(), energy);
            match admission {
                Admission::Added(i) => {
                    if i == self.soaps.len() {
                        self.soaps.push(soap);
                    }
                    0
                }
                Admission::Improved(i) => {
                    if i < self.soaps.len() {
                        self.soaps[i] = soap;
                    }
                    1
                }
                Admission::Duplicate(_) => 2,
                Admission::Displaced(i) => {
                    if i < self.soaps.len() {
                        self.soaps[i] = soap;
                    }
                    3
                }
                Admission::Rejected => 4,
            }
        };
        (kind, self.bank.dcut)
    }

    fn refresh_dcut(&mut self) {
        let n = self.soaps.len();
        let fb = featomic_hop_fallback();
        if n < 2 {
            self.bank.dcut = fb;
            return;
        }
        let mut s = 0.0;
        let mut c = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                let d = soap_l2(self.soaps[i].view(), self.soaps[j].view());
                if d.is_finite() {
                    s += d;
                    c += 1.0;
                }
            }
        }
        let mean = if c > 0.0 { s / c } else { 0.0 };
        let start = if mean > 1e-12 { mean * 0.5 } else { fb };
        // Floor is the packing merge, not the 0.05 leftover fallback.
        // 0.05 splits ico isomers (gap ~0.06) and fills 30 slots with
        // one funnel. 0.10 merges those isomers and keeps Marks (0.163).
        self.bank.dcut = start.max(pack_merge()).max(fb);
    }

    fn nearest(&self, soap: ArrayView1<f64>) -> f64 {
        self.wells
            .iter()
            .map(|w| soap_l2(soap, w.soap.view()))
            .fold(f64::INFINITY, f64::min)
    }

    fn nearest_state(&self, coords: ArrayView1<f64>) -> f64 {
        #[cfg(feature = "ira")]
        {
            let ira = crate::shape::IraMetric::default();
            return self
                .bank
                .members()
                .iter()
                .map(|m| ira.distance(coords, m.state.view()))
                .fold(f64::INFINITY, f64::min);
        }
        #[cfg(not(feature = "ira"))]
        {
            let _ = coords;
            f64::INFINITY
        }
    }

    fn deposit(&mut self, soap: Array1<f64>, increment: f64) -> f64 {
        if !(increment.is_finite() && increment > 0.0) || soap.is_empty() {
            return 0.0;
        }
        let merge = pack_merge();
        if let Some(w) = self
            .wells
            .iter_mut()
            .find(|w| soap_l2(soap.view(), w.soap.view()) <= merge)
        {
            // Raised superbasin: stop filling (AS-KMC resets sightings
            // after the barrier is raised). Height stays at the cap.
            if w.height >= SUPERBASIN_HEIGHT {
                return w.height;
            }
            w.height += increment;
            return w.height;
        }
        self.wells.push(Well {
            height: increment,
            soap,
        });
        increment
    }

    fn bias_of(&self, soap: ArrayView1<f64>) -> f64 {
        let merge = pack_merge();
        self.wells
            .iter()
            .find(|w| soap_l2(soap, w.soap.view()) <= merge)
            .map(|w| w.height)
            .unwrap_or(0.0)
    }

    fn sample(&mut self, seed: u64) -> Option<(f64, Array1<f64>)> {
        let n = self.bank.len();
        if n == 0 {
            return None;
        }
        let merge = pack_merge();
        // Shared GP over packing SOAP. EI says where to go (unvisited
        // morphology scores on variance). UCB on well height is the
        // fallback before three distinct observations. A raised well is
        // skipped so another chain searches an open class. If every
        // class is raised the client still gets a member and hops in
        // the SOAP null of the archive — not a random cluster.
        let use_ei = self.funnel.len() >= 3;
        let e_best = self
            .bank
            .members()
            .iter()
            .map(|m| m.energy)
            .fold(f64::INFINITY, f64::min)
            .min(-1e-12);
        let ln = (1.0 + n as f64).ln();
        let mut best_i = 0usize;
        let mut best_s = f64::NEG_INFINITY;
        for (i, m) in self.bank.members().iter().enumerate() {
            let soap = self.soaps.get(i);
            let h = soap
                .and_then(|s| {
                    self.wells
                        .iter()
                        .find(|w| soap_l2(s.view(), w.soap.view()) <= merge)
                        .map(|w| w.height)
                })
                .unwrap_or(0.0);
            let score = if use_ei {
                if let Some(s) = soap {
                    if !s.is_empty() {
                        self.funnel.expected_improvement(s.view())
                    } else {
                        0.0
                    }
                } else {
                    0.0
                }
            } else {
                let quality = m.energy / e_best;
                let bonus = 0.15 * (ln / (1.0 + h)).sqrt();
                quality + bonus
            };
            let jitter = ((seed.wrapping_mul(i as u64 + 1)) % 1000) as f64 * 1e-6;
            if h >= SUPERBASIN_HEIGHT {
                continue;
            }
            if score + jitter > best_s {
                best_s = score + jitter;
                best_i = i;
            }
        }
        if best_s == f64::NEG_INFINITY {
            // Every class is raised. Hand back the deepest member; the
            // client archive-null hop leaves that packing.
            let i = self
                .bank
                .members()
                .iter()
                .enumerate()
                .min_by(|a, b| {
                    a.1.energy
                        .partial_cmp(&b.1.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);
            let m = &self.bank.members()[i];
            return Some((m.energy, m.state.clone()));
        }
        self.bank.mark_used(best_i);
        let m = &self.bank.members()[best_i];
        Some((m.energy, m.state.clone()))
    }
}

fn featomic_hop_fallback() -> f64 {
    #[cfg(feature = "featomic")]
    {
        crate::featomic_hop::SOAP_DCUT_FALLBACK
    }
    #[cfg(not(feature = "featomic"))]
    {
        0.05
    }
}

/// Listen on `addr` (`host:port`).
pub fn serve(addr: impl AsRef<str>, capacity: usize) -> std::io::Result<()> {
    let listener = TcpListener::bind(addr.as_ref())?;
    listener.set_nonblocking(false)?;
    let inner = Arc::new(Mutex::new(Inner::new(capacity.max(1))));
    eprintln!("bank listening on {} capacity {capacity}", addr.as_ref());
    #[cfg(all(feature = "ira", feature = "featomic"))]
    eprintln!(
        "bank identity: IRA Hausdorff same-state (<={IRA_SAME}), SOAP L2 Lee Dcut, SOAP wells merge {}, FunnelModel EI, AS-KMC/MCAMC raise at {SUPERBASIN_HEIGHT}",
        pack_merge()
    );
    #[cfg(all(feature = "ira", not(feature = "featomic")))]
    eprintln!("bank identity: IRA libira_match Hausdorff (no featomic SOAP)");
    #[cfg(not(feature = "ira"))]
    eprintln!(
        "bank identity: SOAP L2 Lee Dcut, SOAP wells merge {}",
        pack_merge()
    );
    for conn in listener.incoming() {
        let stream = match conn {
            Ok(s) => s,
            Err(e) => {
                eprintln!("bank accept: {e}");
                continue;
            }
        };
        let _ = stream.set_nodelay(true);
        let inner = Arc::clone(&inner);
        std::thread::spawn(move || {
            if let Err(e) = handle(stream, inner) {
                eprintln!("bank client: {e}");
            }
        });
    }
    Ok(())
}

fn handle(mut stream: TcpStream, inner: Arc<Mutex<Inner>>) -> Result<(), String> {
    loop {
        let reader = match serialize::read_message(&mut stream, ReaderOptions::new()) {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };
        let req = reader
            .get_root::<bank_request::Reader>()
            .map_err(|e| e.to_string())?;
        let mut reply = Builder::new_default();
        {
            let mut out = reply.init_root::<bank_reply::Builder>();
            let mut g = inner.lock().map_err(|e| e.to_string())?;
            match req.which().map_err(|e| e.to_string())? {
                bank_request::Offer(o) => {
                    let o = o.map_err(|e| e.to_string())?;
                    let energy = o.get_energy();
                    let coords =
                        Array1::from_iter(o.get_coords().map_err(|e| e.to_string())?.iter());
                    let soap = Array1::from_iter(o.get_soap().map_err(|e| e.to_string())?.iter());
                    let (kind, dcut) = g.offer(energy, coords, soap);
                    out.set_kind(kind);
                    out.set_dcut(dcut);
                    out.set_size(g.bank.len() as u32);
                }
                bank_request::Nearest(s) => {
                    let s = s.map_err(|e| e.to_string())?;
                    let soap = Array1::from_iter(s.iter());
                    out.set_distance(g.nearest(soap.view()));
                    out.set_dcut(g.bank.dcut);
                    out.set_size(g.bank.len() as u32);
                }
                bank_request::Deposit(d) => {
                    let d = d.map_err(|e| e.to_string())?;
                    let soap = Array1::from_iter(d.get_soap().map_err(|e| e.to_string())?.iter());
                    let h = g.deposit(soap, d.get_increment());
                    out.set_height(h);
                    out.set_dcut(g.bank.dcut);
                    out.set_size(g.bank.len() as u32);
                }
                bank_request::BiasOf(s) => {
                    let s = s.map_err(|e| e.to_string())?;
                    let soap = Array1::from_iter(s.iter());
                    out.set_height(g.bias_of(soap.view()));
                    out.set_dcut(g.bank.dcut);
                    out.set_size(g.bank.len() as u32);
                }
                bank_request::Sample(seed) => {
                    match g.sample(seed) {
                        Some((e, x)) => {
                            out.set_empty(false);
                            out.set_energy(e);
                            let mut c = out.reborrow().init_coords(x.len() as u32);
                            for (i, &v) in x.iter().enumerate() {
                                c.set(i as u32, v);
                            }
                        }
                        None => out.set_empty(true),
                    }
                    out.set_dcut(g.bank.dcut);
                    out.set_size(g.bank.len() as u32);
                }
                bank_request::Snapshot(()) => {
                    let n = g.wells.len() as u32;
                    {
                        let mut ws = out.reborrow().init_wells(n);
                        for (i, w) in g.wells.iter().enumerate() {
                            let mut slot = ws.reborrow().get(i as u32);
                            slot.set_height(w.height);
                            let mut s = slot.init_soap(w.soap.len() as u32);
                            for (k, &v) in w.soap.iter().enumerate() {
                                s.set(k as u32, v);
                            }
                        }
                    }
                    let mut es = out.reborrow().init_energies(g.bank.len() as u32);
                    for (i, m) in g.bank.members().iter().enumerate() {
                        es.set(i as u32, m.energy);
                    }
                    out.set_dcut(g.bank.dcut);
                    out.set_size(g.bank.len() as u32);
                }
                bank_request::SetDcut(d) => {
                    if g.seeded >= g.capacity && d.is_finite() && d > 0.0 {
                        g.bank.dcut = d.max(pack_merge());
                    }
                    out.set_dcut(g.bank.dcut);
                    out.set_size(g.bank.len() as u32);
                }
            }
        }
        serialize::write_message(&mut stream, &reply).map_err(|e| e.to_string())?;
        stream.flush().map_err(|e| e.to_string())?;
    }
}
