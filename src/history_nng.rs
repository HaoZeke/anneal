//! nng REQ/REP for [`crate::methods::minima_hopping::MinimumHistory`].
//!
//! One server owns the table and the exact witness. Clients implement
//! [`HistoryHook`]: they send a certified (energy, state, gradient) and
//! get identity plus visit counts back. No coordinates are returned.
//! This is not the census bus (Pub/Sub of live Cartesian minima), not
//! catalog VatNetwork, not bank Cap'n, not a Unix-stream pump, and not
//! UDP: admit is not best-effort. Use `ipc://` on one node and `tcp://`
//! when the table is off-node.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

use ndarray::{Array1, ArrayView1};
use nng::options::{Options, RecvTimeout};
use nng::{Message, Protocol, Socket};

use crate::descriptor_space::DescriptorVector;
use crate::methods::cluster_hopping::QuenchBoundary;
use crate::methods::minima_hopping::{
    HistoryHook, HistoryMembership, HistoryReport, MinimumHistory, history_feedback_membership,
};
use crate::pes_exploration::{ExactStructureWitness, StructureContext};

const TAG_OBSERVE: u8 = b'O';
const TAG_ACCEPT: u8 = b'A';
const TAG_COUNT: u8 = b'C';
const TAG_REPORT: u8 = b'R';
const TAG_OK: u8 = b'K';
const TAG_NONE: u8 = b'N';
const TAG_FAIL: u8 = b'X';

/// Transport or protocol failure. The search must not pretend it shared.
#[derive(Debug, thiserror::Error)]
#[error("history nng: {0}")]
pub struct HistoryNngError(String);

/// Scaled max-norm witness on a box. Same rule as the in-process box hop.
struct WidthWitness {
    widths: Array1<f64>,
    identity_tol: f64,
}

impl ExactStructureWitness for WidthWitness {
    fn equivalent(&self, left: ArrayView1<f64>, right: ArrayView1<f64>) -> bool {
        if left.len() != right.len() || left.len() != self.widths.len() {
            return false;
        }
        left.iter()
            .zip(right.iter())
            .zip(self.widths.iter())
            .all(|((&a, &b), &w)| (a - b).abs() <= self.identity_tol * w.max(1e-12))
    }
}

/// Owns one [`MinimumHistory`] and serves observe / mark_accepted.
pub struct HistoryNngServer {
    url: String,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl HistoryNngServer {
    /// Bind `url` (`ipc:///tmp/...` or `tcp://127.0.0.1:port`) and serve.
    pub fn bind(
        url: &str,
        identity_tol: f64,
        gradient_tolerance: f64,
    ) -> Result<Self, HistoryNngError> {
        admit_url(url)?;
        let history = MinimumHistory::new(gradient_tolerance)
            .map_err(|error| HistoryNngError(error.to_string()))?;
        let rep = Socket::new(Protocol::Rep0)
            .map_err(|error| HistoryNngError(format!("rep: {error}")))?;
        let _ = rep.set_opt::<RecvTimeout>(Some(Duration::from_millis(50)));
        rep.listen(url)
            .map_err(|error| HistoryNngError(format!("listen {url}: {error}")))?;
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = Arc::clone(&stop);
        let thread = thread::Builder::new()
            .name("anneal-history-nng".into())
            .spawn(move || serve_loop(rep, history, identity_tol, thread_stop))
            .map_err(|error| HistoryNngError(format!("spawn: {error}")))?;
        Ok(Self {
            url: url.to_string(),
            stop,
            thread: Some(thread),
        })
    }

    /// Endpoint this server is listening on.
    pub fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for HistoryNngServer {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

fn serve_loop(
    rep: Socket,
    mut history: MinimumHistory,
    identity_tol: f64,
    stop: Arc<AtomicBool>,
) {
    let context = StructureContext::new(None, None, Some("design-box".into()));
    while !stop.load(Ordering::Acquire) {
        let frame = match rep.recv() {
            Ok(frame) => frame,
            Err(nng::Error::TimedOut) => continue,
            Err(_) => break,
        };
        let reply = handle(&mut history, identity_tol, &context, &frame);
        let mut message = Message::new();
        message.push_back(&reply);
        if rep.send(message).is_err() {
            break;
        }
    }
}

fn handle(
    history: &mut MinimumHistory,
    identity_tol: f64,
    context: &StructureContext,
    frame: &[u8],
) -> Vec<u8> {
    if frame.is_empty() {
        return vec![TAG_FAIL];
    }
    match frame[0] {
        TAG_OBSERVE => match decode_observe(&frame[1..]) {
            None => vec![TAG_FAIL],
            Some((energy, state, gradient, widths)) => {
                let Some(quench) =
                    QuenchBoundary::validated(energy, state.clone(), gradient.clone())
                else {
                    return vec![TAG_NONE];
                };
                let Ok(description) = DescriptorVector::from_design(state.to_vec()) else {
                    return vec![TAG_NONE];
                };
                let witness = WidthWitness {
                    widths,
                    identity_tol,
                };
                let Ok(observation) =
                    history.observe(&quench, description, context.clone(), &witness)
                else {
                    return vec![TAG_NONE];
                };
                let Some(accepted) = history.accepted_visits(observation.minimum.id) else {
                    return vec![TAG_NONE];
                };
                encode_raw_report(
                    observation.minimum.id,
                    observation.minimum.is_new,
                    observation.visits,
                    accepted,
                )
            }
        },
        TAG_ACCEPT => {
            if frame.len() < 5 {
                return vec![TAG_FAIL];
            }
            let id = u32::from_le_bytes(frame[1..5].try_into().unwrap()) as usize;
            let _ = history.mark_accepted(id);
            vec![TAG_OK]
        }
        TAG_COUNT => {
            let n = history.minimum_count() as u32;
            let mut out = vec![TAG_OK];
            out.extend_from_slice(&n.to_le_bytes());
            out
        }
        _ => vec![TAG_FAIL],
    }
}

fn decode_observe(body: &[u8]) -> Option<(f64, Array1<f64>, Array1<f64>, Array1<f64>)> {
    if body.len() < 12 {
        return None;
    }
    let n = u32::from_le_bytes(body[0..4].try_into().ok()?) as usize;
    let need = 4 + 8 + 8 * 3 * n;
    if body.len() < need {
        return None;
    }
    let energy = f64::from_le_bytes(body[4..12].try_into().ok()?);
    let mut off = 12;
    let mut take = || {
        let v = f64::from_le_bytes(body[off..off + 8].try_into().ok()?);
        off += 8;
        Some(v)
    };
    let mut state = Vec::with_capacity(n);
    let mut gradient = Vec::with_capacity(n);
    let mut widths = Vec::with_capacity(n);
    for _ in 0..n {
        state.push(take()?);
    }
    for _ in 0..n {
        gradient.push(take()?);
    }
    for _ in 0..n {
        widths.push(take()?);
    }
    Some((
        energy,
        Array1::from(state),
        Array1::from(gradient),
        Array1::from(widths),
    ))
}

fn encode_observe(
    energy: f64,
    state: ArrayView1<f64>,
    gradient: ArrayView1<f64>,
    widths: ArrayView1<f64>,
) -> Option<Vec<u8>> {
    if state.len() != gradient.len() || state.len() != widths.len() {
        return None;
    }
    let n = state.len() as u32;
    let mut frame = Vec::with_capacity(13 + 8 * 3 * state.len());
    frame.push(TAG_OBSERVE);
    frame.extend_from_slice(&n.to_le_bytes());
    frame.extend_from_slice(&energy.to_le_bytes());
    for value in state {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    for value in gradient {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    for value in widths {
        frame.extend_from_slice(&value.to_le_bytes());
    }
    Some(frame)
}

fn encode_raw_report(
    minimum: usize,
    first_observation: bool,
    observed_visits: u64,
    accepted_visits: u64,
) -> Vec<u8> {
    let mut out = vec![TAG_REPORT];
    out.extend_from_slice(&(minimum as u32).to_le_bytes());
    out.push(u8::from(first_observation));
    out.extend_from_slice(&observed_visits.to_le_bytes());
    out.extend_from_slice(&accepted_visits.to_le_bytes());
    out
}

fn decode_raw_report(frame: &[u8]) -> Option<(usize, bool, u64, u64)> {
    if frame.len() < 1 + 4 + 1 + 8 + 8 || frame[0] != TAG_REPORT {
        return None;
    }
    Some((
        u32::from_le_bytes(frame[1..5].try_into().ok()?) as usize,
        frame[5] != 0,
        u64::from_le_bytes(frame[6..14].try_into().ok()?),
        u64::from_le_bytes(frame[14..22].try_into().ok()?),
    ))
}

/// REQ client. One outstanding observe at a time. One socket per replica.
pub struct HistoryNngClient {
    socket: Socket,
    widths: Array1<f64>,
    policy: HistoryMembership,
}

impl HistoryNngClient {
    /// Dial `url` and send box side-lengths with every observe.
    pub fn dial(
        url: &str,
        widths: Array1<f64>,
        policy: HistoryMembership,
    ) -> Result<Self, HistoryNngError> {
        admit_url(url)?;
        let socket =
            Socket::new(Protocol::Req0).map_err(|error| HistoryNngError(format!("req: {error}")))?;
        socket
            .dial(url)
            .map_err(|error| HistoryNngError(format!("dial {url}: {error}")))?;
        Ok(Self {
            socket,
            widths,
            policy,
        })
    }

    /// Distinct exact identities currently held by the server.
    pub fn minimum_count(&self) -> Option<usize> {
        let mut message = Message::new();
        message.push_back(&[TAG_COUNT]);
        self.socket.send(message).ok()?;
        let reply = self.socket.recv().ok()?;
        if reply.len() < 5 || reply.first().copied() != Some(TAG_OK) {
            return None;
        }
        Some(u32::from_le_bytes(reply[1..5].try_into().ok()?) as usize)
    }
}

fn admit_url(url: &str) -> Result<(), HistoryNngError> {
    let scheme = url
        .split_once("://")
        .map(|(scheme, _)| scheme)
        .unwrap_or("")
        .to_ascii_lowercase();
    if scheme == "udp" || scheme == "udp4" || scheme == "udp6" {
        return Err(HistoryNngError(
            "udp is not certified admit; use ipc or tcp".into(),
        ));
    }
    if scheme != "ipc" && scheme != "tcp" && scheme != "inproc" {
        return Err(HistoryNngError(format!(
            "unsupported history url scheme {scheme:?}; use ipc or tcp"
        )));
    }
    Ok(())
}

impl HistoryHook for HistoryNngClient {
    fn observe(
        &mut self,
        energy: f64,
        state: ArrayView1<f64>,
        gradient: ArrayView1<f64>,
    ) -> Option<HistoryReport> {
        let _ = self.policy;
        let frame = encode_observe(energy, state, gradient, self.widths.view())?;
        let mut message = Message::new();
        message.push_back(&frame);
        self.socket.send(message).ok()?;
        let reply = self.socket.recv().ok()?;
        if reply.first().copied() == Some(TAG_NONE) {
            return None;
        }
        let (minimum, first_observation, observed_visits, accepted_visits) =
            decode_raw_report(&reply)?;
        let (is_new, visits) = history_feedback_membership(
            self.policy,
            first_observation,
            observed_visits,
            accepted_visits,
        );
        Some(HistoryReport {
            minimum,
            is_new,
            visits,
            observed_visits,
            first_observation,
        })
    }

    fn mark_accepted(&mut self, minimum: usize) {
        let mut frame = vec![TAG_ACCEPT];
        frame.extend_from_slice(&(minimum as u32).to_le_bytes());
        let mut message = Message::new();
        message.push_back(&frame);
        if self.socket.send(message).is_ok() {
            let _ = self.socket.recv();
        }
    }

    fn cost(&self) -> (usize, usize, f64) {
        (0, 0, 0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn two_clients_share_one_nng_history() {
        let url = format!(
            "ipc:///tmp/anneal-hist-nng-{}",
            std::process::id()
        );
        let _server = HistoryNngServer::bind(&url, 1e-3, 1e-3).expect("bind");
        let widths = array![10.0, 10.0];
        let mut first =
            HistoryNngClient::dial(&url, widths.clone(), HistoryMembership::Accepted).expect("a");
        let mut second =
            HistoryNngClient::dial(&url, widths, HistoryMembership::Accepted).expect("b");
        let zero = array![0.0, 0.0];
        let a = first
            .observe(-1.0, array![2.0, 2.0].view(), zero.view())
            .expect("first observe");
        first.mark_accepted(a.minimum);
        let b = second
            .observe(-0.5, array![2.001, 2.0].view(), zero.view())
            .expect("second observe");
        assert_eq!(a.minimum, b.minimum);
        assert!(!b.is_new);
        assert_eq!(b.observed_visits, 2);
    }

    #[test]
    fn two_processes_share_one_nng_history() {
        const CHILD: &str = "HISTORY_NNG_CHILD";
        let widths = array![10.0, 10.0];
        let zero = array![0.0, 0.0];
        if let Ok(url) = std::env::var(CHILD) {
            let mut client =
                HistoryNngClient::dial(&url, widths, HistoryMembership::Accepted).expect("child");
            let report = client
                .observe(-0.5, array![2.001, 2.0].view(), zero.view())
                .expect("child observe");
            assert!(!report.is_new, "child must see the parent's minimum");
            assert_eq!(report.observed_visits, 2);
            return;
        }
        let url = format!("ipc:///tmp/anneal-hist-proc-{}", std::process::id());
        let _server = HistoryNngServer::bind(&url, 1e-3, 1e-3).expect("bind");
        let mut parent =
            HistoryNngClient::dial(&url, widths, HistoryMembership::Accepted).expect("parent");
        let first = parent
            .observe(-1.0, array![2.0, 2.0].view(), zero.view())
            .expect("parent observe");
        parent.mark_accepted(first.minimum);
        let exe = std::env::current_exe().expect("test exe");
        let status = std::process::Command::new(exe)
            .env(CHILD, &url)
            .args([
                "--exact",
                "history_nng::tests::two_processes_share_one_nng_history",
            ])
            .status()
            .expect("spawn child");
        assert!(status.success(), "child {status}");
    }

    #[test]
    fn udp_url_is_rejected() {
        match HistoryNngServer::bind("udp://127.0.0.1:9", 1e-3, 1e-3) {
            Err(err) => assert!(err.to_string().contains("udp")),
            Ok(_) => panic!("udp bind must fail"),
        }
        match HistoryNngClient::dial(
            "udp://127.0.0.1:9",
            array![1.0],
            HistoryMembership::Accepted,
        ) {
            Err(err) => assert!(err.to_string().contains("udp")),
            Ok(_) => panic!("udp dial must fail"),
        }
    }
}
