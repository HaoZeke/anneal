//! nng carrier for Cap'n Proto.
//!
//! Cap'n keeps the schema, the vat (catalog Session/Subscriber), and the
//! journal. nng is the only byte carrier:
//!
//! * bank: one Req/Rep message per Cap'n call
//! * catalog: advertised Rep hello, then one Pair vat per replica
//! * same-node Pair is `ipc://` (no extra TCP port per chain)
//! * `host:port` still means `tcp://host:port` so CATALOG_RPC / BANK_RPC
//!   keep their current env shape
//!
//! Census and decree already speak nng. This module does not replace them.

use std::collections::VecDeque;
use std::io::{self, IoSlice, Read, Write};
use std::net::SocketAddr;
use std::os::unix::io::{AsRawFd, RawFd};
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll, Waker};
use std::time::{Duration, Instant};

use nng::options::protocol::reqrep::ResendTime;
use nng::options::transport::tcp::{BoundPort, NoDelay};
use nng::options::{LocalAddr, Options, RecvFd, RecvMaxSize, RecvTimeout, SendFd, SendTimeout};
use nng::{Listener, ListenerBuilder, PipeEvent, Protocol, Socket};
use tokio::io::unix::AsyncFd;
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};

const HELLO: &[u8] = b"ANNEAL-PAIR";
static PAIR_SEQ: AtomicU64 = AtomicU64::new(1);

/// Listening socket plus the listener handle that must outlive it.
pub struct BoundListen {
    /// Bound nng socket.
    pub socket: Socket,
    /// Keeps the listener alive. Dropping it unbinds.
    pub listener: Listener,
    /// Canonical nng URL including the allocated port.
    pub url: String,
    /// TCP bind address when the URL is `tcp://`.
    pub addr: Option<SocketAddr>,
}

/// One side of a catalog vat stream.
pub struct PairSession {
    /// Pair socket.
    pub socket: Socket,
    _listener: Option<Listener>,
}

/// Turn `host:port` into `tcp://host:port`. Pass through nng URLs.
pub fn url(spec: &str) -> String {
    let spec = spec.trim();
    if spec.contains("://") {
        spec.to_string()
    } else {
        format!("tcp://{spec}")
    }
}

/// Listen with `protocol` on `spec` (`host:port` or an nng URL).
pub fn listen(protocol: Protocol, spec: &str) -> io::Result<BoundListen> {
    let requested = url(spec);
    let socket = Socket::new(protocol).map_err(nng_io)?;
    socket
        .set_opt::<RecvMaxSize>(if matches!(protocol, Protocol::Pair0) {
            FRAME_HEADER + FRAME_PAYLOAD
        } else {
            0
        })
        .map_err(nng_io)?;
    let builder = ListenerBuilder::new(&socket, &requested).map_err(nng_io)?;
    let _ = builder.set_opt::<NoDelay>(true);
    let listener = match builder.start() {
        Ok(listener) => listener,
        Err((_, error)) => return Err(nng_io(error)),
    };
    let (bound_url, addr) = bound_endpoint(&requested, &listener)?;
    if bound_url.starts_with("tcp://") && tcp_port(&bound_url) == "0" {
        return Err(io::Error::new(
            io::ErrorKind::AddrNotAvailable,
            format!("nng bound port still 0 for {requested}"),
        ));
    }
    Ok(BoundListen {
        socket,
        listener,
        url: bound_url,
        addr,
    })
}

/// Accept one catalog vat. Completes the Rep hello and returns the Pair.
pub fn accept_pair(accept: &Socket) -> io::Result<PairSession> {
    let hello = accept.recv().map_err(nng_io)?;
    finish_pair(accept, hello)
}

/// Non-blocking hello. `Ok(None)` means try again.
pub fn try_accept_pair(accept: &Socket) -> io::Result<Option<PairSession>> {
    match accept.try_recv() {
        Ok(hello) => finish_pair(accept, hello).map(Some),
        Err(nng::Error::TryAgain) => Ok(None),
        Err(error) => Err(nng_io(error)),
    }
}

/// Pollable recv fd. Do not read or write it.
pub fn recv_fd(socket: &Socket) -> io::Result<AsyncFd<PollFd>> {
    let fd = socket.get_opt::<RecvFd>().map_err(nng_io)?;
    AsyncFd::new(PollFd(fd))
}

fn finish_pair(accept: &Socket, hello: nng::Message) -> io::Result<PairSession> {
    if &hello[..] != HELLO {
        let _ = accept.send(nng::Message::new());
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "catalog pair hello mismatch",
        ));
    }
    let bound = listen(Protocol::Pair0, &pair_bind_spec())?;
    match accept.send(bound.url.as_bytes()) {
        Ok(()) => Ok(PairSession {
            socket: bound.socket,
            _listener: Some(bound.listener),
        }),
        Err((_, error)) => Err(nng_io(error)),
    }
}

/// Dial the advertised catalog URL and attach a Pair vat stream.
pub fn dial_pair(advertised: &str, timeout: Duration) -> io::Result<PairSession> {
    let req = Socket::new(Protocol::Req0).map_err(nng_io)?;
    let _ = req.set_opt::<RecvTimeout>(Some(timeout));
    let _ = req.set_opt::<SendTimeout>(Some(timeout));
    let _ = req.set_opt::<ResendTime>(Some(timeout));
    req.dial(&url(advertised)).map_err(nng_io)?;
    match req.send(HELLO) {
        Ok(()) => {}
        Err((_, error)) => return Err(nng_io(error)),
    }
    let reply = req.recv().map_err(nng_io)?;
    drop(req);
    let bound = std::str::from_utf8(&reply).map_err(|error| {
        io::Error::new(io::ErrorKind::InvalidData, format!("pair url: {error}"))
    })?;
    if bound.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "empty pair url"));
    }
    let pair_url = rewrite_pair_url(advertised, bound);
    let pair = Socket::new(Protocol::Pair0).map_err(nng_io)?;
    pair.set_opt::<RecvMaxSize>(FRAME_HEADER + FRAME_PAYLOAD)
        .map_err(nng_io)?;
    let _ = pair.set_opt::<SendTimeout>(Some(timeout));
    let _ = pair.set_opt::<RecvTimeout>(Some(timeout));
    pair.dial(&pair_url).map_err(nng_io)?;
    let _ = pair.set_opt::<RecvTimeout>(None);
    let _ = pair.set_opt::<SendTimeout>(None);
    Ok(PairSession {
        socket: pair,
        _listener: None,
    })
}

const FRAME_MAGIC: &[u8; 4] = b"ANIO";
const FRAME_VERSION: u8 = 1;
const FRAME_HEADER: usize = 24;
const FRAME_PAYLOAD: usize = 64 * 1024;
const STREAM_WINDOW: usize = 1024 * 1024;
const PUMP_BATCH: usize = 16;
const OPERATION_TIMEOUT: Duration = Duration::from_secs(30);
const DATA: u8 = 1;
const FIN: u8 = 2;
const ACK: u8 = 3;
const ACK_FIN: u8 = 1;
const ACK_FIN_RECEIPT: u8 = 2;

/// Tokio byte stream with bounded queues and versioned nng framing.
///
/// Flush acknowledges acceptance by the peer's carrier queue, not consumption
/// by its application. Shutdown closes only the write half. Pending writes,
/// flushes and shutdowns fail after 30 seconds without their required progress;
/// an idle read has no deadline. Drop drains accepted writes and closes the
/// write half through a detached pump, subject to the same finite close limit.
/// Both endpoints must use this carrier protocol and exclusively own their
/// pair sockets; successful nng sends alone are not delivery acknowledgements.
pub struct NngIo {
    shared: Arc<StreamShared>,
}

impl NngIo {
    /// Wrap an exclusively owned pair socket.
    pub fn new(session: PairSession) -> io::Result<Self> {
        Self::with_timeout(session, OPERATION_TIMEOUT)
    }

    fn with_timeout(session: PairSession, timeout: Duration) -> io::Result<Self> {
        let recv = session.socket.get_opt::<RecvFd>().map_err(nng_io)?;
        let send = session.socket.get_opt::<SendFd>().map_err(nng_io)?;
        let (wake, notified) = std::os::unix::net::UnixStream::pair()?;
        wake.set_nonblocking(true)?;
        notified.set_nonblocking(true)?;
        let shared = Arc::new(StreamShared {
            state: Mutex::new(StreamState::default()),
            wake,
            disconnected: AtomicBool::new(false),
            timeout,
        });
        let weak = Arc::downgrade(&shared);
        session
            .socket
            .pipe_notify(move |_, event| {
                if matches!(event, PipeEvent::RemovePost)
                    && let Some(shared) = weak.upgrade()
                {
                    // Socket callbacks only signal; the pump drains received
                    // frames before resolving disconnect against stream EOF.
                    shared.disconnected.store(true, Ordering::Release);
                    shared.notify();
                }
            })
            .map_err(nng_io)?;
        let pump_shared = Arc::clone(&shared);
        std::thread::Builder::new()
            .name("anneal-nng-pump".into())
            .spawn(move || {
                if let Err(error) = pump_pair(&session, &pump_shared, notified, recv, send) {
                    pump_shared.fail(error);
                }
            })?;
        Ok(Self { shared })
    }
}

#[derive(Default)]
struct StreamState {
    read_queue: VecDeque<u8>,
    write_queue: VecDeque<u8>,
    written: u64,
    sent: u64,
    peer_accepted: u64,
    peer_consumed: u64,
    received: u64,
    consumed: u64,
    write_closed: bool,
    fin_sent: bool,
    fin_acked: bool,
    read_fin: bool,
    fin_receipted: bool,
    receipt_sent: bool,
    dropped: bool,
    ack_dirty: bool,
    flush_goal: Option<(u64, Instant)>,
    write_deadline: Option<Instant>,
    close_deadline: Option<Instant>,
    error: Option<(io::ErrorKind, String)>,
    read_waker: Option<Waker>,
    write_waker: Option<Waker>,
    flush_waker: Option<Waker>,
    close_waker: Option<Waker>,
}

impl StreamState {
    fn error(&self) -> Option<io::Error> {
        self.error
            .as_ref()
            .map(|(kind, message)| io::Error::new(*kind, message.clone()))
    }

    fn deadline(&self) -> Option<Instant> {
        [
            self.flush_goal.map(|(_, deadline)| deadline),
            self.write_deadline,
            self.close_deadline,
        ]
        .into_iter()
        .flatten()
        .min()
    }

    fn take_wakers(&mut self) -> [Option<Waker>; 4] {
        [
            self.read_waker.take(),
            self.write_waker.take(),
            self.flush_waker.take(),
            self.close_waker.take(),
        ]
    }
}

struct StreamShared {
    state: Mutex<StreamState>,
    // This Unix pair carries wake bytes only, never application payload.
    wake: std::os::unix::net::UnixStream,
    disconnected: AtomicBool,
    timeout: Duration,
}

impl StreamShared {
    fn notify(&self) {
        loop {
            match (&self.wake).write(&[1]) {
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                // WouldBlock means the pollable wake queue is already nonempty.
                _ => break,
            }
        }
    }

    fn wake_waiters(&self) {
        let wakers = self.state.lock().unwrap().take_wakers();
        for waker in wakers.into_iter().flatten() {
            waker.wake();
        }
    }

    fn fail(&self, error: io::Error) {
        let mut state = self.state.lock().unwrap();
        if state.error.is_none() {
            state.error = Some((error.kind(), error.to_string()));
        }
        drop(state);
        self.wake_waiters();
    }
}

/// nng poll fd. Closing this wrapper does not close the socket.
pub struct PollFd(i32);

impl AsRawFd for PollFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

enum SentFrame {
    Data(usize),
    Fin,
    Ack { receipt: bool },
}

struct Outgoing {
    message: nng::Message,
    kind: SentFrame,
}

fn encode_frame(kind: u8, flags: u8, offset: u64, consumed: u64, body: &[u8]) -> nng::Message {
    let mut frame = Vec::with_capacity(FRAME_HEADER + body.len());
    frame.extend_from_slice(FRAME_MAGIC);
    frame.extend_from_slice(&[FRAME_VERSION, kind, flags, 0]);
    frame.extend_from_slice(&offset.to_le_bytes());
    frame.extend_from_slice(&consumed.to_le_bytes());
    frame.extend_from_slice(body);
    nng::Message::from(frame.as_slice())
}

fn invalid_frame(reason: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, reason)
}

fn receive_frame(shared: &StreamShared, frame: &[u8]) -> io::Result<()> {
    if frame.len() < FRAME_HEADER
        || frame.len() > FRAME_HEADER + FRAME_PAYLOAD
        || &frame[..4] != FRAME_MAGIC
        || frame[4] != FRAME_VERSION
        || frame[7] != 0
    {
        return Err(invalid_frame("invalid nng stream frame header"));
    }
    let kind = frame[5];
    let flags = frame[6];
    let offset = u64::from_le_bytes(frame[8..16].try_into().unwrap());
    let consumed = u64::from_le_bytes(frame[16..24].try_into().unwrap());
    let body = &frame[FRAME_HEADER..];
    let mut state = shared.state.lock().unwrap();
    match kind {
        DATA => {
            let end = offset
                .checked_add(body.len() as u64)
                .ok_or_else(|| invalid_frame("nng stream offset overflow"))?;
            if flags != 0
                || consumed != 0
                || body.is_empty()
                || state.read_fin
                || offset != state.received
                || end - state.consumed > STREAM_WINDOW as u64
            {
                return Err(invalid_frame("unordered or over-window nng stream data"));
            }
            state.read_queue.extend(body);
            state.received = end;
            state.ack_dirty = true;
        }
        FIN => {
            if flags != 0
                || consumed != 0
                || !body.is_empty()
                || state.read_fin
                || offset != state.received
            {
                return Err(invalid_frame("invalid nng stream final offset"));
            }
            state.read_fin = true;
            state.ack_dirty = true;
        }
        ACK => {
            if !body.is_empty()
                || flags & !(ACK_FIN | ACK_FIN_RECEIPT) != 0
                || offset < state.peer_accepted
                || offset > state.sent
                || consumed < state.peer_consumed
                || consumed > offset
                || (flags & ACK_FIN != 0 && (!state.fin_sent || offset != state.written))
                || (flags & ACK_FIN_RECEIPT != 0 && !state.read_fin)
            {
                return Err(invalid_frame("invalid nng stream acknowledgement"));
            }
            state.peer_accepted = offset;
            state.peer_consumed = consumed;
            if flags & ACK_FIN != 0 && !state.fin_acked {
                state.fin_acked = true;
                state.ack_dirty = true;
                // A live owner retains the read half without a close timer.
                if !state.dropped {
                    state.close_deadline = None;
                }
            }
            if flags & ACK_FIN_RECEIPT != 0 {
                state.fin_receipted = true;
            }
            if state.flush_goal.is_some_and(|(target, _)| offset >= target) {
                state.flush_goal = None;
            }
        }
        _ => return Err(invalid_frame("unknown nng stream frame kind")),
    }
    drop(state);
    shared.wake_waiters();
    Ok(())
}

fn next_frame(shared: &StreamShared, prefer_data: &mut bool) -> Option<Outgoing> {
    let mut state = shared.state.lock().unwrap();
    // ACKs have their own coalesced slot and do not consume byte-window credit.
    // Alternation keeps control and payload progress independent under load.
    let credit = (STREAM_WINDOW as u64).saturating_sub(state.sent - state.peer_consumed);
    let data_available = !state.write_queue.is_empty() && credit > 0;
    let fin_available = state.write_closed && state.sent == state.written && !state.fin_sent;
    if state.ack_dirty && (!*prefer_data || (!data_available && !fin_available)) {
        let flags =
            u8::from(state.read_fin) * ACK_FIN | u8::from(state.fin_acked) * ACK_FIN_RECEIPT;
        let message = encode_frame(ACK, flags, state.received, state.consumed, &[]);
        state.ack_dirty = false;
        *prefer_data = true;
        return Some(Outgoing {
            message,
            kind: SentFrame::Ack {
                receipt: state.fin_acked,
            },
        });
    }
    *prefer_data = false;
    if data_available {
        let count = state
            .write_queue
            .len()
            .min(FRAME_PAYLOAD)
            .min(credit as usize);
        let body: Vec<u8> = state.write_queue.drain(..count).collect();
        Some(Outgoing {
            message: encode_frame(DATA, 0, state.sent, 0, &body),
            kind: SentFrame::Data(count),
        })
    } else if fin_available {
        Some(Outgoing {
            message: encode_frame(FIN, 0, state.written, 0, &[]),
            kind: SentFrame::Fin,
        })
    } else {
        None
    }
}

fn frame_sent(shared: &StreamShared, kind: SentFrame) {
    let mut state = shared.state.lock().unwrap();
    match kind {
        SentFrame::Data(count) => {
            state.sent += count as u64;
            state.write_deadline = None;
        }
        SentFrame::Fin => state.fin_sent = true,
        SentFrame::Ack { receipt } => state.receipt_sent |= receipt,
    }
    drop(state);
    shared.wake_waiters();
}

fn pump_pair(
    session: &PairSession,
    shared: &StreamShared,
    mut notified: std::os::unix::net::UnixStream,
    recv_fd: RawFd,
    send_fd: RawFd,
) -> io::Result<()> {
    let mut pending: Option<Outgoing> = None;
    let mut prefer_data = false;
    let mut wake_bytes = [0; 256];
    loop {
        for _ in 0..PUMP_BATCH {
            match notified.read(&mut wake_bytes) {
                Ok(0) => return Err(io::ErrorKind::BrokenPipe.into()),
                Ok(_) => {}
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => break,
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error),
            }
        }

        let mut received = 0;
        for _ in 0..PUMP_BATCH {
            match session.socket.try_recv() {
                Ok(message) => {
                    receive_frame(shared, &message)?;
                    received += 1;
                }
                Err(nng::Error::TryAgain) => break,
                Err(error) => return Err(nng_io(error)),
            }
        }
        // A readiness descriptor is not a transport-disconnect descriptor.
        // Pipe removal terminates this stream instead of replaying on reconnect.
        if shared.disconnected.load(Ordering::Acquire) {
            if received == PUMP_BATCH {
                continue;
            }
            return Err(io::Error::new(
                io::ErrorKind::ConnectionAborted,
                "nng stream peer disconnected",
            ));
        }

        let mut sent = 0;
        for _ in 0..PUMP_BATCH {
            if shared.disconnected.load(Ordering::Acquire) {
                break;
            }
            let outgoing = match pending
                .take()
                .or_else(|| next_frame(shared, &mut prefer_data))
            {
                Some(outgoing) => outgoing,
                None => break,
            };
            let Outgoing { message, kind } = outgoing;
            match session.socket.try_send(message) {
                Ok(()) => {
                    frame_sent(shared, kind);
                    sent += 1;
                }
                Err((message, nng::Error::TryAgain)) => {
                    pending = Some(Outgoing { message, kind });
                    break;
                }
                Err((_, error)) => return Err(nng_io(error)),
            }
        }

        let state = shared.state.lock().unwrap();
        // FIN acceptance proves all preceding bytes reached the peer pump.
        // A receipt proves our ACK of the peer's FIN reached that pump too.
        if state.dropped
            && state.fin_acked
            && state.receipt_sent
            && (!state.read_fin || state.fin_receipted)
            && pending.is_none()
            && !state.ack_dirty
        {
            return Ok(());
        }
        let deadline = state.deadline();
        drop(state);
        let timeout = if let Some(deadline) = deadline {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "nng stream write, flush or close acknowledgement timed out",
                ));
            }
            remaining
                .as_millis()
                .saturating_add(1)
                .min(i32::MAX as u128) as i32
        } else {
            -1
        };
        if received == PUMP_BATCH || sent == PUMP_BATCH {
            continue;
        }
        let mut fds = [
            libc::pollfd {
                fd: notified.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: recv_fd,
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                // Both nng readiness descriptors signal with readability.
                fd: if pending.is_some() { send_fd } else { -1 },
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        // The descriptors remain owned by the live session and wake pair.
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, timeout) };
        if ready < 0 {
            let error = io::Error::last_os_error();
            if error.kind() != io::ErrorKind::Interrupted {
                return Err(error);
            }
        }
        if fds
            .iter()
            .any(|fd| fd.revents & (libc::POLLERR | libc::POLLNVAL) != 0)
        {
            return Err(io::Error::other("nng stream poll descriptor failed"));
        }
    }
}

impl AsyncRead for NngIo {
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        if buf.remaining() == 0 {
            return Poll::Ready(Ok(()));
        }
        let mut state = self.shared.state.lock().unwrap();
        if !state.read_queue.is_empty() {
            let (first, second) = state.read_queue.as_slices();
            let count = buf.remaining().min(first.len() + second.len());
            let first_count = count.min(first.len());
            buf.put_slice(&first[..first_count]);
            buf.put_slice(&second[..count - first_count]);
            state.read_queue.drain(..count);
            state.consumed += count as u64;
            state.ack_dirty = true;
            drop(state);
            self.shared.notify();
            return Poll::Ready(Ok(()));
        }
        if state.read_fin {
            return Poll::Ready(Ok(()));
        }
        if let Some(error) = state.error() {
            return Poll::Ready(Err(error));
        }
        state.read_waker = Some(cx.waker().clone());
        Poll::Pending
    }
}

impl AsyncWrite for NngIo {
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        let mut state = self.shared.state.lock().unwrap();
        if let Some(error) = state.error() {
            return Poll::Ready(Err(error));
        }
        if state.write_closed {
            return Poll::Ready(Err(io::ErrorKind::BrokenPipe.into()));
        }
        if buf.is_empty() {
            return Poll::Ready(Ok(0));
        }
        let capacity = STREAM_WINDOW - (state.written - state.sent) as usize;
        let count = capacity.min(buf.len());
        if count == 0 {
            state.write_waker = Some(cx.waker().clone());
            state
                .write_deadline
                .get_or_insert_with(|| Instant::now() + self.shared.timeout);
            drop(state);
            self.shared.notify();
            return Poll::Pending;
        }
        let Some(written) = state.written.checked_add(count as u64) else {
            return Poll::Ready(Err(invalid_frame("nng stream write offset overflow")));
        };
        state.write_queue.extend(&buf[..count]);
        state.written = written;
        state.write_deadline = None;
        drop(state);
        self.shared.notify();
        Poll::Ready(Ok(count))
    }

    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut state = self.shared.state.lock().unwrap();
        if state.peer_accepted == state.written {
            state.flush_goal = None;
            return Poll::Ready(Ok(()));
        }
        if let Some(error) = state.error() {
            return Poll::Ready(Err(error));
        }
        let written = state.written;
        let goal = state
            .flush_goal
            .get_or_insert_with(|| (written, Instant::now() + self.shared.timeout));
        goal.0 = written;
        state.flush_waker = Some(cx.waker().clone());
        drop(state);
        self.shared.notify();
        Poll::Pending
    }

    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let mut state = self.shared.state.lock().unwrap();
        if state.fin_acked {
            return Poll::Ready(Ok(()));
        }
        if let Some(error) = state.error() {
            return Poll::Ready(Err(error));
        }
        state.write_closed = true;
        state
            .close_deadline
            .get_or_insert_with(|| Instant::now() + self.shared.timeout);
        state.close_waker = Some(cx.waker().clone());
        drop(state);
        self.shared.notify();
        Poll::Pending
    }

    fn poll_write_vectored(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        self.poll_write(
            cx,
            bufs.iter()
                .find(|buf| !buf.is_empty())
                .map_or(&[], |buf| &buf[..]),
        )
    }
}

impl Drop for NngIo {
    fn drop(&mut self) {
        let mut state = self.shared.state.lock().unwrap();
        state.dropped = true;
        state.write_closed = true;
        state
            .close_deadline
            .get_or_insert_with(|| Instant::now() + self.shared.timeout);
        drop(state);
        self.shared.notify();
    }
}

/// Map an nng failure onto `io::Error`.
pub fn nng_io(error: nng::Error) -> io::Error {
    let kind = match error {
        nng::Error::TimedOut => io::ErrorKind::TimedOut,
        nng::Error::Closed => io::ErrorKind::ConnectionAborted,
        nng::Error::Canceled => io::ErrorKind::Interrupted,
        _ => io::ErrorKind::Other,
    };
    io::Error::new(kind, error)
}

fn pair_bind_spec() -> String {
    // Occupancy and tests are same-node. ipc keeps the vat off a second
    // TCP port per replica. Remote hosts still reach the advertised
    // Rep URL over tcp://; the pair stays on the node.
    let seq = PAIR_SEQ.fetch_add(1, Ordering::Relaxed);
    format!("ipc:///tmp/anneal-pair-{}-{seq}", std::process::id())
}

fn bound_endpoint(
    requested: &str,
    listener: &Listener,
) -> io::Result<(String, Option<SocketAddr>)> {
    if let Ok(port) = listener.get_opt::<BoundPort>()
        && port != 0
    {
        let host = tcp_host(requested);
        let host = if host.is_empty() || host == "*" {
            "127.0.0.1"
        } else {
            host
        };
        let url = if host.contains(':') && !host.starts_with('[') {
            format!("tcp://[{host}]:{port}")
        } else {
            format!("tcp://{host}:{port}")
        };
        let addr = format!("{host}:{port}").parse().ok();
        return Ok((url, addr));
    }
    if let Ok(local) = listener.get_opt::<LocalAddr>() {
        return Ok(format_bound(requested, local));
    }
    Err(io::Error::new(
        io::ErrorKind::AddrNotAvailable,
        format!("nng listener has no bound address for {requested}"),
    ))
}

fn rewrite_pair_url(advertised: &str, bound: &str) -> String {
    let advertised = url(advertised);
    if !advertised.starts_with("tcp://") || !bound.starts_with("tcp://") {
        return bound.to_string();
    }
    let host = tcp_host(&advertised);
    let port = tcp_port(bound);
    let host = if host == "0.0.0.0" || host == "*" || host == "::" || host.is_empty() {
        "127.0.0.1"
    } else {
        host
    };
    format!("tcp://{host}:{port}")
}

fn tcp_host(tcp_url: &str) -> &str {
    let rest = tcp_url.strip_prefix("tcp://").unwrap_or(tcp_url);
    if let Some(stripped) = rest.strip_prefix('[') {
        return stripped.split(']').next().unwrap_or(stripped);
    }
    rest.rsplit_once(':').map(|(host, _)| host).unwrap_or(rest)
}

fn tcp_port(tcp_url: &str) -> &str {
    let rest = tcp_url.strip_prefix("tcp://").unwrap_or(tcp_url);
    if let Some(idx) = rest.rfind(':') {
        &rest[idx + 1..]
    } else {
        "0"
    }
}

fn format_bound(requested: &str, local: nng::SocketAddr) -> (String, Option<SocketAddr>) {
    match local {
        nng::SocketAddr::Inet(v4) => (format!("tcp://{v4}"), Some(SocketAddr::V4(v4))),
        nng::SocketAddr::Inet6(v6) => (format!("tcp://{v6}"), Some(SocketAddr::V6(v6))),
        nng::SocketAddr::Ipc(path) => (format!("ipc://{}", path.display()), None),
        nng::SocketAddr::InProc(name) => (format!("inproc://{name}"), None),
        _ => (requested.to_string(), None),
    }
}

#[cfg(test)]
mod lifecycle_tests {
    use super::*;
    use std::process::{Command, Stdio};
    use std::sync::mpsc;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    const CHILD: &str = "ANNEAL_NNG_DEADLINE_CHILD";
    const LIMIT: Duration = Duration::from_millis(500);
    const TEST_LIMIT: Duration = Duration::from_secs(3);

    fn in_bounded_child(name: &str) -> bool {
        if std::env::var(CHILD).as_deref() == Ok(name) {
            return true;
        }
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args(["--exact", name, "--nocapture"])
            .env(CHILD, name)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let started = Instant::now();
        let terminated = loop {
            if child.try_wait().unwrap().is_some() {
                break true;
            }
            if started.elapsed() >= Duration::from_secs(30) {
                child.kill().unwrap();
                break false;
            }
            std::thread::sleep(Duration::from_millis(10));
        };
        let output = child.wait_with_output().unwrap();
        assert!(
            terminated && output.status.success(),
            "{name} must satisfy its carrier deadline contract; terminated={terminated}:\n{}\n{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        false
    }

    fn runtime() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    fn unacknowledged_operation_times_out(shutdown: bool) {
        let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
        let advertised = accept.url.clone();
        let (received, wait_for_received) = mpsc::channel();
        let (finish, wait_for_finish) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let pair = accept_pair(&accept.socket).unwrap();
            pair.socket
                .set_opt::<RecvTimeout>(Some(TEST_LIMIT))
                .unwrap();
            let frame = pair.socket.recv().unwrap();
            assert_eq!(&frame[FRAME_HEADER..], b"pending");
            received.send(()).unwrap();
            // Transport receipt without a carrier ACK cannot satisfy flush or
            // shutdown. Keep the peer connected throughout the operation.
            wait_for_finish.recv_timeout(TEST_LIMIT).unwrap();
            drop(pair);
        });

        runtime().block_on(async move {
            let pair = dial_pair(&advertised, TEST_LIMIT).unwrap();
            let mut io = NngIo::with_timeout(pair, LIMIT).unwrap();
            io.write_all(b"pending").await.unwrap();
            wait_for_received.recv_timeout(TEST_LIMIT).unwrap();
            let result = tokio::time::timeout(TEST_LIMIT, async {
                if shutdown {
                    io.shutdown().await
                } else {
                    io.flush().await
                }
            })
            .await
            .expect("an outstanding carrier acknowledgement must have a finite deadline");
            assert_eq!(result.unwrap_err().kind(), io::ErrorKind::TimedOut);
            let mut byte = [0];
            let read_error = tokio::time::timeout(TEST_LIMIT, io.read(&mut byte))
                .await
                .expect("a failed carrier must wake its reader")
                .unwrap_err();
            assert_eq!(read_error.kind(), io::ErrorKind::TimedOut);
            finish.send(()).unwrap();
        });
        server.join().unwrap();
    }

    #[test]
    fn flush_without_peer_acknowledgement_has_a_finite_deadline() {
        if in_bounded_child(
            "nng_rpc::lifecycle_tests::flush_without_peer_acknowledgement_has_a_finite_deadline",
        ) {
            unacknowledged_operation_times_out(false);
        }
    }

    #[test]
    fn shutdown_without_peer_acknowledgement_has_a_finite_deadline() {
        if in_bounded_child(
            "nng_rpc::lifecycle_tests::shutdown_without_peer_acknowledgement_has_a_finite_deadline",
        ) {
            unacknowledged_operation_times_out(true);
        }
    }

    #[test]
    fn idle_reads_outlive_the_operation_deadline_and_receive_data() {
        if !in_bounded_child(
            "nng_rpc::lifecycle_tests::idle_reads_outlive_the_operation_deadline_and_receive_data",
        ) {
            return;
        }
        const PAYLOAD: &[u8] = b"result after quiet computation";
        let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
        let advertised = accept.url.clone();
        let (constructed, wait_for_constructed) = mpsc::channel();
        let (write, wait_for_write) = mpsc::channel();
        let server = std::thread::spawn(move || {
            let pair = accept_pair(&accept.socket).unwrap();
            runtime().block_on(async move {
                let mut io = NngIo::with_timeout(pair, LIMIT).unwrap();
                constructed.send(()).unwrap();
                wait_for_write.recv_timeout(TEST_LIMIT).unwrap();
                tokio::time::timeout(TEST_LIMIT, async {
                    io.write_all(PAYLOAD).await.unwrap();
                    io.flush().await.unwrap();
                    io.shutdown().await.unwrap();
                })
                .await
                .expect("the peer must remain usable across its idle interval");
            });
        });

        runtime().block_on(async move {
            let pair = dial_pair(&advertised, TEST_LIMIT).unwrap();
            let mut io = NngIo::with_timeout(pair, LIMIT).unwrap();
            wait_for_constructed.recv_timeout(TEST_LIMIT).unwrap();
            let mut byte = [0];
            let quiet = tokio::time::timeout(LIMIT * 3, io.read(&mut byte)).await;
            assert!(
                quiet.is_err(),
                "an idle read must remain pending, not fail or report EOF"
            );
            write.send(()).unwrap();
            let mut received = Vec::new();
            tokio::time::timeout(TEST_LIMIT, io.read_to_end(&mut received))
                .await
                .expect("the idle carrier must receive the response and FIN")
                .unwrap();
            assert_eq!(received, PAYLOAD);
        });
        server.join().unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::compat::TokioAsyncReadCompatExt;

    #[test]
    fn host_port_becomes_tcp_url() {
        assert_eq!(url("127.0.0.1:9"), "tcp://127.0.0.1:9");
        assert_eq!(url("tcp://127.0.0.1:9"), "tcp://127.0.0.1:9");
        assert_eq!(url("ipc:///tmp/x"), "ipc:///tmp/x");
    }

    #[test]
    fn ephemeral_tcp_listen_reports_a_real_port() {
        let bound = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
        let addr = bound.addr.expect("tcp");
        assert_ne!(addr.port(), 0);
        assert!(bound.url.starts_with("tcp://127.0.0.1:"));
        assert!(!bound.url.ends_with(":0"));
    }

    #[test]
    fn two_vats_exchange_bytes() {
        let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
        let advertised = accept.url.clone();
        let server = std::thread::spawn(move || {
            let pair = accept_pair(&accept.socket).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let mut io = NngIo::new(pair).unwrap();
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 4];
                io.read_exact(&mut buf).await.unwrap();
                assert_eq!(&buf, b"ping");
                io.write_all(b"pong").await.unwrap();
                io.flush().await.unwrap();
            });
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let pair = dial_pair(&advertised, Duration::from_secs(2)).unwrap();
            let mut io = NngIo::new(pair).unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            io.write_all(b"ping").await.unwrap();
            io.flush().await.unwrap();
            let mut buf = [0u8; 4];
            io.read_exact(&mut buf).await.unwrap();
            assert_eq!(&buf, b"pong");
        });
        server.join().unwrap();
    }

    #[test]
    fn compat_split_survives_vat_style_use() {
        let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
        let advertised = accept.url.clone();
        let server = std::thread::spawn(move || {
            let pair = accept_pair(&accept.socket).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            let local = tokio::task::LocalSet::new();
            local.block_on(&runtime, async move {
                let io = NngIo::new(pair).unwrap();
                let (mut reader, mut writer) = TokioAsyncReadCompatExt::compat(io).split();
                use futures::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 3];
                reader.read_exact(&mut buf).await.unwrap();
                writer.write_all(&buf).await.unwrap();
                writer.flush().await.unwrap();
            });
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let pair = dial_pair(&advertised, Duration::from_secs(2)).unwrap();
            let io = NngIo::new(pair).unwrap();
            let (mut reader, mut writer) = TokioAsyncReadCompatExt::compat(io).split();
            use futures::{AsyncReadExt, AsyncWriteExt};
            writer.write_all(b"abc").await.unwrap();
            writer.flush().await.unwrap();
            let mut buf = [0u8; 3];
            reader.read_exact(&mut buf).await.unwrap();
            assert_eq!(&buf, b"abc");
        });
        server.join().unwrap();
    }

    #[test]
    fn pair_roundtrip_prints_frame_rtt() {
        let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
        let advertised = accept.url.clone();
        let server = std::thread::spawn(move || {
            let pair = accept_pair(&accept.socket).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let mut io = NngIo::new(pair).unwrap();
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 8];
                for _ in 0..2000 {
                    io.read_exact(&mut buf).await.unwrap();
                    io.write_all(&buf).await.unwrap();
                }
            });
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let pair = dial_pair(&advertised, Duration::from_secs(2)).unwrap();
            let mut io = NngIo::new(pair).unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 8];
            io.write_all(&[0u8; 8]).await.unwrap();
            io.read_exact(&mut buf).await.unwrap();
            let started = std::time::Instant::now();
            for i in 0..2000u64 {
                buf.copy_from_slice(&i.to_le_bytes());
                io.write_all(&buf).await.unwrap();
                io.read_exact(&mut buf).await.unwrap();
            }
            let elapsed = started.elapsed();
            eprintln!(
                "PAIR_RTT frames=2000 total={:?} per={:?}",
                elapsed,
                elapsed / 2000
            );
        });
        server.join().unwrap();
    }

    #[test]
    fn tcp_loopback_roundtrip_prints_stream_rtt() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let server = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            stream.set_nodelay(true).unwrap();
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .unwrap();
            runtime.block_on(async move {
                let mut io = tokio::net::TcpStream::from_std(stream).unwrap();
                use tokio::io::{AsyncReadExt, AsyncWriteExt};
                let mut buf = [0u8; 8];
                for _ in 0..2000 {
                    io.read_exact(&mut buf).await.unwrap();
                    io.write_all(&buf).await.unwrap();
                }
            });
        });
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let mut io = tokio::net::TcpStream::connect(addr).await.unwrap();
            io.set_nodelay(true).unwrap();
            use tokio::io::{AsyncReadExt, AsyncWriteExt};
            let mut buf = [0u8; 8];
            io.write_all(&[0u8; 8]).await.unwrap();
            io.read_exact(&mut buf).await.unwrap();
            let started = std::time::Instant::now();
            for i in 0..2000u64 {
                buf.copy_from_slice(&i.to_le_bytes());
                io.write_all(&buf).await.unwrap();
                io.read_exact(&mut buf).await.unwrap();
            }
            let elapsed = started.elapsed();
            eprintln!(
                "TCP_RTT frames=2000 total={:?} per={:?}",
                elapsed,
                elapsed / 2000
            );
        });
        server.join().unwrap();
    }
}
