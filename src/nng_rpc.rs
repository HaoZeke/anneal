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

use std::io::{self, IoSlice};
use std::net::SocketAddr;
use std::os::unix::io::{AsRawFd, RawFd};
use std::pin::Pin;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::Duration;

use nng::options::protocol::reqrep::ResendTime;
use nng::options::transport::tcp::{BoundPort, NoDelay};
use nng::options::{LocalAddr, Options, RecvFd, RecvMaxSize, RecvTimeout, SendTimeout};
use nng::{Listener, ListenerBuilder, Protocol, Socket};
use tokio::io::{AsyncRead, AsyncWrite, ReadBuf};
use tokio::io::unix::AsyncFd;

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
    let _ = socket.set_opt::<RecvMaxSize>(0);
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
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "empty pair url",
        ));
    }
    let pair_url = rewrite_pair_url(advertised, bound);
    let pair = Socket::new(Protocol::Pair0).map_err(nng_io)?;
    let _ = pair.set_opt::<RecvMaxSize>(0);
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

/// Tokio byte stream. A pump thread copies nng messages onto a Unix
/// pair so Cap'n vat code keeps stream semantics and nng keeps frames.
pub struct NngIo {
    stream: tokio::net::UnixStream,
}

impl NngIo {
    /// Wrap a pair (or any bidirectional) socket.
    pub fn new(session: PairSession) -> io::Result<Self> {
        let (local, remote) = std::os::unix::net::UnixStream::pair()?;
        local.set_nonblocking(true)?;
        remote.set_nonblocking(false)?;
        let stream = tokio::net::UnixStream::from_std(local)?;
        std::thread::Builder::new()
            .name("anneal-nng-pump".into())
            .spawn(move || pump_pair(session, remote))
            .map_err(|error| io::Error::new(io::ErrorKind::Other, error))?;
        Ok(Self { stream })
    }
}

/// nng poll fd. Closing this wrapper does not close the socket.
pub struct PollFd(i32);

impl AsRawFd for PollFd {
    fn as_raw_fd(&self) -> RawFd {
        self.0
    }
}

fn pump_pair(session: PairSession, unix: std::os::unix::net::UnixStream) {
    let nng_fd = match session.socket.get_opt::<RecvFd>() {
        Ok(fd) => fd,
        Err(_) => return,
    };
    let unix_fd = unix.as_raw_fd();
    let mut buf = [0u8; 65536];
    loop {
        let mut fds = [
            libc::pollfd {
                fd: unix_fd,
                events: libc::POLLIN,
                revents: 0,
            },
            libc::pollfd {
                fd: nng_fd,
                events: libc::POLLIN,
                revents: 0,
            },
        ];
        let ready = unsafe { libc::poll(fds.as_mut_ptr(), 2, 100) };
        if ready < 0 {
            break;
        }
        if fds[0].revents & (libc::POLLHUP | libc::POLLERR) != 0 {
            break;
        }
        if fds[0].revents & libc::POLLIN != 0 {
            match std::io::Read::read(&mut &unix, &mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if session.socket.send(&buf[..n]).is_err() {
                        break;
                    }
                }
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {}
                Err(_) => break,
            }
        }
        if fds[1].revents & libc::POLLIN != 0 {
            match session.socket.try_recv() {
                Ok(message) => {
                    if std::io::Write::write_all(&mut &unix, &message).is_err() {
                        break;
                    }
                }
                Err(nng::Error::TryAgain) => {}
                Err(_) => break,
            }
        }
        if fds[1].revents & (libc::POLLHUP | libc::POLLERR) != 0 {
            break;
        }
    }
}

impl AsyncRead for NngIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_read(cx, buf)
    }
}

impl AsyncWrite for NngIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write(cx, buf)
    }

    fn poll_flush(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_flush(cx)
    }

    fn poll_shutdown(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Pin::new(&mut self.stream).poll_shutdown(cx)
    }

    fn poll_write_vectored(
        mut self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        bufs: &[IoSlice<'_>],
    ) -> Poll<io::Result<usize>> {
        Pin::new(&mut self.stream).poll_write_vectored(cx, bufs)
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

fn bound_endpoint(requested: &str, listener: &Listener) -> io::Result<(String, Option<SocketAddr>)> {
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
