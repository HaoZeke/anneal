#![cfg(feature = "bank-rpc")]

use anneal_core::nng_rpc::{NngIo, accept_pair, dial_pair, listen};
use nng::Protocol;
use std::future::poll_fn;
use std::pin::Pin;
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Barrier, mpsc};
use std::task::Poll;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWrite, AsyncWriteExt};

const CHILD: &str = "ANNEAL_NNG_HALF_CLOSE_CHILD";
const IO_DEADLINE: Duration = Duration::from_secs(3);
const BULK_DEADLINE: Duration = Duration::from_secs(10);

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
        "{name} must satisfy stream lifecycle semantics within its deadline; terminated={terminated}:\n{}\n{}",
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

fn payload(size: usize, salt: u8) -> Vec<u8> {
    (0..size)
        .map(|index| {
            let mixed = (index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            ((mixed >> 56) as u8).wrapping_add(salt)
        })
        .collect()
}

fn assert_exact_payload(received: &[u8], expected: &[u8]) {
    assert_eq!(received.len(), expected.len(), "stream length differs");
    let mismatch = received
        .iter()
        .zip(expected)
        .position(|(actual, wanted)| actual != wanted);
    assert!(mismatch.is_none(), "stream differs at byte {mismatch:?}");
}

#[test]
fn shutdown_closes_only_the_write_half() {
    if !in_bounded_child("shutdown_closes_only_the_write_half") {
        return;
    }

    const REQUEST: &[u8] = b"request ending at write EOF";
    const RESPONSE: &[u8] = b"response remains readable after write shutdown";
    let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
    let advertised = accept.url.clone();
    let server = std::thread::spawn(move || {
        let pair = accept_pair(&accept.socket).unwrap();
        runtime().block_on(async move {
            let mut io = NngIo::new(pair).unwrap();
            tokio::time::timeout(IO_DEADLINE, async {
                let mut request = Vec::new();
                io.read_to_end(&mut request).await.unwrap();
                assert_exact_payload(&request, REQUEST);
                io.write_all(RESPONSE).await.unwrap();
                io.flush().await.unwrap();
                io.shutdown().await.unwrap();
            })
            .await
            .expect("request EOF must leave the response direction writable");
        });
    });

    runtime().block_on(async move {
        let pair = dial_pair(&advertised, IO_DEADLINE).unwrap();
        let mut io = NngIo::new(pair).unwrap();
        tokio::time::timeout(IO_DEADLINE, async {
            io.write_all(REQUEST).await.unwrap();
            io.flush().await.unwrap();
            io.shutdown().await.unwrap();
            let mut response = Vec::new();
            io.read_to_end(&mut response).await.unwrap();
            assert_exact_payload(&response, RESPONSE);
        })
        .await
        .expect("write shutdown must preserve reading the complete response and its EOF");
    });
    server.join().unwrap();
}

async fn exchange_duplex(io: NngIo, outgoing: &[u8], expected: &[u8]) {
    tokio::time::timeout(BULK_DEADLINE, async {
        let (mut reader, mut writer) = tokio::io::split(io);
        let send = async {
            writer.write_all(outgoing).await?;
            writer.flush().await?;
            writer.shutdown().await
        };
        let receive = async {
            let mut received = Vec::new();
            reader.read_to_end(&mut received).await?;
            Ok::<_, std::io::Error>(received)
        };
        let (_, received) = tokio::try_join!(send, receive)?;
        assert_exact_payload(&received, expected);
        Ok::<_, std::io::Error>(())
    })
    .await
    .expect("concurrent duplex traffic and both EOFs must make bounded progress")
    .unwrap();
}

#[test]
fn simultaneous_large_writes_preserve_both_streams() {
    if !in_bounded_child("simultaneous_large_writes_preserve_both_streams") {
        return;
    }

    const CLIENT_SIZE: usize = 3 * 1024 * 1024 + 17;
    const SERVER_SIZE: usize = 4 * 1024 * 1024 + 29;
    let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
    let advertised = accept.url.clone();
    let start = Arc::new(Barrier::new(2));
    let server_start = Arc::clone(&start);
    let server = std::thread::spawn(move || {
        let pair = accept_pair(&accept.socket).unwrap();
        let outgoing = payload(SERVER_SIZE, 173);
        let expected = payload(CLIENT_SIZE, 41);
        runtime().block_on(async move {
            let io = NngIo::new(pair).unwrap();
            server_start.wait();
            exchange_duplex(io, &outgoing, &expected).await;
        });
    });

    runtime().block_on(async move {
        let pair = dial_pair(&advertised, IO_DEADLINE).unwrap();
        let io = NngIo::new(pair).unwrap();
        let outgoing = payload(CLIENT_SIZE, 41);
        let expected = payload(SERVER_SIZE, 173);
        start.wait();
        exchange_duplex(io, &outgoing, &expected).await;
    });
    server.join().unwrap();
}

#[test]
fn flush_waits_for_peer_pump_but_not_application_read() {
    if !in_bounded_child("flush_waits_for_peer_pump_but_not_application_read") {
        return;
    }

    const PAYLOAD: &[u8] = b"peer pump acceptance is the flush barrier";
    let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
    let advertised = accept.url.clone();
    let (permit_pump, wait_for_pump) = mpsc::channel();
    let (permit_read, wait_for_read) = mpsc::channel();
    let read_started = Arc::new(AtomicBool::new(false));
    let server_read_started = Arc::clone(&read_started);
    let server = std::thread::spawn(move || {
        let pair = accept_pair(&accept.socket).unwrap();
        wait_for_pump
            .recv_timeout(IO_DEADLINE)
            .expect("the sender must poll flush before the receiver constructs its pump");
        runtime().block_on(async move {
            let mut io = NngIo::new(pair).unwrap();
            wait_for_read
                .recv_timeout(IO_DEADLINE)
                .expect("flush must finish without application consumption");
            server_read_started.store(true, Ordering::Release);
            let mut received = [0u8; PAYLOAD.len()];
            tokio::time::timeout(IO_DEADLINE, io.read_exact(&mut received))
                .await
                .expect("the accepted payload must be readable")
                .unwrap();
            assert_exact_payload(&received, PAYLOAD);
        });
    });

    runtime().block_on(async move {
        let pair = dial_pair(&advertised, IO_DEADLINE).unwrap();
        let mut io = NngIo::new(pair).unwrap();
        tokio::time::timeout(IO_DEADLINE, io.write_all(PAYLOAD))
            .await
            .expect("a small write must fit without peer application consumption")
            .unwrap();
        poll_fn(|cx| {
            assert!(
                Pin::new(&mut io).poll_flush(cx).is_pending(),
                "flush cannot complete before the peer pump exists"
            );
            Poll::Ready(())
        })
        .await;
        permit_pump.send(()).unwrap();
        tokio::time::timeout(IO_DEADLINE, io.flush())
            .await
            .expect("peer pump acceptance must complete flush without an application read")
            .unwrap();
        assert!(
            !read_started.load(Ordering::Acquire),
            "the receiver application must not consume bytes before flush completes"
        );
        permit_read.send(()).unwrap();
    });
    server.join().unwrap();
}
