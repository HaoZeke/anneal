#![cfg(feature = "bank-rpc")]

use anneal_core::nng_rpc::{NngIo, accept_pair, dial_pair, listen};
use nng::Protocol;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

const CHILD: &str = "ANNEAL_NNG_TERMINAL_DELIVERY_CHILD";
const IO_DEADLINE: Duration = Duration::from_secs(3);

fn run_bounded_child(name: &str) {
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
        "{name} must preserve stream delivery within its deadline; terminated={terminated}:\n{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
}

fn payload(round: usize, size: usize) -> Vec<u8> {
    (0..size)
        .map(|index| index.wrapping_mul(17).wrapping_add(round) as u8)
        .collect()
}

#[test]
fn terminal_write_then_drop_preserves_every_byte() {
    const NAME: &str = "terminal_write_then_drop_preserves_every_byte";
    if std::env::var(CHILD).as_deref() != Ok(NAME) {
        run_bounded_child(NAME);
        return;
    }

    // Each response ends its stream. The larger responses span multiple
    // carrier frames and exercise buffered bytes at the local close boundary.
    const SIZES: [usize; 3] = [3, 65_537, 1_048_579];
    const ROUNDS: usize = 24;
    let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
    let advertised = accept.url.clone();
    let server = std::thread::spawn(move || {
        let runtime = runtime();
        for round in 0..ROUNDS {
            let pair = accept_pair(&accept.socket).unwrap();
            runtime.block_on(async move {
                let mut io = NngIo::new(pair).unwrap();
                let mut request = [0u8; 8];
                io.read_exact(&mut request).await.unwrap();
                assert_eq!(u64::from_le_bytes(request), round as u64);
                let response = payload(round, SIZES[round % SIZES.len()]);
                io.write_all(&response).await.unwrap();
                io.flush().await.unwrap();
                drop(io);
            });
        }
    });

    runtime().block_on(async move {
        for round in 0..ROUNDS {
            let pair = dial_pair(&advertised, IO_DEADLINE).unwrap();
            let mut io = NngIo::new(pair).unwrap();
            io.write_all(&(round as u64).to_le_bytes()).await.unwrap();
            io.flush().await.unwrap();
            let expected = payload(round, SIZES[round % SIZES.len()]);
            let mut received = vec![0u8; expected.len()];
            let result = tokio::time::timeout(IO_DEADLINE, io.read_exact(&mut received)).await;
            assert!(
                result.is_ok(),
                "round {round}: terminal response of {} bytes must arrive before the deadline",
                expected.len(),
            );
            result.unwrap().unwrap_or_else(|error| {
                panic!("round {round}: terminal response was truncated: {error}")
            });
            assert_eq!(
                received, expected,
                "round {round}: terminal payload differs"
            );
        }
    });
    server.join().unwrap();
}

#[test]
fn a_peer_drop_after_acknowledged_delivery_produces_stream_eof() {
    const NAME: &str = "a_peer_drop_after_acknowledged_delivery_produces_stream_eof";
    if std::env::var(CHILD).as_deref() != Ok(NAME) {
        run_bounded_child(NAME);
        return;
    }

    let accept = listen(Protocol::Rep0, "127.0.0.1:0").unwrap();
    let advertised = accept.url.clone();
    let (dropped, wait_for_drop) = mpsc::channel();
    let server = std::thread::spawn(move || {
        let pair = accept_pair(&accept.socket).unwrap();
        runtime().block_on(async move {
            let mut io = NngIo::new(pair).unwrap();
            io.write_all(b"terminal").await.unwrap();
            io.flush().await.unwrap();
            // An application acknowledgement proves receipt independently of
            // flush completion or acceptance by the local nng send operation.
            let mut acknowledgement = [0u8; 3];
            io.read_exact(&mut acknowledgement).await.unwrap();
            assert_eq!(&acknowledgement, b"ack");
            drop(io);
            dropped.send(()).unwrap();
        });
    });

    runtime().block_on(async move {
        let pair = dial_pair(&advertised, IO_DEADLINE).unwrap();
        let mut io = NngIo::new(pair).unwrap();
        let mut response = [0u8; 8];
        tokio::time::timeout(IO_DEADLINE, io.read_exact(&mut response))
            .await
            .expect("the response must arrive while the peer remains open")
            .unwrap();
        assert_eq!(&response, b"terminal");
        io.write_all(b"ack").await.unwrap();
        io.flush().await.unwrap();
        wait_for_drop
            .recv_timeout(IO_DEADLINE)
            .expect("the peer must observe the acknowledgement and drop its stream");

        let mut extra = [0u8; 1];
        let result = tokio::time::timeout(IO_DEADLINE, io.read(&mut extra)).await;
        assert!(
            result.is_ok(),
            "a dropped peer with no outstanding bytes must not leave a stream read pending"
        );
        assert_eq!(result.unwrap().unwrap(), 0, "peer closure is stream EOF");
    });
    server.join().unwrap();
}
