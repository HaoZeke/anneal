//! Catalog I/O on a dedicated thread.
//!
//! The hop loop never owns the socket. Talking is a mailbox: post a
//! request, keep hopping, apply the last answer when it arrives. That
//! is how a cooperative replica stays at least as strong as the same
//! single chain. A blocking `recv` on the hop thread can only be worse.

use std::sync::mpsc::{self, Sender};
use std::thread::{self, JoinHandle};

use super::client::CatalogClient;

enum CatalogJob {
    Run(Box<dyn FnOnce(&mut CatalogClient) + Send>),
}

/// Owns one `CatalogClient` on an I/O thread.
pub struct CatalogMailbox {
    jobs: Option<Sender<CatalogJob>>,
    thread: Option<JoinHandle<()>>,
}

impl CatalogMailbox {
    /// Move the client onto its I/O thread.
    pub fn spawn(mut client: CatalogClient) -> Self {
        let (jobs, rx) = mpsc::channel();
        let thread = thread::Builder::new()
            .name("catalog-io".to_owned())
            .spawn(move || {
                while let Ok(CatalogJob::Run(job)) = rx.recv() {
                    job(&mut client);
                }
            })
            .expect("catalog I/O thread starts");
        Self {
            jobs: Some(jobs),
            thread: Some(thread),
        }
    }

    /// Run one client call and wait for it. Tests and rare control paths.
    pub fn exec<T, F>(&self, work: F) -> T
    where
        T: Send + 'static,
        F: FnOnce(&mut CatalogClient) -> T + Send + 'static,
    {
        let (tx, rx) = mpsc::sync_channel(1);
        self.post(move |client| {
            let _ = tx.send(work(client));
        });
        rx.recv().expect("catalog I/O thread answers exec")
    }

    /// Queue work. The hop thread does not wait.
    pub fn post<F>(&self, work: F)
    where
        F: FnOnce(&mut CatalogClient) + Send + 'static,
    {
        let Some(jobs) = self.jobs.as_ref() else {
            return;
        };
        let _ = jobs.send(CatalogJob::Run(Box::new(work)));
    }
}

impl Drop for CatalogMailbox {
    fn drop(&mut self) {
        self.jobs.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}
