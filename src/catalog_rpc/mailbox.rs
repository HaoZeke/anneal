//! Catalog I/O on a dedicated thread.
//!
//! The hop loop never owns the RPC executor. Talking is a mailbox: post a
//! request, keep hopping, apply the last answer when it arrives. That
//! is how a cooperative replica stays at least as strong as the same
//! single chain. A blocking `recv` on the hop thread can only be worse.
//! `CatalogClient` itself runs `RpcSystem` on a LocalSet; this queue
//! keeps that work off the hop thread.

use std::sync::mpsc::{self, Sender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use super::client::CatalogClient;
use crate::surface_evidence::{SurfaceEvidenceBook, SurfaceEvidenceMessage};

enum CatalogJob {
    Run(Box<dyn FnOnce(&mut CatalogClient) + Send>),
}

/// Owns one `CatalogClient` on an I/O thread.
pub struct CatalogMailbox {
    jobs: Option<Sender<CatalogJob>>,
    thread: Option<JoinHandle<()>>,
    /// Peer surface rewards, applied on the catalog thread under their original key.
    surface_evidence: Arc<Mutex<SurfaceEvidenceBook>>,
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
            surface_evidence: Arc::new(Mutex::new(SurfaceEvidenceBook::new(0))),
        }
    }

    /// Queue one surface-evidence reply. The hop thread does not wait.
    ///
    /// The message keeps the source key from the block that produced it.
    pub fn post_surface_evidence(&self, message: SurfaceEvidenceMessage) {
        let book = Arc::clone(&self.surface_evidence);
        let arms = message.arms.len();
        self.post(move |_client| {
            let mut evidence = book.lock().expect("surface evidence book");
            if evidence.arms() == 0 {
                *evidence = SurfaceEvidenceBook::new(arms);
            }
            let _ = evidence.exchange(message);
        });
    }

    /// Book of surface-evidence messages applied by this mailbox.
    pub fn surface_evidence(&self) -> Arc<Mutex<SurfaceEvidenceBook>> {
        Arc::clone(&self.surface_evidence)
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

    /// Wait for every job posted before this call to finish.
    ///
    /// The queue is FIFO on one thread, so an empty job that answers is
    /// a barrier. A caller that posted work and then wants to read what
    /// the coordinator made of it has no other way to know it landed.
    pub fn drain(&self) {
        self.exec(|_| ());
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
