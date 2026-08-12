//! Optional informer, not the walk.
//!
//! The chain is a kubelet: it owns its leftover search. The bank is
//! the control plane. If the plane is down, slow, or on the login
//! node under IRA load, the walk continues. A refused connect is
//! not a panic; a hung snapshot is a dropped client, not a blocked hop.

use std::io::Write;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use capnp::message::{Builder, HeapAllocator, ReaderOptions};
use capnp::serialize;
use ndarray::{Array1, ArrayView1};

use crate::Bank_capnp::{bank_reply, bank_request};

const CONNECT: Duration = Duration::from_secs(2);
const IO: Duration = Duration::from_secs(5);

/// One connection to the shared bank.
pub struct BankClient {
    stream: TcpStream,
}

impl BankClient {
    /// Connect to `host:port` with a short timeout.
    pub fn connect(addr: impl AsRef<str>) -> std::io::Result<Self> {
        let addr = addr.as_ref();
        let sock = addr.to_socket_addrs()?.next().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("no address for {addr}"),
            )
        })?;
        let stream = TcpStream::connect_timeout(&sock, CONNECT)?;
        stream.set_nodelay(true)?;
        stream.set_read_timeout(Some(IO))?;
        stream.set_write_timeout(Some(IO))?;
        Ok(Self { stream })
    }

    fn call(&mut self, build: impl FnOnce(bank_request::Builder<'_>)) -> Result<Reply, String> {
        let mut message = Builder::new_default();
        build(message.init_root::<bank_request::Builder>());
        write_msg(&mut self.stream, &message)?;
        let reader = serialize::read_message(&mut self.stream, ReaderOptions::new())
            .map_err(|e| format!("bank reply: {e}"))?;
        let r = reader
            .get_root::<bank_reply::Reader>()
            .map_err(|e| format!("bank root: {e}"))?;
        Reply::from_reader(r)
    }

    /// Offer a quenched member. Returns the admission kind and current Dcut.
    pub fn offer(
        &mut self,
        energy: f64,
        coords: ArrayView1<f64>,
        soap: ArrayView1<f64>,
    ) -> Result<(u16, f64), String> {
        let reply = self.call(|mut req| {
            let mut o = req.reborrow().init_offer();
            o.set_energy(energy);
            fill_list(
                o.reborrow().init_coords(coords.len() as u32),
                coords.as_slice().unwrap_or(&[]),
            );
            fill_list(
                o.init_soap(soap.len() as u32),
                soap.as_slice().unwrap_or(&[]),
            );
        })?;
        Ok((reply.kind, reply.dcut))
    }

    /// Distance in packing SOAP to the nearest known well.
    pub fn nearest(&mut self, soap: ArrayView1<f64>) -> Result<f64, String> {
        let reply = self.call(|mut req| {
            fill_list(
                req.reborrow().init_nearest(soap.len() as u32),
                soap.as_slice().unwrap_or(&[]),
            );
        })?;
        Ok(reply.distance)
    }

    /// Add `increment` to the well containing `soap`.
    pub fn deposit(&mut self, soap: ArrayView1<f64>, increment: f64) -> Result<f64, String> {
        let reply = self.call(|mut req| {
            let mut d = req.reborrow().init_deposit();
            fill_list(
                d.reborrow().init_soap(soap.len() as u32),
                soap.as_slice().unwrap_or(&[]),
            );
            d.set_increment(increment);
        })?;
        Ok(reply.height)
    }

    /// Current height of the well containing `soap`.
    pub fn bias_of(&mut self, soap: ArrayView1<f64>) -> Result<f64, String> {
        let reply = self.call(|mut req| {
            fill_list(
                req.reborrow().init_bias_of(soap.len() as u32),
                soap.as_slice().unwrap_or(&[]),
            );
        })?;
        Ok(reply.height)
    }

    /// A bank member to start from. `None` if the bank is empty.
    pub fn sample(&mut self, seed: u64) -> Result<Option<(f64, Array1<f64>)>, String> {
        let reply = self.call(|mut req| {
            req.set_sample(seed);
        })?;
        if reply.empty {
            return Ok(None);
        }
        Ok(Some((reply.energy, reply.coords)))
    }

    /// Packing wells held on the server.
    pub fn wells(&mut self) -> Result<Vec<(Array1<f64>, f64)>, String> {
        Ok(self.snapshot()?.wells)
    }

    /// Bank size, Dcut, member energies, and SOAP wells.
    pub fn snapshot(&mut self) -> Result<Snapshot, String> {
        let reply = self.call(|mut req| {
            req.set_snapshot(());
        })?;
        Ok(Snapshot {
            size: reply.size,
            dcut: reply.dcut,
            energies: reply.energies,
            wells: reply.wells,
        })
    }

    /// Publish a Dcut. Ignored while the first bank is still seeding.
    pub fn set_dcut(&mut self, dcut: f64) -> Result<(), String> {
        self.call(|mut req| {
            req.set_set_dcut(dcut);
        })?;
        Ok(())
    }
}

/// Live bank contents from `snapshot`.
pub struct Snapshot {
    /// Members currently held.
    pub size: u32,
    /// Current Lee Dcut.
    pub dcut: f64,
    /// Member energies, bank order.
    pub energies: Vec<f64>,
    /// SOAP packing wells `(spectrum, height)`.
    pub wells: Vec<(Array1<f64>, f64)>,
}

/// Decoded reply.
struct Reply {
    kind: u16,
    energy: f64,
    dcut: f64,
    distance: f64,
    height: f64,
    size: u32,
    coords: Array1<f64>,
    energies: Vec<f64>,
    empty: bool,
    wells: Vec<(Array1<f64>, f64)>,
}

impl Reply {
    fn from_reader(r: bank_reply::Reader<'_>) -> Result<Self, String> {
        let coords = list_f64(r.get_coords().map_err(|e| e.to_string())?);
        let energies = if let Ok(es) = r.get_energies() {
            es.iter().collect()
        } else {
            Vec::new()
        };
        let mut wells = Vec::new();
        if let Ok(ws) = r.get_wells() {
            for w in ws.iter() {
                let soap = list_f64(w.get_soap().map_err(|e| e.to_string())?);
                wells.push((soap, w.get_height()));
            }
        }
        Ok(Self {
            kind: r.get_kind(),
            energy: r.get_energy(),
            dcut: r.get_dcut(),
            distance: r.get_distance(),
            height: r.get_height(),
            size: r.get_size(),
            coords,
            energies,
            empty: r.get_empty(),
            wells,
        })
    }
}

fn fill_list(mut b: capnp::primitive_list::Builder<'_, f64>, xs: &[f64]) {
    for (i, &v) in xs.iter().enumerate() {
        b.set(i as u32, v);
    }
}

fn list_f64(r: capnp::primitive_list::Reader<'_, f64>) -> Array1<f64> {
    Array1::from_iter(r.iter())
}

fn write_msg(stream: &mut TcpStream, message: &Builder<HeapAllocator>) -> Result<(), String> {
    serialize::write_message(&mut *stream, message).map_err(|e| format!("bank write: {e}"))?;
    stream.flush().map_err(|e| format!("bank flush: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::BankClient;
    use std::time::Instant;

    #[test]
    fn a_closed_port_fails_fast() {
        let t = Instant::now();
        let r = BankClient::connect("127.0.0.1:1");
        assert!(r.is_err(), "closed port must not hang");
        assert!(
            t.elapsed() < std::time::Duration::from_secs(3),
            "connect took {:?}",
            t.elapsed()
        );
    }
}
