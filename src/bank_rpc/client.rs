//! Blocking Cap'n Proto client over a Unix socket.

use std::io::Write;
use std::net::TcpStream;

use capnp::message::{Builder, HeapAllocator, ReaderOptions};
use capnp::serialize;
use ndarray::{Array1, ArrayView1};

use crate::Bank_capnp::{bank_reply, bank_request};

/// One connection to the shared bank.
pub struct BankClient {
    stream: TcpStream,
}

impl BankClient {
    /// Connect to `host:port`.
    pub fn connect(addr: impl AsRef<str>) -> std::io::Result<Self> {
        let stream = TcpStream::connect(addr.as_ref())?;
        stream.set_nodelay(true)?;
        Ok(Self { stream })
    }

    fn call(
        &mut self,
        build: impl FnOnce(bank_request::Builder<'_>),
    ) -> Result<Reply, String> {
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
            fill_list(o.reborrow().init_coords(coords.len() as u32), coords.as_slice().unwrap_or(&[]));
            fill_list(o.init_soap(soap.len() as u32), soap.as_slice().unwrap_or(&[]));
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
        let reply = self.call(|mut req| {
            req.set_snapshot(());
        })?;
        Ok(reply.wells)
    }

    /// Publish a Dcut. Ignored while the first bank is still seeding.
    pub fn set_dcut(&mut self, dcut: f64) -> Result<(), String> {
        self.call(|mut req| {
            req.set_set_dcut(dcut);
        })?;
        Ok(())
    }
}

/// Decoded reply.
struct Reply {
    kind: u16,
    energy: f64,
    dcut: f64,
    distance: f64,
    height: f64,
    coords: Array1<f64>,
    empty: bool,
    wells: Vec<(Array1<f64>, f64)>,
}

impl Reply {
    fn from_reader(r: bank_reply::Reader<'_>) -> Result<Self, String> {
        let coords = list_f64(r.get_coords().map_err(|e| e.to_string())?);
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
            coords,
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
