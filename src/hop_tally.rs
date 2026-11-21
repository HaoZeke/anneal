//! Move tallies for one search.
//!
//! Counts are plain integers. The generator wrapper forwards every
//! draw and then adds one. Nothing here writes a coordinate or a
//! random value.

use rand::RngCore;
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufWriter, Write};

struct Row {
    name: String,
    proposed: u64,
    accepted: u64,
    charged: u64,
}

struct Step {
    index: u32,
    name: u16,
    iter_energy_bits: u64,
    proposal_bits: u64,
    draws_iter: u64,
    draws_begin: u64,
    draws_end: u64,
    accepted: u8,
    charged: u64,
    spent: u64,
}

struct State {
    print: bool,
    trace_path: Option<String>,
    draws: u64,
    gap: u64,
    last_spent: Option<u64>,
    open_spent: u64,
    open_draws: u64,
    iter_draws: u64,
    iter_energy_bits: u64,
    rows: Vec<Row>,
    names: Vec<String>,
    steps: Vec<Step>,
    next_step: u32,
}

impl State {
    fn fresh() -> Self {
        let print = std::env::var("ANNEAL_MOVE_COUNT").ok().as_deref() == Some("1");
        let trace_path = std::env::var("ANNEAL_STEP_TRACE")
            .ok()
            .filter(|p| !p.is_empty());
        Self {
            print,
            trace_path,
            draws: 0,
            gap: 0,
            last_spent: None,
            open_spent: 0,
            open_draws: 0,
            iter_draws: 0,
            iter_energy_bits: 0,
            rows: Vec::new(),
            names: Vec::new(),
            steps: Vec::new(),
            next_step: 0,
        }
    }

    fn row(&mut self, name: &str) -> &mut Row {
        if let Some(index) = self.rows.iter().position(|row| row.name == name) {
            return &mut self.rows[index];
        }
        self.rows.push(Row {
            name: name.to_owned(),
            proposed: 0,
            accepted: 0,
            charged: 0,
        });
        self.rows.last_mut().unwrap()
    }

    fn name_id(&mut self, name: &str) -> u16 {
        if let Some(index) = self.names.iter().position(|held| held == name) {
            return index as u16;
        }
        self.names.push(name.to_owned());
        (self.names.len() - 1) as u16
    }
}

thread_local! {
    static STATE: RefCell<State> = RefCell::new(State::fresh());
}

fn bump() {
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.draws = state.draws.saturating_add(1);
    });
}

pub(crate) fn begin_run() {
    STATE.with(|cell| {
        *cell.borrow_mut() = State::fresh();
    });
}

pub(crate) fn at_iteration(spent: usize, energy: f64) {
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        state.iter_draws = state.draws;
        state.iter_energy_bits = energy.to_bits();
        let _ = spent;
    });
}

pub(crate) fn begin_move(name: &str, spent: usize) {
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let spent = spent as u64;
        let gap = match state.last_spent {
            Some(last) => spent.saturating_sub(last),
            None => spent,
        };
        state.gap = state.gap.saturating_add(gap);
        state.open_spent = spent;
        state.open_draws = state.draws;
        let row = state.row(name);
        row.proposed = row.proposed.saturating_add(1);
    });
}

pub(crate) fn finish_move(name: &str, accepted: bool, spent: usize, proposal: f64) {
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let spent = spent as u64;
        let charged = spent.saturating_sub(state.open_spent);
        {
            let row = state.row(name);
            row.charged = row.charged.saturating_add(charged);
            if accepted {
                row.accepted = row.accepted.saturating_add(1);
            }
        }
        if state.trace_path.is_some() {
            let name_id = state.name_id(name);
            let step = Step {
                index: state.next_step,
                name: name_id,
                iter_energy_bits: state.iter_energy_bits,
                proposal_bits: proposal.to_bits(),
                draws_iter: state.iter_draws,
                draws_begin: state.open_draws,
                draws_end: state.draws,
                accepted: u8::from(accepted),
                charged,
                spent,
            };
            state.next_step = state.next_step.saturating_add(1);
            state.steps.push(step);
        }
        state.last_spent = Some(spent);
    });
}

pub(crate) fn emit(total: usize, best: f64) {
    STATE.with(|cell| {
        let mut state = cell.borrow_mut();
        let total = total as u64;
        let tail = total.saturating_sub(state.last_spent.unwrap_or(0));
        state.gap = state.gap.saturating_add(tail);
        if state.print {
            for row in &state.rows {
                println!(
                    "tally,move,{},{},{},{}",
                    row.name, row.proposed, row.accepted, row.charged
                );
            }
            println!("tally,gap,between,0,0,{}", state.gap);
            println!("tally,total,charged,{total},0,0");
            println!("tally,draws,{},0,0,0", state.draws);
            println!("tally,best,{:016x},0,0,0", best.to_bits());
        }
        if let Some(path) = state.trace_path.clone() {
            let _ = write_trace(&path, &state);
        }
    });
}

fn write_trace(path: &str, state: &State) -> std::io::Result<()> {
    let file = File::create(path)?;
    let mut out = BufWriter::new(file);
    out.write_all(
        b"step,move,iter_energy_bits,proposal_bits,draws_iter,draws_begin,draws_end,accepted,charged,spent\n",
    )?;
    for step in &state.steps {
        let name = state
            .names
            .get(step.name as usize)
            .map(String::as_str)
            .unwrap_or("");
        writeln!(
            out,
            "{},{},{:016x},{:016x},{},{},{},{},{},{}",
            step.index,
            name,
            step.iter_energy_bits,
            step.proposal_bits,
            step.draws_iter,
            step.draws_begin,
            step.draws_end,
            step.accepted,
            step.charged,
            step.spent
        )?;
    }
    out.flush()
}

pub(crate) struct DrawRng<'a, R: ?Sized> {
    pub(crate) inner: &'a mut R,
}

impl<R: RngCore + ?Sized> RngCore for DrawRng<'_, R> {
    fn next_u32(&mut self) -> u32 {
        let value = self.inner.next_u32();
        bump();
        value
    }

    fn next_u64(&mut self) -> u64 {
        let value = self.inner.next_u64();
        bump();
        value
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        self.inner.fill_bytes(dest);
        bump();
    }
}
