//! Leader consensus over exploration decrees.
//!
//! The per-chain server brains need one of themselves to decide where
//! the ensemble goes next: which seam gets a bridge, which community
//! each chain works, when an exploration policy changes. A decree is
//! only safe to act on when a majority agrees it is part of the shared
//! history, which is the raft guarantee: elected leadership, an
//! append-only replicated log, and commitment by majority match with
//! the current-term restriction that closes the stale-commit hazard.
//!
//! The node here is pure state and arithmetic: no clock, no socket, no
//! randomness. The caller supplies logical time to [`RaftNode::tick`]
//! and delivers messages to [`RaftNode::receive`]; both return the
//! messages to send, and any transport that carries them faithfully
//! (an in-memory bus in tests, nng between processes) yields the same
//! transcripts. Election timeouts stagger deterministically by node
//! identity, so a quiet cluster elects the same leader every replay.

use std::collections::{BTreeMap, BTreeSet};

/// Node identity inside one consensus group.
pub type NodeId = u32;
/// Election term.
pub type Term = u64;

/// One replicated decree: an opaque payload the state machine above
/// interprets (the wire format is the capnp decree, never inspected
/// here).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decree {
    /// Term the decree was proposed in.
    pub term: Term,
    /// Opaque decree payload.
    pub payload: Vec<u8>,
}

/// Messages between raft nodes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RaftMessage {
    /// A candidate asks for a vote.
    RequestVote {
        /// Candidate's term.
        term: Term,
        /// Index of the candidate's last log entry.
        last_log_index: u64,
        /// Term of the candidate's last log entry.
        last_log_term: Term,
    },
    /// A voter answers a vote request.
    VoteReply {
        /// Voter's current term.
        term: Term,
        /// Whether the vote was granted.
        granted: bool,
    },
    /// Leader replication and heartbeat.
    AppendEntries {
        /// Leader's term.
        term: Term,
        /// Index of the entry preceding `entries`.
        prev_log_index: u64,
        /// Term of the entry preceding `entries`.
        prev_log_term: Term,
        /// Entries to append; empty for a heartbeat.
        entries: Vec<Decree>,
        /// Leader's commit index.
        leader_commit: u64,
    },
    /// Follower answers replication.
    AppendReply {
        /// Follower's current term.
        term: Term,
        /// Whether the append matched.
        success: bool,
        /// Highest index known replicated on the follower when
        /// successful; the follower's log length hint otherwise.
        match_index: u64,
    },
}

/// The three raft roles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    /// Accepts decrees from the current leader.
    Follower,
    /// Standing for election.
    Candidate,
    /// Replicates decrees and answers proposals.
    Leader,
}

/// Proposing on a node that is not the leader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("decree proposed on a non-leader")]
pub struct NotLeader;

/// One raft node's complete state.
#[derive(Debug, Clone)]
pub struct RaftNode {
    id: NodeId,
    peers: Vec<NodeId>,
    term: Term,
    voted_for: Option<NodeId>,
    role: Role,
    log: Vec<Decree>,
    commit_index: u64,
    delivered: u64,
    votes: BTreeSet<NodeId>,
    next_index: BTreeMap<NodeId, u64>,
    match_index: BTreeMap<NodeId, u64>,
    election_timeout: u64,
    heartbeat_interval: u64,
    election_deadline: u64,
    heartbeat_due: u64,
    leader_hint: Option<NodeId>,
}

impl RaftNode {
    /// A follower with a deterministic, identity-staggered election
    /// timeout: `base + id * stagger` ticks of silence start an
    /// election, so no two quiet nodes fire together and the same
    /// configuration elects the same first leader on every replay.
    pub fn new(id: NodeId, peers: Vec<NodeId>, base_timeout: u64, stagger: u64) -> Self {
        let election_timeout = base_timeout + u64::from(id) * stagger.max(1);
        Self {
            id,
            peers,
            term: 0,
            voted_for: None,
            role: Role::Follower,
            log: Vec::new(),
            commit_index: 0,
            delivered: 0,
            votes: BTreeSet::new(),
            next_index: BTreeMap::new(),
            match_index: BTreeMap::new(),
            election_timeout,
            heartbeat_interval: (base_timeout / 4).max(1),
            election_deadline: election_timeout,
            heartbeat_due: 0,
            leader_hint: None,
        }
    }

    /// This node's identity.
    pub fn id(&self) -> NodeId {
        self.id
    }

    /// Current role.
    pub fn role(&self) -> Role {
        self.role
    }

    /// Current term.
    pub fn term(&self) -> Term {
        self.term
    }

    /// The node this one believes leads, when any.
    pub fn leader_hint(&self) -> Option<NodeId> {
        if self.role == Role::Leader {
            Some(self.id)
        } else {
            self.leader_hint
        }
    }

    /// Committed decrees not yet handed to the state machine. Each
    /// decree is delivered exactly once, in log order.
    pub fn take_committed(&mut self) -> Vec<Decree> {
        let start = usize::try_from(self.delivered).expect("log fits memory");
        let end = usize::try_from(self.commit_index).expect("log fits memory");
        let slice = self.log[start..end].to_vec();
        self.delivered = self.commit_index;
        slice
    }

    /// Propose a decree; only the leader accepts. Returns the log
    /// index the decree occupies.
    pub fn propose(&mut self, payload: Vec<u8>) -> Result<u64, NotLeader> {
        if self.role != Role::Leader {
            return Err(NotLeader);
        }
        self.log.push(Decree {
            term: self.term,
            payload,
        });
        let index = self.log.len() as u64;
        self.match_index.insert(self.id, index);
        // A single-node group commits its own decrees immediately.
        self.advance_commit();
        Ok(index)
    }

    fn last_log_index(&self) -> u64 {
        self.log.len() as u64
    }

    fn last_log_term(&self) -> Term {
        self.log.last().map_or(0, |entry| entry.term)
    }

    fn become_follower(&mut self, term: Term, now: u64) {
        self.term = term;
        self.role = Role::Follower;
        self.voted_for = None;
        self.votes.clear();
        self.election_deadline = now + self.election_timeout;
    }

    fn quorum(&self) -> usize {
        self.peers.len().div_ceil(2) + 1
    }

    fn broadcast(&self, message: RaftMessage) -> Vec<(NodeId, RaftMessage)> {
        self.peers
            .iter()
            .map(|&peer| (peer, message.clone()))
            .collect()
    }

    fn append_for(&self, peer: NodeId) -> RaftMessage {
        let next = self.next_index.get(&peer).copied().unwrap_or(1).max(1);
        let prev_log_index = next - 1;
        let prev_log_term = if prev_log_index == 0 {
            0
        } else {
            self.log[usize::try_from(prev_log_index - 1).expect("log fits memory")].term
        };
        let entries =
            self.log[usize::try_from(prev_log_index).expect("log fits memory")..].to_vec();
        RaftMessage::AppendEntries {
            term: self.term,
            prev_log_index,
            prev_log_term,
            entries,
            leader_commit: self.commit_index,
        }
    }

    fn advance_commit(&mut self) {
        // Majority match, restricted to entries of the current term:
        // the Figure 8 rule that a leader never counts replication of
        // an older term toward commitment.
        let mut candidate = self.commit_index;
        for index in (self.commit_index + 1)..=self.last_log_index() {
            let replicated = 1 + self
                .peers
                .iter()
                .filter(|peer| self.match_index.get(peer).copied().unwrap_or(0) >= index)
                .count();
            let entry_term = self.log[usize::try_from(index - 1).expect("log fits memory")].term;
            if replicated >= self.quorum() && entry_term == self.term {
                candidate = index;
            }
        }
        self.commit_index = candidate;
    }

    /// Advance logical time; returns messages to send.
    pub fn tick(&mut self, now: u64) -> Vec<(NodeId, RaftMessage)> {
        match self.role {
            Role::Leader => {
                if now >= self.heartbeat_due {
                    self.heartbeat_due = now + self.heartbeat_interval;
                    return self
                        .peers
                        .iter()
                        .map(|&peer| (peer, self.append_for(peer)))
                        .collect();
                }
                Vec::new()
            }
            Role::Follower | Role::Candidate => {
                if now >= self.election_deadline {
                    self.term += 1;
                    self.role = Role::Candidate;
                    self.voted_for = Some(self.id);
                    self.votes.clear();
                    self.votes.insert(self.id);
                    self.election_deadline = now + self.election_timeout;
                    if self.votes.len() >= self.quorum() {
                        // A group of one leads itself.
                        self.become_leader(now);
                        return Vec::new();
                    }
                    return self.broadcast(RaftMessage::RequestVote {
                        term: self.term,
                        last_log_index: self.last_log_index(),
                        last_log_term: self.last_log_term(),
                    });
                }
                Vec::new()
            }
        }
    }

    fn become_leader(&mut self, now: u64) {
        self.role = Role::Leader;
        self.heartbeat_due = now;
        self.next_index = self
            .peers
            .iter()
            .map(|&peer| (peer, self.last_log_index() + 1))
            .collect();
        self.match_index = self.peers.iter().map(|&peer| (peer, 0)).collect();
        self.match_index.insert(self.id, self.last_log_index());
    }

    /// Deliver one message; returns messages to send.
    pub fn receive(
        &mut self,
        from: NodeId,
        message: RaftMessage,
        now: u64,
    ) -> Vec<(NodeId, RaftMessage)> {
        match message {
            RaftMessage::RequestVote {
                term,
                last_log_index,
                last_log_term,
            } => {
                if term > self.term {
                    self.become_follower(term, now);
                }
                let log_ok = (last_log_term, last_log_index)
                    >= (self.last_log_term(), self.last_log_index());
                let granted = term == self.term
                    && log_ok
                    && (self.voted_for.is_none() || self.voted_for == Some(from));
                if granted {
                    self.voted_for = Some(from);
                    self.election_deadline = now + self.election_timeout;
                }
                vec![(
                    from,
                    RaftMessage::VoteReply {
                        term: self.term,
                        granted,
                    },
                )]
            }
            RaftMessage::VoteReply { term, granted } => {
                if term > self.term {
                    self.become_follower(term, now);
                    return Vec::new();
                }
                if self.role == Role::Candidate && term == self.term && granted {
                    self.votes.insert(from);
                    if self.votes.len() >= self.quorum() {
                        self.become_leader(now);
                        return self
                            .peers
                            .iter()
                            .map(|&peer| (peer, self.append_for(peer)))
                            .collect();
                    }
                }
                Vec::new()
            }
            RaftMessage::AppendEntries {
                term,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit,
            } => {
                if term < self.term {
                    return vec![(
                        from,
                        RaftMessage::AppendReply {
                            term: self.term,
                            success: false,
                            match_index: self.last_log_index(),
                        },
                    )];
                }
                if term > self.term || self.role != Role::Follower {
                    self.become_follower(term, now);
                }
                self.leader_hint = Some(from);
                self.election_deadline = now + self.election_timeout;
                let prev_ok = prev_log_index == 0
                    || self
                        .log
                        .get(usize::try_from(prev_log_index - 1).expect("log fits memory"))
                        .is_some_and(|entry| entry.term == prev_log_term);
                if !prev_ok {
                    return vec![(
                        from,
                        RaftMessage::AppendReply {
                            term: self.term,
                            success: false,
                            match_index: self
                                .last_log_index()
                                .min(prev_log_index.saturating_sub(1)),
                        },
                    )];
                }
                // Truncate any conflicting suffix, then append.
                let base = usize::try_from(prev_log_index).expect("log fits memory");
                for (offset, entry) in entries.iter().enumerate() {
                    let position = base + offset;
                    if self
                        .log
                        .get(position)
                        .is_some_and(|existing| existing.term != entry.term)
                    {
                        self.log.truncate(position);
                    }
                    if self.log.len() == position {
                        self.log.push(entry.clone());
                    }
                }
                let match_index = (base + entries.len()) as u64;
                self.commit_index = self
                    .commit_index
                    .max(leader_commit.min(self.last_log_index()));
                vec![(
                    from,
                    RaftMessage::AppendReply {
                        term: self.term,
                        success: true,
                        match_index,
                    },
                )]
            }
            RaftMessage::AppendReply {
                term,
                success,
                match_index,
            } => {
                if term > self.term {
                    self.become_follower(term, now);
                    return Vec::new();
                }
                if self.role != Role::Leader || term != self.term {
                    return Vec::new();
                }
                if success {
                    self.match_index.insert(from, match_index);
                    self.next_index.insert(from, match_index + 1);
                    self.advance_commit();
                    Vec::new()
                } else {
                    let next = self.next_index.entry(from).or_insert(1);
                    *next = (*next).saturating_sub(1).max(1).min(match_index + 1);
                    vec![(from, self.append_for(from))]
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A deterministic bus: FIFO delivery, optional partitions.
    struct Cluster {
        nodes: Vec<RaftNode>,
        bus: VecDeque<(NodeId, NodeId, RaftMessage)>,
        cut: BTreeSet<NodeId>,
        now: u64,
    }

    impl Cluster {
        fn new(size: u32) -> Self {
            let ids: Vec<NodeId> = (0..size).collect();
            let nodes = ids
                .iter()
                .map(|&id| {
                    let peers = ids.iter().copied().filter(|&p| p != id).collect();
                    RaftNode::new(id, peers, 20, 7)
                })
                .collect();
            Self {
                nodes,
                bus: VecDeque::new(),
                cut: BTreeSet::new(),
                now: 0,
            }
        }

        fn send(&mut self, from: NodeId, batch: Vec<(NodeId, RaftMessage)>) {
            for (to, message) in batch {
                if !self.cut.contains(&from) && !self.cut.contains(&to) {
                    self.bus.push_back((from, to, message));
                }
            }
        }

        fn step(&mut self) {
            self.now += 1;
            for index in 0..self.nodes.len() {
                let out = self.nodes[index].tick(self.now);
                let from = self.nodes[index].id();
                self.send(from, out);
            }
            // Drain everything currently queued, so a step is one tick
            // of time plus complete message settlement.
            while let Some((from, to, message)) = self.bus.pop_front() {
                let index = to as usize;
                let out = self.nodes[index].receive(from, message, self.now);
                self.send(to, out);
            }
        }

        fn run(&mut self, steps: u64) {
            for _ in 0..steps {
                self.step();
            }
        }

        fn leaders(&self) -> Vec<NodeId> {
            self.nodes
                .iter()
                .filter(|node| node.role() == Role::Leader)
                .map(RaftNode::id)
                .collect()
        }
    }

    #[test]
    fn a_quiet_cluster_elects_exactly_one_leader() {
        let mut cluster = Cluster::new(3);
        cluster.run(60);
        assert_eq!(
            cluster.leaders().len(),
            1,
            "leaders: {:?}",
            cluster.leaders()
        );
        // The stagger makes node 0 time out first, every replay.
        assert_eq!(cluster.leaders(), vec![0]);
        let mut replay = Cluster::new(3);
        replay.run(60);
        assert_eq!(replay.leaders(), vec![0]);
    }

    #[test]
    fn no_step_ever_shows_two_leaders_in_one_term() {
        let mut cluster = Cluster::new(5);
        for _ in 0..200 {
            cluster.step();
            let mut by_term = BTreeMap::<Term, usize>::new();
            for node in &cluster.nodes {
                if node.role() == Role::Leader {
                    *by_term.entry(node.term()).or_default() += 1;
                }
            }
            for (term, count) in by_term {
                assert!(count <= 1, "term {term} has {count} leaders");
            }
        }
    }

    #[test]
    fn decrees_replicate_and_commit_on_every_node() {
        let mut cluster = Cluster::new(3);
        cluster.run(60);
        let leader = cluster.leaders()[0] as usize;
        cluster.nodes[leader].propose(b"decree-a".to_vec()).unwrap();
        cluster.nodes[leader].propose(b"decree-b".to_vec()).unwrap();
        cluster.run(20);
        for node in &mut cluster.nodes {
            let committed = node.take_committed();
            let payloads: Vec<&[u8]> = committed.iter().map(|d| d.payload.as_slice()).collect();
            assert_eq!(
                payloads,
                vec![b"decree-a".as_slice(), b"decree-b".as_slice()],
                "node {} log diverged",
                node.id()
            );
        }
        // Delivery is exactly once.
        for node in &mut cluster.nodes {
            assert!(node.take_committed().is_empty());
        }
    }

    #[test]
    fn followers_reject_proposals() {
        let mut cluster = Cluster::new(3);
        cluster.run(60);
        let follower = cluster
            .nodes
            .iter()
            .position(|node| node.role() != Role::Leader)
            .unwrap();
        assert_eq!(
            cluster.nodes[follower].propose(b"x".to_vec()),
            Err(NotLeader)
        );
    }

    #[test]
    fn a_partitioned_leader_steps_down_and_the_cluster_recovers() {
        let mut cluster = Cluster::new(3);
        cluster.run(60);
        let old_leader = cluster.leaders()[0];
        cluster.cut.insert(old_leader);
        cluster.run(200);
        let survivors = cluster.leaders();
        let new_leader = survivors
            .iter()
            .copied()
            .find(|&id| id != old_leader)
            .expect("majority side elected a replacement");
        // Commit a decree on the majority side while the old leader is
        // isolated.
        cluster.nodes[new_leader as usize]
            .propose(b"post-partition".to_vec())
            .unwrap();
        cluster.run(20);
        // Heal: the stale leader rejoins, observes the higher term, and
        // steps down; its log converges on the majority history.
        cluster.cut.clear();
        cluster.run(60);
        assert_eq!(cluster.leaders().len(), 1);
        for node in &mut cluster.nodes {
            let committed = node.take_committed();
            assert_eq!(
                committed.last().map(|d| d.payload.clone()),
                Some(b"post-partition".to_vec()),
                "node {} missed the majority decree",
                node.id()
            );
        }
    }

    #[test]
    fn a_single_node_group_leads_and_commits_alone() {
        let mut node = RaftNode::new(0, Vec::new(), 20, 7);
        let out = node.tick(21);
        assert!(out.is_empty());
        assert_eq!(node.role(), Role::Leader);
        node.propose(b"solo".to_vec()).unwrap();
        assert_eq!(node.take_committed().len(), 1);
    }
}

/// Wire encoding of consensus traffic and decree payloads.
#[cfg(feature = "bank-rpc")]
pub mod wire {
    use super::{Decree, NodeId, RaftMessage};
    use crate::Raft_capnp::{exploration_decree, raft_envelope};
    use capnp::message::{Builder, ReaderOptions};
    use capnp::serialize;
    use std::io::Cursor;

    /// One replica's marching orders inside an exploration decree.
    #[derive(Debug, Clone, PartialEq)]
    pub struct ReplicaAssignment {
        /// Replica identifier.
        pub replica: u32,
        /// Seam side to work: `false` left, `true` right.
        pub right_side: bool,
        /// Class indices of the histogram target.
        pub histogram_classes: Vec<u32>,
        /// Normalized masses of the histogram target.
        pub histogram_masses: Vec<f64>,
        /// Anchor basin for boundary transport on the assigned side.
        pub anchor_basin: u64,
        /// Whether the replica runs confined bridge segments.
        pub bridge_duty: bool,
        /// Decree sequence for tracing.
        pub decree_index: u64,
    }

    /// The decree payload: seam evidence and per-replica assignments.
    #[derive(Debug, Clone, PartialEq)]
    pub struct ExplorationDecree {
        /// Second Laplacian eigenvalue behind the decree.
        pub algebraic_connectivity: f64,
        /// Conductance of the seam behind the decree.
        pub seam_conductance: f64,
        /// Left representative basin.
        pub left_basin: u64,
        /// Right representative basin.
        pub right_basin: u64,
        /// Assignments in replica order.
        pub assignments: Vec<ReplicaAssignment>,
    }

    /// Encode a decree payload.
    pub fn encode_decree(decree: &ExplorationDecree) -> Vec<u8> {
        let mut message = Builder::new_default();
        let mut root = message.init_root::<exploration_decree::Builder>();
        root.set_algebraic_connectivity(decree.algebraic_connectivity);
        root.set_seam_conductance(decree.seam_conductance);
        root.set_left_basin(decree.left_basin);
        root.set_right_basin(decree.right_basin);
        let mut assignments = root.init_assignments(decree.assignments.len() as u32);
        for (index, assignment) in decree.assignments.iter().enumerate() {
            let mut row = assignments.reborrow().get(index as u32);
            row.set_replica(assignment.replica);
            row.set_side(u8::from(assignment.right_side));
            row.set_anchor_basin(assignment.anchor_basin);
            row.set_bridge_duty(assignment.bridge_duty);
            row.set_decree_index(assignment.decree_index);
            {
                let mut classes = row
                    .reborrow()
                    .init_histogram_classes(assignment.histogram_classes.len() as u32);
                for (i, class) in assignment.histogram_classes.iter().enumerate() {
                    classes.set(i as u32, *class);
                }
            }
            let mut masses = row
                .reborrow()
                .init_histogram_masses(assignment.histogram_masses.len() as u32);
            for (i, mass) in assignment.histogram_masses.iter().enumerate() {
                masses.set(i as u32, *mass);
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &message).expect("in-memory write cannot fail");
        bytes
    }

    /// Decode a decree payload.
    pub fn decode_decree(bytes: &[u8]) -> Result<ExplorationDecree, capnp::Error> {
        let reader = serialize::read_message(&mut Cursor::new(bytes), ReaderOptions::new())?;
        let root = reader.get_root::<exploration_decree::Reader>()?;
        let mut assignments = Vec::new();
        for row in root.get_assignments()?.iter() {
            assignments.push(ReplicaAssignment {
                replica: row.get_replica(),
                right_side: row.get_side() != 0,
                histogram_classes: row.get_histogram_classes()?.iter().collect(),
                histogram_masses: row.get_histogram_masses()?.iter().collect(),
                anchor_basin: row.get_anchor_basin(),
                bridge_duty: row.get_bridge_duty(),
                decree_index: row.get_decree_index(),
            });
        }
        Ok(ExplorationDecree {
            algebraic_connectivity: root.get_algebraic_connectivity(),
            seam_conductance: root.get_seam_conductance(),
            left_basin: root.get_left_basin(),
            right_basin: root.get_right_basin(),
            assignments,
        })
    }

    /// Encode one consensus message between two brains.
    pub fn encode_envelope(from: NodeId, to: NodeId, message: &RaftMessage) -> Vec<u8> {
        let mut builder = Builder::new_default();
        let mut root = builder.init_root::<raft_envelope::Builder>();
        root.set_from(from);
        root.set_to(to);
        let wire = root.init_message();
        match message {
            RaftMessage::RequestVote {
                term,
                last_log_index,
                last_log_term,
            } => {
                let mut vote = wire.init_request_vote();
                vote.set_term(*term);
                vote.set_last_log_index(*last_log_index);
                vote.set_last_log_term(*last_log_term);
            }
            RaftMessage::VoteReply { term, granted } => {
                let mut reply = wire.init_vote_reply();
                reply.set_term(*term);
                reply.set_granted(*granted);
            }
            RaftMessage::AppendEntries {
                term,
                prev_log_index,
                prev_log_term,
                entries,
                leader_commit,
            } => {
                let mut append = wire.init_append_entries();
                append.set_term(*term);
                append.set_prev_log_index(*prev_log_index);
                append.set_prev_log_term(*prev_log_term);
                append.set_leader_commit(*leader_commit);
                let mut wire_entries = append.init_entries(entries.len() as u32);
                for (index, entry) in entries.iter().enumerate() {
                    let mut row = wire_entries.reborrow().get(index as u32);
                    row.set_term(entry.term);
                    row.set_payload(&entry.payload);
                }
            }
            RaftMessage::AppendReply {
                term,
                success,
                match_index,
            } => {
                let mut reply = wire.init_append_reply();
                reply.set_term(*term);
                reply.set_success(*success);
                reply.set_match_index(*match_index);
            }
        }
        let mut bytes = Vec::new();
        serialize::write_message(&mut bytes, &builder).expect("in-memory write cannot fail");
        bytes
    }

    /// Decode one consensus message.
    pub fn decode_envelope(bytes: &[u8]) -> Result<(NodeId, NodeId, RaftMessage), capnp::Error> {
        let reader = serialize::read_message(&mut Cursor::new(bytes), ReaderOptions::new())?;
        let root = reader.get_root::<raft_envelope::Reader>()?;
        let message = match root.get_message().which()? {
            raft_envelope::message::RequestVote(vote) => RaftMessage::RequestVote {
                term: vote.get_term(),
                last_log_index: vote.get_last_log_index(),
                last_log_term: vote.get_last_log_term(),
            },
            raft_envelope::message::VoteReply(reply) => RaftMessage::VoteReply {
                term: reply.get_term(),
                granted: reply.get_granted(),
            },
            raft_envelope::message::AppendEntries(append) => {
                let mut entries = Vec::new();
                for row in append.get_entries()?.iter() {
                    entries.push(Decree {
                        term: row.get_term(),
                        payload: row.get_payload()?.to_vec(),
                    });
                }
                RaftMessage::AppendEntries {
                    term: append.get_term(),
                    prev_log_index: append.get_prev_log_index(),
                    prev_log_term: append.get_prev_log_term(),
                    entries,
                    leader_commit: append.get_leader_commit(),
                }
            }
            raft_envelope::message::AppendReply(reply) => RaftMessage::AppendReply {
                term: reply.get_term(),
                success: reply.get_success(),
                match_index: reply.get_match_index(),
            },
        };
        Ok((root.get_from(), root.get_to(), message))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn every_envelope_round_trips() {
            let messages = vec![
                RaftMessage::RequestVote {
                    term: 3,
                    last_log_index: 7,
                    last_log_term: 2,
                },
                RaftMessage::VoteReply {
                    term: 3,
                    granted: true,
                },
                RaftMessage::AppendEntries {
                    term: 4,
                    prev_log_index: 6,
                    prev_log_term: 2,
                    entries: vec![
                        Decree {
                            term: 4,
                            payload: b"a".to_vec(),
                        },
                        Decree {
                            term: 4,
                            payload: Vec::new(),
                        },
                    ],
                    leader_commit: 5,
                },
                RaftMessage::AppendReply {
                    term: 4,
                    success: false,
                    match_index: 6,
                },
            ];
            for message in messages {
                let bytes = encode_envelope(2, 0, &message);
                assert_eq!(decode_envelope(&bytes).unwrap(), (2, 0, message));
            }
        }

        #[test]
        fn a_decree_round_trips_with_histogram_targets() {
            let decree = ExplorationDecree {
                algebraic_connectivity: 0.013,
                seam_conductance: 0.004,
                left_basin: 11,
                right_basin: 29,
                assignments: vec![ReplicaAssignment {
                    replica: 2,
                    right_side: true,
                    histogram_classes: vec![0, 2],
                    histogram_masses: vec![0.13, 0.27],
                    anchor_basin: 29,
                    bridge_duty: true,
                    decree_index: 5,
                }],
            };
            let bytes = encode_decree(&decree);
            assert_eq!(decode_decree(&bytes).unwrap(), decree);
        }
    }
}
