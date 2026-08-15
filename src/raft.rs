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
        (self.peers.len() + 1) / 2 + 1
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
        let entries = self.log[usize::try_from(prev_log_index).expect("log fits memory")..].to_vec();
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
            let entry_term =
                self.log[usize::try_from(index - 1).expect("log fits memory")].term;
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
                            match_index: self.last_log_index().min(prev_log_index.saturating_sub(1)),
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
        assert_eq!(cluster.leaders().len(), 1, "leaders: {:?}", cluster.leaders());
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
            let payloads: Vec<&[u8]> =
                committed.iter().map(|d| d.payload.as_slice()).collect();
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
