@0xb6e1f0c2a4d38e91;

# Consensus traffic between per-chain server brains, and the decrees
# they replicate. The raft state machine is transport-blind; this is
# the one wire encoding every transport carries.

struct Decree {
  term @0 :UInt64;
  payload @1 :Data;
}

# An exploration decree payload: where the leader sends each replica
# next. Versioned by the log index it is committed at; a chain applies
# the highest committed decree it has seen and never blocks waiting
# for a newer one.
struct ExplorationDecree {
  # Seam evidence the decree was computed from.
  algebraicConnectivity @0 :Float64;
  seamConductance @1 :Float64;
  leftBasin @2 :UInt64;
  rightBasin @3 :UInt64;
  # Per-replica assignments, in replica order.
  assignments @4 :List(ReplicaAssignment);
}

struct ReplicaAssignment {
  replica @0 :UInt32;
  # Which seam side the replica works: 0 left, 1 right.
  side @1 :UInt8;
  # Class-histogram target: class indices and normalized masses the
  # replica's escape screen should move toward. Empty means pure
  # novelty.
  histogramClasses @5 :List(UInt32);
  histogramMasses @6 :List(Float64);
  # Anchor basin for boundary transport on the assigned side.
  anchorBasin @2 :UInt64;
  # Whether the replica should run confined bridge segments.
  bridgeDuty @3 :Bool;
  # Decree sequence for tracing.
  decreeIndex @4 :UInt64;
}

struct RaftEnvelope {
  from @0 :UInt32;
  to @1 :UInt32;

  message :union {
    requestVote :group {
      term @2 :UInt64;
      lastLogIndex @3 :UInt64;
      lastLogTerm @4 :UInt64;
    }
    voteReply :group {
      term @5 :UInt64;
      granted @6 :Bool;
    }
    appendEntries :group {
      term @7 :UInt64;
      prevLogIndex @8 :UInt64;
      prevLogTerm @9 :UInt64;
      entries @10 :List(Decree);
      leaderCommit @11 :UInt64;
    }
    appendReply :group {
      term @12 :UInt64;
      success @13 :Bool;
      matchIndex @14 :UInt64;
    }
  }
}
