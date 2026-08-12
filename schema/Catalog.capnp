@0x8d9e5bb4eab31f27;

# Versioned cooperative catalog protocol. Every request carries the complete
# ensemble and system identity required before scientific state can mutate.

struct CatalogIdentity {
  campaign @0 :Text;
  ensemble @1 :Text;
  replica @2 :UInt32;
  signatureDigest @3 :Data;
}

struct CensusObservation {
  basinId @0 :UInt64;
  created @1 :Bool;
  descriptor @2 :List(Float64);
}

struct CandidateRecord {
  basinId @0 :UInt64;
  energy @1 :Float64;
  coordinates @2 :List(Float64);
  descriptor @3 :List(Float64);
  provenance @4 :Text;
}

struct LedgerEvent {
  kind @0 :UInt16;
  chargedCalls @1 :UInt64;
  cumulativeCharged @2 :UInt64;
}

struct CatalogRequest {
  protocolVersion @0 :UInt16;
  identity @1 :CatalogIdentity;
  eventSequence @2 :UInt64;
  snapshotVersion @3 :UInt64;

  operation :union {
    snapshot @4 :Void;
    recordVisit @5 :CensusObservation;
    offerCandidate @6 :CandidateRecord;
    sample @7 :UInt64;
    descriptorHole @8 :UInt32;
    ledgerEvent @9 :LedgerEvent;
  }
}

enum RejectionKind {
  malformed @0;
  unsupportedVersion @1;
  campaignMismatch @2;
  ensembleMismatch @3;
  replicaMismatch @4;
  signatureMismatch @5;
  sequenceReplay @6;
  sequenceRegression @7;
  snapshotRegression @8;
  validationRejected @9;
}

struct CatalogReply {
  protocolVersion @0 :UInt16;
  eventSequence @1 :UInt64;
  snapshotVersion @2 :UInt64;

  result :union {
    accepted @3 :Data;
    rejected @4 :RejectionKind;
  }
}
