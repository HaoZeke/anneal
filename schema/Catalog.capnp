@0x8d9e5bb4eab31f27;

# Versioned cooperative catalog protocol. Every request carries the complete
# ensemble and system identity required before scientific state can mutate.

struct CatalogIdentity {
  campaign @0 :Text;
  ensemble @1 :Text;
  replica @2 :UInt32;
  signatureDigest @3 :Data;
}

struct CandidateRecord {
  producerReplica @0 :UInt32;
  coordinates @1 :List(Float64);
  cell :union {
    absent @2 :Void;
    present @3 :List(Float64);
  }
  energy @4 :Float64;
  forces @5 :List(Float64);
  gradientNorm @6 :Float64;
  descriptor @7 :List(Float64);
  descriptorSchemaVersion @8 :UInt32;
  quenchStatus @9 :QuenchStatus;
  chargedWork @10 :UInt64;
  eventSequence @11 :UInt64;
  seed @12 :UInt64;
  censusBasin :union {
    unassigned @13 :Void;
    assigned @14 :UInt64;
  }
}

enum QuenchStatus {
  unconverged @0;
  converged @1;
}

struct LedgerEvent {
  kind @0 :UInt16;
  chargedCalls @1 :UInt64;
  cumulativeCharged @2 :UInt64;
}

struct DescriptorHoleRequest {
  current @0 :List(Float64);
  samples @1 :UInt32;
  draw @2 :UInt64;
}

struct DescriptorHoleReply {
  target @0 :List(Float64);
  increment @1 :List(Float64);
  nearestCatalogDistance @2 :Float64;
}

struct BoundaryCrossingRequest {
  current @0 :List(Float64);
  draw @1 :UInt64;
}

struct PolicyStateRequest {
  descriptor @0 :List(Float64);
  energy @1 :Float64;
}

enum CatalogRelation {
  empty @0;
  incumbent @1;
  sameBasin @2;
  unrelatedNoAnchor @3;
  unrelatedLowerAnchor @4;
}

struct PolicyStateReply {
  totalVisits @0 :UInt64;
  singletonBasins @1 :UInt64;
  localBasinVisits @2 :UInt64;
  globallySaturated @3 :Bool;
  relation @4 :CatalogRelation;
  aggregateCharged @5 :UInt64;
  aggregateBudget @6 :UInt64;
  localBasin :union {
    unassigned @7 :Void;
    assigned @8 :UInt64;
  }
  localBasinDistance @9 :Float64;
  novelty @10 :Float64;
  transitionUncertainty @11 :Float64;
}

enum CatalogMutationKind {
  added @0;
  replacedSameBasin @1;
  replacedConflicts @2;
  replacedCapacity @3;
  rejectedSameBasin @4;
  rejectedConflict @5;
  rejectedCapacity @6;
}

struct CatalogMutationReply {
  basinId @0 :UInt64;
  kind @1 :CatalogMutationKind;
  evicted @2 :List(UInt64);
  incumbentBasin :union {
    absent @3 :Void;
    present @4 :UInt64;
  }
}

struct PopulationSubmitRequest {
  epoch @0 :UInt64;
  candidate @1 :CandidateRecord;
}

struct PopulationPlanRequest {
  epoch @0 :UInt64;
}

struct TransitionRecord {
  action @0 :Text;
  adopted @3 :Bool;
  destination :union {
    unresolved @1 :Void;
    resolved @2 :CandidateRecord;
  }
}

struct PopulationPlan {
  epoch @0 :UInt64;
  destinations @1 :List(UInt32);
  parents @2 :List(UInt32);
  weights @3 :List(Float64);
  effectiveSampleSize @4 :Float64;
  uniqueParents @5 :UInt32;
  maxFamilySize @6 :UInt32;
  offspringVariance @7 :Float64;
  parentCandidates @8 :List(CandidateRecord);
}

struct PopulationEpochReply {
  epoch @0 :UInt64;
  submitted @1 :UInt32;
  required @2 :UInt32;
  result :union {
    pending @3 :Void;
    ready @4 :PopulationPlan;
  }
}

struct CatalogRequest {
  protocolVersion @0 :UInt16;
  identity @1 :CatalogIdentity;
  eventSequence @2 :UInt64;
  snapshotVersion @3 :UInt64;

  operation :union {
    snapshot @4 :Void;
    recordVisit @5 :CandidateRecord;
    offerCandidate @6 :CandidateRecord;
    sample @7 :UInt64;
    descriptorHole @8 :DescriptorHoleRequest;
    ledgerEvent @9 :LedgerEvent;
    policyState @10 :PolicyStateRequest;
    populationSubmit @11 :PopulationSubmitRequest;
    populationPlan @12 :PopulationPlanRequest;
    recordTransition @13 :TransitionRecord;
    boundaryCrossing @14 :BoundaryCrossingRequest;
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
    accepted @3 :AcceptedReply;
    rejected @4 :RejectionKind;
  }
}

struct AcceptedReply {
  duplicate @0 :Bool;
  censusVisits @1 :UInt64;
  activeEntries @2 :UInt32;
  payload :union {
    none @3 :Void;
    candidate @4 :CandidateRecord;
    descriptorHole @5 :DescriptorHoleReply;
    policyState @6 :PolicyStateReply;
    populationEpoch @9 :PopulationEpochReply;
    catalogMutation @10 :CatalogMutationReply;
  }
  aggregateCharged @7 :UInt64;
  aggregateBudget @8 :UInt64;
}
