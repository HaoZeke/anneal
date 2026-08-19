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

struct BoundaryCrossingReply {
  action @0 :Text;
  from @1 :List(Float64);
  to @2 :List(Float64);
  sourceBasin @3 :UInt64;
  destinationBasin @4 :UInt64;
}

struct PolicyStateRequest {
  descriptor @0 :List(Float64);
  energy @1 :Float64;
  # Highest leftover-SOAP lambda on this replica's Leave path.
  leftoverLambda @2 :Float64;
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
  exploreCollapsed @12 :Bool;
  certifiedAttractor @13 :Bool;
  pruned @14 :Bool;
  leftoverLambda @15 :Float64;
  interfaceRank @16 :UInt32;
  interfaceThreshold @17 :Float64;
  interfaceCount @18 :UInt32;
  occupiedFamilyCount @19 :UInt32;
  # DECAF-family Good-Turing: unseen packing mass is small.
  packingSaturated @20 :Bool;
  # Consecutive leftover-SOAP occupancy_gt records under the ceiling.
  leftoverDwell @21 :Bool;
  # FunnelModel Jones EI on seen packings is at most the model noise.
  eiExhausted @22 :Bool;
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

# Sent when a replica reaches a barrier with no validated representative.
# Without it the epoch waits for a submission that is never coming and the
# replicas already waiting poll until their budgets drain.
struct PopulationAbstainRequest {
  epoch @0 :UInt64;
}

# Join an epoch by reference to the replica's best already-validated
# candidate on file with the coordinator. The client does not re-ship or
# re-validate state at barrier time: the coordinator has fresh-evaluated
# every visit it accepted, so barrier membership is a lookup, not a
# transfer.
struct PopulationJoinRequest {
  epoch @0 :UInt64;
}

# A commissioned bridge between two catalog minima. The string images
# tile the channel between the two descriptors with Voronoi regions;
# a replica holding an assignment confines its sampling to one region,
# restarts from stored entry states, and reports attempted exits as
# crossings. Images are row-major: imageCount rows of the descriptor
# dimension.
struct BridgeAssignment {
  bridge @0 :UInt64;
  fromBasin @1 :UInt64;
  toBasin @2 :UInt64;
  images @3 :List(Float64);
  imageCount @4 :UInt32;
  region @5 :UInt32;
  tubeRadius @6 :Float64;
  entry :union {
    none @7 :Void;
    state @8 :List(Float64);
  }
}

# An attempted exit from a bridge region, reported by the confined
# replica. The coordinator transfers region weight with the attempted
# flux and stores the crossing state as an entry for the target region.
struct BridgeCrossingRequest {
  bridge @0 :UInt64;
  fromRegion @1 :UInt32;
  toRegion @2 :UInt32;
  descriptor @3 :List(Float64);
  state @4 :List(Float64);
  energy @5 :Float64;
}

# Read-only coordinator status for an observer outside the ensemble. A
# server that can only be interrogated by its own four replicas cannot be
# watched, which is how a deadlocked campaign ran for hours looking
# healthy. Aggregates only: no coordinates, no descriptors, and the reply
# is never journaled, because observation is not history.
struct ReplicaProgress {
  replica @0 :UInt32;
  chargedWork @1 :UInt64;
  bestEnergy @2 :Float64;
}

struct CoordinatorStatus {
  snapshotVersion @0 :UInt64;
  openEpoch @1 :UInt64;
  epochSubmitted @2 :UInt32;
  epochRequired @3 :UInt32;
  censusVisits @4 :UInt64;
  activeEntries @5 :UInt32;
  aggregateCharged @6 :UInt64;
  aggregateBudget @7 :UInt64;
  replicas @8 :List(ReplicaProgress);
  # The referee's view of the explored landscape: how many basins the
  # transition stream has linked, how metastable the explored graph is,
  # and the seam between its two most weakly coupled communities. The
  # seam fields carry data only when communityLeft and communityRight
  # are both nonzero; a status written before these fields existed reads
  # as zeros, which is the honest no-referee answer.
  landscapeBasins @9 :UInt32;
  algebraicConnectivity @10 :Float64;
  seamConductance @11 :Float64;
  communityLeft @12 :UInt32;
  communityRight @13 :UInt32;
  seamLeftBasin @14 :UInt64;
  seamRightBasin @15 :UInt64;
}

struct TransitionRecord {
  action @0 :Text;
  adopted @3 :Bool;
  destination :union {
    unresolved @1 :Void;
    resolved @2 :CandidateRecord;
  }
}

# Which selection produced a barrier's parent map. The two branches carry
# different weights, so a reader that cannot tell them apart cannot
# interpret the weights it is given. Zero is unspecified rather than a
# real branch, so a record written before this field existed cannot be
# read as though it had taken one.
enum PopulationSelection {
  unspecified @0;
  systematicResampling @1;
  regionCovering @2;
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
  selection @9 :PopulationSelection;
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
    populationAbstain @15 :PopulationAbstainRequest;
    populationJoin @16 :PopulationJoinRequest;
    observerStatus @17 :Void;
    bridgeAssignment @18 :UInt64;
    bridgeCrossing @19 :BridgeCrossingRequest;
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
    boundaryCrossing @11 :BoundaryCrossingReply;
    coordinatorStatus @12 :CoordinatorStatus;
    bridgeAssignment @13 :BridgeAssignment;
  }
  aggregateCharged @7 :UInt64;
  aggregateBudget @8 :UInt64;
}
