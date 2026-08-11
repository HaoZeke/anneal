@0xbd8e4c1a7f2093e5;

# Shared CSA bank + packing superbasin. Chains are separate processes.
# Distance on the wire is unit high-l mean SOAP (packing), not leftover RMS.
# IRA/SOFI stay on the hop; this fabric carries known packings and bias.

struct Offer {
  energy @0 :Float64;
  coords @1 :List(Float64);
  soap @2 :List(Float64);
}

struct Deposit {
  soap @0 :List(Float64);
  increment @1 :Float64;
}

struct BankRequest {
  union {
    offer @0 :Offer;
    nearest @1 :List(Float64);
    deposit @2 :Deposit;
    biasOf @3 :List(Float64);
    sample @4 :UInt64;
    snapshot @5 :Void;
    setDcut @6 :Float64;
  }
}

struct Well {
  soap @0 :List(Float64);
  height @1 :Float64;
}

struct BankReply {
  kind @0 :UInt16;
  energy @1 :Float64;
  dcut @2 :Float64;
  distance @3 :Float64;
  height @4 :Float64;
  size @5 :UInt32;
  coords @6 :List(Float64);
  energies @7 :List(Float64);
  empty @8 :Bool;
  wells @9 :List(Well);
}
