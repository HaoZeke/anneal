- Added user-facing analytic model support end to end: potserv now serves
  `DoubleWell` and `CoupledProtonWell`, clients can call `calculateHvp` over
  RPC for their analytic Hessian-vector products, and the C model API
  documents constructor and evaluation usage.
