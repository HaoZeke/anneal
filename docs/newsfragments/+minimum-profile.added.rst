Persistent in-process minimum-profile adapters for molecular-cluster and slab
searches. ``nwchemc`` and ``cpmdc`` load once, reuse the same ``ProfileEngine``
for the complete hop loop, and require neither an RPC server nor a result
cache. Molecular requests omit a cell while periodic slab requests carry it
through the same request type.
