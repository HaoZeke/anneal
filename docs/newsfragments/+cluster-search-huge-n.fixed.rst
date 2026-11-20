``cluster_search`` and the TIP4P ``Config`` presets (``for_tip4p``, ``recommended_tip4p``, ``communicating_tip4p``) raise for a number of points the machine cannot hold, where they aborted the process or raised ``PanicException``.
A state length that overflows a machine word raises ``InvalidArgumentError``.
``cluster_search`` asks the allocator for 32 states and 8 pair descriptors before it builds anything, and raises ``MemoryError`` when the allocator refuses.
No largest ``n`` is built in.
The checked sizes are in the new ``memory`` module of the Rust crate.
