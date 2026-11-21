``cluster_search`` and the TIP4P ``Config`` presets (``for_tip4p``, ``recommended_tip4p``, ``communicating_tip4p``) raise for a number of points the machine cannot hold, where they aborted the process or raised ``PanicException``.
A state length that overflows a machine word raises ``InvalidArgumentError``.
``cluster_search`` asks the allocator for what its first hop holds before it builds anything, and raises ``MemoryError`` when the allocator refuses.
The size is added up from the vectors and structures that the hop creates, each at its own length, and ``tests/first_hop_allocation.rs`` fails when it is less than the bytes a real first hop holds.
The round counts it replaces were below those bytes for the ``recommended`` preset (half of them at 38 points), because they left out the table that the symmetry move sorts.
No largest ``n`` is built in.
The checked sizes are in the new ``memory`` module of the Rust crate.
