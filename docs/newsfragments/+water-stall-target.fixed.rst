A stalled rigid chain no longer reads the run target or draws a new cluster.
The chain keeps the eight lowest distinct minima of its quenched start, accepted quenches and lowest recorded minimum.
Each stall follows 1000 hops with no new lowest minimum.
At each stall the walker moves to the lowest kept minimum that is neither its own nor one of the last four stall destinations.
When no kept minimum qualifies, the walker stays put.
Two runs now follow one trajectory when their only difference is a target neither reaches.
The recommended preset no longer puts the random restart among a rigid chain's stall arms.
On a call plateau a rigid chain draws no fresh cluster even with ``restart_on_stall`` on and ``escape_on_stall`` off.
