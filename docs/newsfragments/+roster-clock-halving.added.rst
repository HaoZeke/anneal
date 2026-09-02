Live catalog roster with Attach/Detach/Tick/Scale, a deterministic
coordinator clock, and successive-halving spawn/retire decisions.
The supervisor launches workers for pending spawns; workers Attach
themselves and honour a retired policy reply.
