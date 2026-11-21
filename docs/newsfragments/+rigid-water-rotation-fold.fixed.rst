Folding the rotation vectors of rigid TIP4P molecules into the ball of radius pi takes one step for any angle, where it subtracted a turn at a time.
A rotation of ``1e12`` took as many steps as it had turns, and one that was infinite or NaN never ended; both now return, and a rotation that is not finite is left as it was given.
A finite rotation folds to the same vector as before, to ``1e-12``.
