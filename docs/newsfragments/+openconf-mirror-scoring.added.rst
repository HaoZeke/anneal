The openconf pairing scorer (``scripts/openconf_h2h/paired_score.py``) writes a ``hit_mirror`` column beside ``hit``.
``hit_mirror`` also accepts a candidate whose reflection is the reference conformer, and it counts that match only when the reflected geometry keeps the stereo key of the input SMILES.
A molecule without handedness, or a meso one, accepts the reflection; a chiral molecule never does, since its reflection is the other enantiomer.
``hit`` is unchanged.
