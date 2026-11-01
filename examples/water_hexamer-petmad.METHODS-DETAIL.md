# PET-MAD water hexamer

Engine kind `metatomic-rgpot-v1`. The plugin is `libmetatomic_engine.so`
(`RGPOT_METATOMIC_ENGINE`). The model file cited by the potlib dlopen
test is `pet-mad-xs-v1.5.0.pt`, passed as `RGPOT_METATOMIC_MODEL`.

Device `cpu`. Length unit angstrom. Energy is eV. The force call uses
the same 60 angstrom vacuum box as the GFN2 handle. The model's cutoff
is inside the file and is not named here. Consistency check off,
uncertainty threshold 0, no random rotation. The search gradient is the
negative of the returned force.

Budget 4000. Seeds 0-31. Selector `petmad`. See
`water_hexamer.METHODS-DETAIL.md`.

A missing plugin or model is an engine-load failure. The smoke file
does not invent an energy.
