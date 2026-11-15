`EMTPot`, `D3Pot` and `D4Pot` rebuild their cached structure when the atomic
numbers change. Before, they did this only when the atom count changed, so a
reused instance gave the energy and forces of the previous species.
