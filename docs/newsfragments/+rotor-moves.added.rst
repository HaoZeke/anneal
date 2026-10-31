``recommended_molecular`` adds a rotor arm (``ClusterMove::Rotor``) for molecules that can have a rotatable bond.
Rotors are the acyclic single bonds of each hop's own bond graph, with methyl-like terminal groups and amide-like linkages excluded by default (``RotorRules``).
The smaller side turns to a staggered or planar preferred value or a uniform angle, sometimes with adjacent rotors of the same molecule or as a counter-rotation of two parallel rotors.
Draws that clash or leave the container are redrawn before any energy call, and the driver does not clamp a rotor trial to a frozen frame, so it keeps every bond.
Water and other small rigid groups are unchanged.
