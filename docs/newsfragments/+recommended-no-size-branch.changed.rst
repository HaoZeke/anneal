``Config::recommended`` keeps the lean burst library at every size.
The surface portfolio is the plain surface plus the relative diameter penalty, at every size.
``Config::communicating`` keeps an empty surface list at every size.
A closed Wulff shell replaces the liquid start when the lattice has one.
Every other size keeps the liquid start.
A full quench stops at the record gradient.
The iteration cap stays 200.
The record gradient stays 0.001.
The Bayes threshold stays the closed form at a screen of 25 and a quench of 200.
That quotient is 175/375, which is 7/15.
