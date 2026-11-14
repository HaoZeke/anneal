`potserv` on the Morse backend applies `configure(PotentialConfig.morse)`, rebuilding the pot with the sent `de`, `a`, `re` and `cutoff` and keeping its switch width; it used to refuse every configure call that was not for NWChem or CPMD.
A value it cannot use, such as a cutoff below the switch width, gets `ok=false` and leaves the pot unchanged.
On every backend `none` is a no-op, and any other arm the backend does not apply, including one from a newer schema, gets `ok=false` with a message that names the arm.
