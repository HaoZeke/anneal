CPMDPot now applies selectOrbitals immediately on a live cpmdc session and re-applies the key after session recreation, so select-select-force sequences are unambiguous.
CPMDPot adds copyOrbitals(from_key, to_key) and uses cpmdc_session_copy_orbitals when the engine exports it, while engines without that primitive report unsupported instead of silently succeeding.
