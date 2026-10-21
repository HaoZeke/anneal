An in-process census cohort dials neighbours only after every publisher
is bound. A dial that landed on a port reservation completed the TCP
handshake there and waited for a protocol header the reservation never
sends. That wait is ten seconds. macOS does not reset the unaccepted
connection when the reservation closes, so a five second connectivity
check expired with replicas missing.
