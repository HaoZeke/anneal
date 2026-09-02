The cooperative catalog speaks Cap'n Proto RPC. A replica attaches
with a subscriber capability, calls through a session bound to its
identity, and receives epoch-close, roster, and retire events. The
existing CatalogRequest/CatalogReply payload, journal, and operation
handlers are unchanged. Observe needs no identity.
