Fixed the directed stall-leave clock so it advances only when a stall response
actually produces a candidate rather than consuming a full patience window on
no-op exits.
