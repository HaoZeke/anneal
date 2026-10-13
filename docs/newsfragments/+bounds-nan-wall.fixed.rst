``Bounds`` refuses a NaN wall with ``ValueError``, naming the coordinate:
``coordinate i has a NaN wall: low ..., high ...``. It had built such a box,
which contains no point. An infinite wall is still taken.
