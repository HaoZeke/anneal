The release packaging step lists each native library archive once to a file
and checks every entry with ``grep -Fxq``. ``tar tzf | grep -Fq`` failed with
status 141 under ``pipefail`` when grep stopped reading early and tar took
SIGPIPE, and ``-F`` without ``-x`` let ``anneal-core.hpp`` stand in for
``anneal-core.h``. The step now runs under bash with ``pipefail``.
