The Python 3.10 CI cells install ``tomli`` before ``cargo test``. The release
contract tests run ``scripts/native_pkgconfig.py`` with the matrix
interpreter, and on 3.10 it found neither ``tomllib`` nor ``tomli``.
