Test binaries link libpython when the python feature is on.
pyo3's extension-module feature leaves the Python symbols undefined.
Linux then rejects the link, and macOS reaches process start without those symbols.
The wheel build still leaves the symbols for the interpreter that imports the extension.
