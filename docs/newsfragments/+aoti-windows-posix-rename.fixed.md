On Windows, publishing an AOTI package now replaces a file a reader still has open, as the 3.6.0 notes said it did.
`MoveFileExW` refuses to replace a file that has any open handle, so a reader that kept the package open could make the publish fail with access denied after its two-second deadline.
The replace now falls back to POSIX rename semantics (Windows 10 1607 or later, NTFS), and the reader keeps the complete package it opened.
A read, reservation or publish that gives up reports the error code, the attempts and the elapsed time.
