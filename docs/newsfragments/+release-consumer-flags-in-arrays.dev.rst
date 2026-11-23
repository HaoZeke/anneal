The release consumer check reads the ``pkg-config`` flags for the archive into
bash arrays and passes them quoted, so shellcheck reports no SC2046 and the
flags are not globbed. A failing ``pkg-config`` now fails the step.
