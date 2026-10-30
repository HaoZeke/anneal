``lj_causal_runner`` no longer fails on a checkout the causal runner refuses
to seal. Before it runs the script, the test applies the script's own
``git diff --quiet HEAD --`` check. On a tree whose tracked files differ from
HEAD, or on a directory that is not a git checkout, it prints a warning naming
the reason and returns. On a clean commit it runs as before.
