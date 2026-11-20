The release workflow contract tests parse ``release.yml`` as YAML. They check
the step order, that a step has no ``if`` or ``continue-on-error``, the exact
``cargo publish`` command and the version it publishes, and the wheel matrix.
A step that moves, is skipped or swallows its exit status now fails them.
