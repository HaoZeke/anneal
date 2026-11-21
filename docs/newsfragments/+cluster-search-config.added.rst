``cluster_search`` takes ``config=``, a ``Config``, and runs the search it
describes, so every setting reaches the search and not only the few that
have a keyword. A config of the wrong size, or one given beside a keyword
that it already decides, raises ``InvalidArgumentError``.
