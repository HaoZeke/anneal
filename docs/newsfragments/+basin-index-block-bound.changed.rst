``BasinIndex::lookup`` skips a stored centre when the block sums of the
two descriptors already put it past the merge radius, before reading
the descriptor. The bound allows for the rounding of the sums, so every
lookup returns the basin the plain walk returns. It applies to metrics
whose new ``BasinMetric::is_euclidean`` is true, which
``EuclideanMetric`` is.
