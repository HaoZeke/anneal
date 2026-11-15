`evaluate_batch` raises `TypeError` for a positions or box entry that is not a
C-contiguous float64 array, as the other entry points do. Before, it raised
`RuntimeError` with the message `std::bad_cast`.
