Periodic cutoff lists keep every linkcell image inside the cutoff, after
dividing coordinates and the cell by the descriptor length scale. Where one
direction is not periodic, the list keeps one shortest image and does not
wrap that direction. Free-cluster shell counts use that unwrapped cutoff.
