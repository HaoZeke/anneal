``Config::retry_noop_stall_leave``,
``Config::align_directed_stall_source``,
``Config::capped_lindh_quench``, ``Config::call_floor``,
``Config::call_floor_elements``, ``Config::branch_on_stall`` and
``Config::for_cluster_with_scales`` are public. None of these names
exists in v0.12.0. ``molecular_call_floor`` is a different function
and is unchanged.
``retry_noop_stall_leave`` retries a call-clock leave at once when no
escape candidate exists. Off, an attempted leave still consumes its
plateau window. On, only a leave that runs consumes it.
``align_directed_stall_source`` starts those leaves from the recorded
best. Off keeps the live walker as the activation target.
``capped_lindh_quench`` quenches a Cartesian molecule with capped
Lindh steps. It is off unless set. A non-molecular or rigid-body
library ignores it.
``call_floor`` is an optional positive pair distance. A trial inside
it is opened just past the floor before relaxation, and dropped
uncharged when the opening fails. A zero-step read inside it is
refused uncharged. A floor that is not positive is refused when the
search starts. Declared species without a floor keep
``OVERLAP_CALL_LIMIT``. Stock presets leave it unset.
``call_floor_elements`` is one atomic number per point, read only by
``call_floor``. Each pair floor is ``element_call_floor`` and is never
below ``call_floor``. A count other than the point count is refused.
``branch_on_stall`` spends a stalled stretch on independent descents
from fresh random clusters. Off leaves the chain on its own
trajectory. The walk does not see a branch.
``for_cluster_with_scales(n_points, length_scale, energy_scale)`` is
``for_cluster`` at those scales, including the scaled shell core.
``for_cluster`` calls it with the reduced Lennard-Jones scales.
