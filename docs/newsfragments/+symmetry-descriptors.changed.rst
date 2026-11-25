``MotifCounts`` is ``LocalPairCounts`` and ``MotifClass`` is
``DescriptorClass``. ``motif_counts`` and ``motif_class`` are
``local_pair_counts`` and ``descriptor_class``. The count fields
``five_fold``, ``fcc`` and ``hcp`` are ``ring_like``,
``branch_disjoint`` and ``branch_shared``.
``SPARSE_FIVE_FOLD_PER_ATOM`` and ``DENSE_FIVE_FOLD_PER_ATOM`` are
``SPARSE_DESCRIPTOR_PER_ATOM`` and ``DENSE_DESCRIPTOR_PER_ATOM``.

``symmetrise::generate_group`` returns
``Result<Vec<Rot>, GroupCapExceeded>``. A set that would pass ``cap``
before it closes is ``Err(GroupCapExceeded)``, whose ``elements`` are the
operations collected up to ``cap`` and whose ``cap`` is the cap that
stopped it. A caller that kept the truncated set takes ``elements``.
``symmetrise::detect`` and ``symmetrise::detect_all`` skip a turn that
carries no atom onto another atom. ``symmetrise::best_order_about`` and
``symmetrise::detect_orders`` are new.

``Config::detected_symmetry``, set by ``Config::with_detected_symmetry``,
is off by default and is in the resolved configuration only when on.
On, detection scans every order the point count can hold instead of
``[2, 3, 4, 5, 6]``, symmetry groups must close, tolerantly if the
operations are approximate, and the continuous-symmetry measure projects
onto the operations detected on the incumbent. Off, the orbit
completions of every ``OrbitCore``, the symmetry moves and the
continuous-symmetry measure are as before, bit for bit.
``Activation::detected_symmetry`` likewise turns on the force-residual
and dominant-axis-order openings, and ``ClusterMove::OrbitComplete`` and
``ClusterMove::PointSymmetrise`` carry a ``detected_symmetry`` field.

``csm::propose`` projects onto ``D_5h``, ``C_5`` and ``C_i`` as before.
``csm::propose_for`` takes the flag: on, it projects in the input frame
onto the detected operations that carry every atom onto an atom of its
own class, and its ``GroupProjection::group`` is ``"detected"``.
