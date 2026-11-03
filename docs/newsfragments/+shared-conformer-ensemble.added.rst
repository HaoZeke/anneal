In a thread ensemble, ``Config::ensemble_report`` and ``Config::entropy_stop`` read the shared history, which keeps every minimum any chain certified.
The chains then keep no list of their own.
``EnsembleReport::conformers`` reports the shared ensemble, and ``MinimumHistory::energies`` gives the energy of every minimum in the history.
Under the entropy stop a round is ``Config::entropy_round`` visits to the history by any replica.
Once the shared entropy converges, every replica still running retires with the reason ``entropy_converged``.
A replica that spends its budget first keeps the reason ``budget``.
The summed call count at the stop can differ between two runs of one seed, because the round a replica closes follows the scheduler.
``EnsembleConfig::validate`` refuses the report and the stop without ``HistoryMode::Shared``.
