The ``conformer_search`` example searches the conformers of a flexible molecule at a fixed call budget.
Its chains divide the budget and can share one minimum history.
The default is one chain with preconditioned quenches, and a tenth of the budget is kept back to settle the best minimum.
The validated minima go to a readcon-db corpus.
