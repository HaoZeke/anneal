Copyright (c) anneal developers

# Si13 ring-target rerun

The rescore rule was set after the campaign data were recorded. Each retained
final geometry is evaluated cold with standalone GFN2-xTB at its default
accuracy and at tight accuracy 0.01; a run is a false hit if either SCC does
not converge or either standalone energy differs from the recorded best by
more than 0.01 eV. The tight standalone energy determines the corrected hit
against the original target and 0.001 eV tolerance. A run without a retained
geometry is not reconstructed and is a false hit and a corrected miss.
Corrected rank tests use the original evaluations to hit for a corrected hit
and budget plus one for a miss. The 2.8 A coordination cutoff and structural
diagnostics do not affect the classification.
