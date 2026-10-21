Binary Lennard-Jones and rigid TIP4P on the stock cluster presets.
``cluster_search`` takes ``species``, ``sigmas``, ``rigid``, ``layout``,
and ``target``. Two or more species labels add one identity exchange.
With diameters the library is a collective displacement, an angular
relocation, and that exchange. At least half of the proposals take the
exchange. A record within
one energy unit of ``target`` is finished by downhill quench-assisted
exchanges. A rigid chart orients every body and keeps the centres
fixed. The acceptance temperature of a rigid chart is scaled inside
the hop. A chart with more than ten bodies keeps that factor at or
below 4. Preset temperature, screen length,
relax length, and record gradient stay on the preset. None of this is a
new parameter, so the C header is unchanged.
