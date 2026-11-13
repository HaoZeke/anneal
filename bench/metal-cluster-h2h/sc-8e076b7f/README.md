# Sutton–Chen, anneal against GMIN

The minima are the Cambridge Cluster Database tables for the Sutton–Chen 12-6, 10-8 and 9-6 families, at 38, 55 and 80 atoms. Doye and Wales, *New J. Chem.* **22**, 733 (1998). The point files are

```text
https://doye.chem.ox.ac.uk/jon/structures/SC/12-6/table.12-6.html
https://doye.chem.ox.ac.uk/jon/structures/SC/10-8/table.10-8.html
https://doye.chem.ox.ac.uk/jon/structures/SC/9-6/table.9-6.html
```

12-6 models silver and rhodium, 10-8 gold and platinum, and 9-6 nickel and copper. The database lists every size through 80. This panel is the three sizes above. Energies are the reduced units on those tables, ε = 1 and a = 1. The c, n and m values are the first line of the point files: 12-6 is c = 144.41, 10-8 is c = 34.408, 9-6 is c = 39.432.

The energy and the forces are rgpot's `rgpot_sutton_chen_energy_forces`, with no cutoff. GMIN's `SC` keyword is the same reduced row. A deposited 10-8 38-atom geometry is −10117.245407 in both. A hit is within 0.001 of the printed table.

`PAPER_BUDGETS` for 38 and 55 atoms are the LJ38 and LJ55 entries of `scripts/lj_board/boardlib.py`, 400000 and 1000000. 80 atoms is not in that map. It uses the 55-atom entry, 1000000. The LJ dict itself is unchanged. The search is `Config::recommended`, whose basin stride is 1. Lennard-Jones and TIP4P do not go through this example.

Six seeds, 2000–2005. The (system, seed, arm) list is shuffled with schedule seed 20261115. One job at a time, `taskset -c 0`, `sudo unshare --uts`, hostname `bench`, working directory `/work`. `pot_per_eval_s` is potential time per charged call. `outside_per_eval_s` is the rest of the wall on the same denominator. The wall ratio is the GMIN median wall divided by the anneal median wall.

```text
snakemake -s workflow/Snakefile sc_h2h
```
