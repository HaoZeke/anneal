# LAND

- Target repo: HaoZeke/anneal
- Kind: record kit (data plus rescore script), no PR named; destination for Inky to choose (record/ branch or fold into the GFN2 work)
- Base: 61fa67cf3 (develop)
- Payload: 8 commits, HEAD 252cda72afd47299fff130ceab511c33ebafe322, tree b5c1c2ec7c2d044a3337abd7b9762e850a206c28
- First 2 commits are the fix/gfn2-linesearch kit (tree cf1100449f536eba7de300fbad1c7d38e89af557)
- Builder family: told to Inky by message
- Verdict: the recorded Si13 ring-target result (anneal 12/12, CREST 0/12) is VOID.
  Standalone-xtb rescore of the kept final geometries: anneal 0/12 (10 SCC
  failures, min Si-Si 1.55-1.81 A; 2 converged at -595.2 and -591.5 eV, about
  50 eV above the recorded values), CREST 0/12 (9 converged at -597.036 eV, a
  constant 2.309 eV above the recorded -599.345; 3 seeds never launched after a
  preopt SCC failure). Fisher 1.0. The rescore rule was set after the data.
- Rank test (0.028) only reflects arm-specific miss budgets and means nothing.
