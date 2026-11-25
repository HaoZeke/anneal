# ASE panel rerun with the per-geometry call count

Your prompt names SYSTEM (lj38, lj55, lj98, lj104 or au55). Run every
seed 2000-2011 of that system, serially, in this one session.

1. `git fetch origin bench/g2-shard-harness-fix2` then
   `git checkout -b panel 8ebcf574b34870ed74db5a3265f18902ba6a4992`.
   `git rev-parse HEAD^{tree}` must be
   `01162549e910cc8d34188719438c92e3b233a6a5`. HEAD is BASE; set
   `REV=8ebcf574b34870ed74db5a3265f18902ba6a4992`. Stop and report on
   any mismatch. Make no code edits.
2. Build the panel binaries from this tree:
   `sudo mkdir -p /work && sudo chown $(id -un) /work`, then
   `CARGO_TARGET_DIR=/work/target-panel cargo build --release --example ase_panel --example asg_eval`.
   Install unmodified ASE 3.29. No GMIN.
3. The caps are the wall caps stored on the earlier panel rows, in s:
   lj38 4.653972, lj55 4.424556, lj98 246.546693, lj104 280.907064,
   au55 112.374012 and 122.695098 (au55 runs both caps as two
   campaigns). Budgets are the defaults in `run_one.py`. Do not change
   either.
4. For each seed, run three arms in this order: even seeds ase_bh,
   ase_mh, anneal; odd seeds anneal, ase_mh, ase_bh. ase_bh and ase_mh
   get `CAP` set to the cap above; anneal gets `CAP=` (empty) and is the
   same-session host-speed control. Each run is one foreground command:

       mkdir -p /work/ase-panel/runs /work/ase-panel/home /work/ase-panel/numba
       sudo -n unshare --uts sh -c "hostname bench; exec sudo -u $(id -un) env \
         HOME=/work/ase-panel/home NUMBA_CACHE_DIR=/work/ase-panel/numba \
         PYTHONPATH=$(python3 -c 'import site; print(site.getusersitepackages())') \
         OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
         VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 PYTHONUNBUFFERED=1 \
         ASE_PANEL_WORK=/work/ase-panel/runs \
         ASE_PANEL=/work/target-panel/release/examples/ase_panel \
         ASG_EVAL=/work/target-panel/release/examples/asg_eval \
         ASE_PANEL_WALL_CAP=$CAP ASE_PANEL_REVISION=$REV \
         taskset -c 0 python3 $PWD/scripts/ase_panel/run_one.py SYSTEM ARM SEED" \
         > out.txt 2> err.txt

   Append the one line of out.txt that starts with `SYSTEM,` to
   `bench/ase-SYSTEM/rows-ljfix.csv` (au55: `rows-ljfix-cap112.csv` and
   `rows-ljfix-cap122.csv`), creating it with the header
   `system,arm,seed,budget,hit,evals_to_hit,charged,best,wall_s,inside_s,outside_s,code_revision,host,stop,cap_s`.
   Keep the `shape_json=` lines in `bench/ase-SYSTEM/shapes-ljfix.jsonl`.
   Never drop, edit or rerun a row. A capped or budget-stopped run is a
   miss. A failed run is recorded as a failure with its stderr tail in
   your report, not retried.
5. Hold the turn for the whole session with foreground commands. Never
   end your turn with runs going: the machine pauses when your run ends.
6. Commit the new row and shape files as one commit (append-only). Then
   `git format-patch BASE..HEAD --stdout > panel.mbox` (not committed).
7. Final message, always: per arm the hits out of 12, median charged
   calls, median wall, and the anneal control's median wall; the CPU
   model and core; then `xz -9e < panel.mbox | base64 -w0` between
   BEGIN-PANEL-B64 and END-PANEL-B64, with the sha256 of panel.mbox,
   HEAD and tree. Keep the message short apart from the b64.
