# G2 fair panel: one ASE minima-hopping seed

Your prompt names CELL and SEED. Run that one seed and nothing else.

1. `git fetch origin bench/g2-shard-harness-fix2` then
   `git checkout -b shard 8ebcf574b34870ed74db5a3265f18902ba6a4992`.
   `git rev-parse HEAD^{tree}` must be
   `01162549e910cc8d34188719438c92e3b233a6a5`. HEAD is BASE. Stop and
   report on any mismatch. Read `scripts/ase_nonlj/cpu_cap_spec.py` and
   `run_cpu_cap_seed.sh` to see which binary or env the cell needs, and
   build it in release mode from this tree if needed. Make no code edits.
2. Run serial and pinned, as ONE foreground command that holds your turn
   for the whole seed:
   `taskset -c 0 env LJOS_SEAT=benchbot bash scripts/ase_nonlj/run_cpu_cap_seed.sh CELL SEED`.
   If the shell tool caps the command, nohup the run and keep the turn
   alive with a foreground loop such as
   `while kill -0 $PID; do sleep 600; ps -o cputime= -p $PID; done`.
   Never end your turn mid-seed: the machine pauses when your run ends.
   The script enforces the pre-registered 12 h CPU cap (43200 s). A
   capped seed is a miss at the cap with best E, calls and CPU recorded.
   Never drop it, never call it infeasible, never estimate. Unmodified
   ASE 3.29. No GMIN.
3. Commit the appended rows as one commit (append-only: `git diff` shows
   no deleted row lines). Then
   `git format-patch BASE..HEAD --stdout > shard.mbox` in the repo root
   (not committed).
4. Final message, always: the appended row lines verbatim, calls, best
   energy, hit, CPU s, wall s, CPU/wall, core, machine shape; then
   `xz -9e < shard.mbox | base64 -w0` between BEGIN-SHARD-B64 and
   END-SHARD-B64, with the sha256 of shard.mbox, HEAD and tree. Keep the
   message short apart from the b64.
