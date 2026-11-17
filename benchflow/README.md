# The benchmark suite

This suite times the Python force calls of rgpot.
It reports a slowdown between two revisions.
Timings come from airspeed velocity (asv).
Memory comes from memray and its `asv_bench_memray` plugin (`ray_*` methods).
Snakemake is the only driver.
Tool versions are pinned in `benchflow/requirements.txt`.
`asv.conf.json` repeats the same versions.
The suite uses three commands:

```sh
python -m pip install -r benchflow/requirements.txt
snakemake -c 1
python -m unittest discover -s benchflow
```

## Noise calibration

This section explains how the suite turns measured noise into an upper bound.
`benchflow/config.yaml` sets `rounds` (10) and `runs` (10).
Those runs of `develop` on one process measure the noise.
Each run uses `--interleave-rounds` without `--quick` or `--skip-existing`.
Two runs of the same commit give a calibration ratio.
That ratio is the median of the second run over the median of the first.
The relative threshold uses the median absolute deviation (MAD) of those ratios.
The threshold is the largest of these four values:

```text
max(0.05, 3 * MAD, the 99th percentile of |ratio - 1|, the maximum |ratio - 1|)
```

Each benchmark's upper bound is one plus that threshold.
So the upper bound is at least the widest calibration ratio.
It is also at least five percent above one.

## Run output and the machine record

This section explains what those timing runs leave behind.
Each run writes asv results under `benchflow/out/results/<run>`.
The exporter takes the result for the run's commit, machine, and environment.
It fails when no result or more than one result matches.

For the machine, every run uses the name `bench-runner`.
Its `os`, `arch`, `cpu`, `num_cpu`, and `ram` fields have fixed generic values.
That record is in `benchflow/out/home/.asv-machine.json`, the asv machine file.
The machine check reads only the `bench-runner` record.
It ignores every other record in that file.

## Revision comparison

This section explains how the suite compares revisions and reports a slowdown.
`snakemake -c 1 pr_compare` compares two distinct revisions, `base` and `head`.
It runs one `asv continuous --interleave-rounds` for them.
It writes the table to `benchflow/out/pr_compare.txt`.
A table row fails when its head/base median ratio is above its own upper bound.
The result goes to `benchflow/out/pr_compare.status`.
`pr_compare` itself exits zero.
Equal revisions do not run a comparison, and their status is non-zero.
`pr_gate` exits with that status, so it fails on a slowdown.
Continuous integration should run `pr_gate`.

## Warm calls

This section explains why warmup does not change what each sample measures.
`warmup_time` drops calls before the first sample.
Those warm calls leave the timed geometry alone.
Each `evaluate_lj` call builds a new calculator.
It does not write the coordinate arrays.
`LJPot` is one object, and setup evaluates it once to fill the pair list.
The timed calls then repeat that same geometry.
The arrays from setup are not replaced.

## The wheel

This section describes how the wheel is built.
The wheel is the installed package the timings call.
The force classes call into that wheel's `librgpot`.
Those classes are `LJClusterPot`, `MorsePot`, `WaterHPot`, and `LJPot`.
`benchflow/build_wheel.py` passes `-Dbuildtype=release`.
It enables the Fortran kernels with `-Dwith_fortran_pots=enabled`.
Dispersion corrections and the metatomic engine stay off.
Meson's default buildtype is debug.
A configure without the release flag would time a different build.
The build keeps its meson tree.
It writes the output of `meson introspect --buildoptions` beside the wheel.

The wheel build also needs a few tools in place.
`patchelf` must be on `PATH`, because meson-python calls it.
It rewrites the run-time search path (RPATH).
`CC` and `CXX` are pinned to `gcc` and `g++`.
The build needs Python's development headers and a Fortran compiler.
