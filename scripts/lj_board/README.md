# LJ55 driver

The AGOX arm is `python3 scripts/lj_board/run_arm.py agox 55 <seed>`.
The anneal arm is `lj55_board_arm <seed> <budget>`.

AGOX version 3.11.1.
Wheel `agox-3.11.1-cp312-cp312-linux_x86_64.whl`.
sha256 `7f6a4939d981e1826c8ca4243279d9271c70313e1785b7873d1ae91f1ce8ab47`.

## LJ98

The current arm is `python3 scripts/lj_board/run_arm.py agox 98 <seed>`.
The documented arm is `python3 scripts/lj_board/run_arm.py agox-documented 98 <seed>`.
Both arms pin AGOX 3.11.1, source `0a5136605161aaff2610584cd93ae6e9564fda46`,
wheel sha256 `f8b4f49b0e5a053c1b69205633c8b19523cd188b4001a12f093aaaa7f6250457`.
The documented arm keeps the stock box-force adjustment.
Seeds 6400-6405, budget 4,000,000, host bench, reference -543.665361.
Current AGOX hit 0 of 6, median wall 7031.920068 s.
Documented AGOX hit 0 of 6, median wall 8399.304438 s.
Anneal at 30a4ea8801236aba5175eea07b07e2424a056bd7 hit 2 of 6, median wall 81.334000 s.
Type-7 median wall ratio, current AGOX over anneal: 87.498181.
Type-7 median wall ratio, anneal over current AGOX: 0.011434.
Type-7 median wall ratio, documented AGOX over anneal: 103.653286.
Type-7 median wall ratio, anneal over documented AGOX: 0.009648.
The rows are in `bench/agox-lj98`.

## LJ75

The current arm is `python3 scripts/lj_board/run_arm.py agox 75 <seed>`.
The documented arm is `python3 scripts/lj_board/run_arm.py agox-documented 75 <seed>`.
The anneal arm is `h2h_arm3 lj75 default 4000000 <seed>`.
Seeds 6400-6405, budget 4,000,000, host bench, reference -397.492331.
The rows below were measured in one fresh session. No earlier log was copied in.
Base of this record: develop 98dcfb3106f191baf7381fecd5624426831094fe.
Anneal code_revision ccc22cee607b18463b3891dd2f9048810d671ac0.
AGOX 3.11.1 source 0a5136605161aaff2610584cd93ae6e9564fda46.
The wheel that produced the rows is sha256
5ecb3640fa1701c56007ed267573a8c62fdc3c6c69e15c77428b62516aad3eff.
The tree pin f8b4f49b0e5a053c1b69205633c8b19523cd188b4001a12f093aaaa7f6250457
was not the file on that machine.

| arm | hits | median wall s | median best | gap to -397.492331 | AGOX/anneal | anneal/AGOX |
| --- | --- | --- | --- | --- | --- | --- |
| current AGOX | 0/6 | 5655.108823 | -392.980553 | 4.511777 | 103.663412 | 0.009647 |
| documented AGOX | 0/6 | 14493.759020 | -372.106043 | 25.386287 | 263.511422 | 0.003795 |
| anneal | 0/6 | 55.299841 | -396.282249 | 1.210082 | 1 | 1 |

The rows are in `bench/agox-lj75`.
