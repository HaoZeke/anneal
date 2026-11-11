# LJ104 measure v4.14

The global-minimum count is 0 of 6.
The best energy is -581.80180238.
A hit requires energy at or below -582.086642 plus 1e-6.
None of the six seeds reached that level.

The timed commit is 7a05b51318dad29e03780d981dd3543aa8cd0457.
The timed tree is e2120c08e91f9698e121d0c38e79ce64de8a74e3.
The arm opened at 19:08:07 and closed at 19:24:09.

Each receipt records rustc 1.88.0 (6b00bc388 2025-06-23).
Each receipt records cargo 1.88.0 (873a06493 2025-05-10).
rust-toolchain.toml is absent.
The pin was not added.

The example line cover on rustc 1.88.0 is 98.39 percent.
The example line cover on nightly-2025-06-26 is 98.99 percent.
The example branch cover on that nightly is 100 percent.
The line gap is 0.60 points, three closure lines.
Region cover on that nightly is 97.59 percent.
That is 1.40 points under the line cover, on the same five missed lines.
The library files the example calls are in logs/coverage/summary.txt.
No coverage file is excluded.

The rohit-paper profile was not run.
The program proseguard is not installed on this machine.
The profile sha follows.
654d0de1c2632feb93ddfc2f72a08372ad672a8c7cb4a6ad30cca69b2783ddc6.
Those v4.13 sentences were rewritten before this report.

PROSE_EXTRACT.txt is a raw record.
Its writer copies each prose file and each commit message.
snapper is not applied to that copy.

The sha-map writer rewrites only rows it already has.
Paths that were never rows stay outside the map.
logs/sha-map-gap.txt names each of those paths.

The tree check transcript follows.

```
producer=python3 -B scripts/lj104_tree_check.py
receipt_root=logs/budget
timed=7a05b51318dad29e03780d981dd3543aa8cd0457
timed_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
receipt logs/budget/2000/receipt.log
receipt_head=7a05b51318dad29e03780d981dd3543aa8cd0457
receipt_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
head_match=yes
tree_match=yes
receipt logs/budget/2001/receipt.log
receipt_head=7a05b51318dad29e03780d981dd3543aa8cd0457
receipt_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
head_match=yes
tree_match=yes
receipt logs/budget/2002/receipt.log
receipt_head=7a05b51318dad29e03780d981dd3543aa8cd0457
receipt_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
head_match=yes
tree_match=yes
receipt logs/budget/2003/receipt.log
receipt_head=7a05b51318dad29e03780d981dd3543aa8cd0457
receipt_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
head_match=yes
tree_match=yes
receipt logs/budget/2004/receipt.log
receipt_head=7a05b51318dad29e03780d981dd3543aa8cd0457
receipt_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
head_match=yes
tree_match=yes
receipt logs/budget/2005/receipt.log
receipt_head=7a05b51318dad29e03780d981dd3543aa8cd0457
receipt_tree=e2120c08e91f9698e121d0c38e79ce64de8a74e3
head_match=yes
tree_match=yes
covered_diff_count=0
tree check ok: 6
exit=0
```
