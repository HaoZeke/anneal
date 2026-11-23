# LAND: anneal #211 r-next2 bit-identity gate

target: HaoZeke/anneal PR #211
lease: 86821e1b879e89f635ecc3a23dd42829a4277a85
payload: 0070ca5707cdff624d6d83e3afca6786fd97a2f0
payload tree: b57a30db80d41fddce42b4509bc40d9004261779
develop base: 5abb4dedec6cd2e95e7b17472639529fd139b278
measurer: cloud-agent family
builder: told to Inky by message
This kit adds bench/ rows and print-only example lines;
the payload is not modified.

Verdict: PASS (B bit-identical to A, 12/12 rows per cell)

| Cell | Ident | Hits A/B | Fisher | B/A outside [95% CI] |
| --- | --- | --- | --- | --- |
| tip4p9 | 12/12 | 12/12 | 1 | 1.0077 [0.983, 1.074] |
| tip4p10 | 12/12 | 12/12 | 1 | 1.0090 [0.997, 1.028] |
| tip4p14 | 12/12 | 12/12 | 1 | 1.0015 [0.997, 1.012] |
| tip4p16 | 12/12 | 2/2 | 0.705 | 1.0027 [0.983, 1.017] |
| tip4p17 | 12/12 | 5/5 | 0.660 | 1.0061 [1.001, 1.015] |
| lj38 | 12/12 | 12/12 | 1 | 1.0088 [0.978, 1.032] |
| lj75 | 12/12 | 5/5 | 0.660 | 1.0215 [0.931, 1.086] |
| m55r6 | 12/12 | 12/12 | 1 | 1.0057 [0.974, 1.039] |

unconverged_records identical per row. Max upper 1.0862.
