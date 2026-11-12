# GFN2 (H2O)20, anneal at a CREST-matched wall

Anneal at 28000 charged calls against the CREST rows already on this VM. CREST was not rerun. The wrapper is `scripts/h2o20_goat_campaign.sh` with `H2O_SKIP_CREST=1`: `sudo unshare --uts`, hostname `bench`, working directory `/work`, one thread. Seeds 15001–15006, arm order shuffled per seed. A hit is a vtight GFN2 energy within 0.01 eV of the reference, −2768.551087 eV. xtb is 6.7.1 (edcfbbe). GOAT is absent because ORCA 6 is login-gated.

The anneal mean wall is 1016.3 s. The CREST mean wall is 974.3 s. The ratio of those means is 1.04. Anneal is lower on seed 15006 only. No seed hits the reference.

| Seed | Anneal eV | Anneal wall (s) | CREST eV | CREST wall (s) | Hit |
| --- | --- | --- | --- | --- | --- |
| 15001 | −2767.991098 | 1130.635 | −2768.212431 | 595.009 | no |
| 15002 | −2765.669528 | 980.861 | −2766.858968 | 818.437 | no |
| 15003 | −2766.589248 | 958.514 | −2767.528621 | 1354.714 | no |
| 15004 | −2767.756252 | 936.122 | −2768.407371 | 1317.922 | no |
| 15005 | −2767.711161 | 949.654 | −2768.479940 | 1046.707 | no |
| 15006 | −2768.271766 | 1143.136 | −2766.250279 | 712.840 | no |
