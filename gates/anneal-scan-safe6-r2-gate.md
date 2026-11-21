# Scan-safe6 r2 gate

## Pre-registration

This section was written before any gate build or run.

### Inputs

- The base arm is commit `61fa67cf`.
- The candidate arm is the verified candidate tree
  `68b35a35c13fd4c5ce683594c744f3b5cb2cad8d`.
- A second, independent base build is the noise-control arm.
- Seeds are exactly 2000, 2001, and 2002.
- The cells are every system and driver in the paper panel runner:
  `lj38`, `lj55`, `lj75`, `lj98`, `lj102`, `lj103`, `lj104`,
  `lj150`, `au55`, `m80r6`, `m80r14`, and the second `lj75`
  driver.

### Execution

- Every arm is a release build from its recorded commit and tree.
- The two base builds use separate build directories.
- Runs are serial and pinned to core 0.
- All supported thread-count environment variables are set to one.
- No shortened mode is used.
- All arms use the same generated start and seed within a cell.
- The candidate comparison runs in A-B-B-A order.
- The noise comparison runs in A-C-C-A order, where C is the second
  base build.
- Wall time is measured around the search. Potential time is measured
  inside charged potential evaluations.

### Gate 1: bit identity

For every cell and seed, the candidate must match the base in:

- final-energy bits;
- final-structure hash;
- charged call count; and
- the complete accepted-trajectory digest.

The trajectory digest covers every accepted transition in order,
including its step, action, energy bits, flags, and destination-state
bits. Improvement records retain their charged-call positions. Any
divergence is reported with the first differing step or call and the
values from both arms.

Gate 1 passes only when every candidate row is bit-identical to its
paired base row. Timing differences do not affect Gate 1.

### Gate 2: outside time

For each cell, outside time per call is:

`(wall time - potential time) / charged calls`.

The cell statistic is the median over all rounds and the three seeds.
The candidate-to-base ratio must be at most 1.10 in every cell. The
separate-base-to-base ratio is reported alongside it as the observed
noise band. Gate 2 passes only when every cell meets the 1.10 limit.

Failed runs, missing rows, unequal seed sets, non-positive call counts,
or negative unrounded outside times fail the relevant gate. Rows are
append-only during execution; summaries are derived without selecting,
retuning, relabelling, or replacing observations.

## Results

- Gate 1, bit identity: **FAIL**
- Gate 2, outside time per call: **FAIL**
- Overall verdict: **FAIL**

### Gate 1 per-cell results

| Cell | Driver | Energy | Structure | Calls | Trajectory | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| lj38 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj55 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj75 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj98 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj102 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj103 | h2h_gmin_proof+lj103 | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj104 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj150 | h2h_gmin_proof | FAIL | FAIL | FAIL | FAIL | FAIL |
| au55 | h2h_metal | FAIL | FAIL | FAIL | FAIL | FAIL |
| m80r6 | h2h_morse | FAIL | FAIL | FAIL | FAIL | FAIL |
| m80r14 | h2h_morse | FAIL | FAIL | FAIL | FAIL | FAIL |
| lj75-seed | h2h_seed | FAIL | FAIL | FAIL | FAIL | FAIL |

### First divergences

- lj38 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c06524c046bbb1d9', 'hash': '4d7dddf4214b5d88e4b28579dd357540579553bb51aa6f00fd4d2b8386921e54', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0651eae7156cf66', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c06524c046bbb1d9', 'hash': '977c911e3488eba0743ccaa57e9628d64277b33132214acb0786d518f5995987', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c06524c0305410ae', 'validated': False}
- lj38 seed 2001: transition step 9; A={'action': 'sym', 'adopted': True, 'from_bits': 'c065272f1d48cb74', 'hash': '48e1d906fade97c0d4081cf14d9c9c94c6bf0c2a06d9abd1256e313e57d7224a', 'index': 7, 'kind': 'transition', 'step': 9, 'to_bits': 'c0651433b66bbd75', 'validated': False}; B={'action': 'continuous-symmetry-csm', 'adopted': False, 'from_bits': 'c065272f1d48cb74', 'hash': 'aea9ab378b05ae83468a3d13232b330566d9193117e5551660bfd00445dc8f44', 'index': 7, 'kind': 'transition', 'step': 9, 'to_bits': 'c065272f1d48cb73', 'validated': True}
- lj38 seed 2002: transition step 3; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c0646c48df2bbbf7', 'hash': '749d35d8a7a4d5a6a062ec0ed5a0bb43398a6125e866b42a03f81395279a982e', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c064be18836943da', 'validated': True}; B={'action': 'surface', 'adopted': True, 'from_bits': 'c0646c48df2bbbf7', 'hash': 'a5e6545e38ecc6ddc379e126a44a8f1eeb1db658c35b928b10c572ec1fc1ea82', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c064aeedf19aae5e', 'validated': True}
- lj55 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c070926afbbe83d8', 'hash': '76ffe6932d0255a7cdb981f28b5c3dd91c035fe39add692c06fc14f52f9d20af', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0708b799e5fb226', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c070926afbbe83d8', 'hash': 'a8a1a5d033cfe6b50fd9216425746670d18bf6b8e4777c093329eb7aaa7fb956', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c070926afb86f6fc', 'validated': False}
- lj55 seed 2001: transition step 6; A={'action': 'sym', 'adopted': True, 'from_bits': 'c070a61f39be20cf', 'hash': '663aee6f17e66b6467981df674028fd69972c1999f770e9ca2acc2c10fda71e6', 'index': 4, 'kind': 'transition', 'step': 6, 'to_bits': 'c070a21a9b678f98', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c070a61f39be20cf', 'hash': 'a45de848106d705b0c5302af3f9695f45d6ee830c061db169c59840faab77a4a', 'index': 4, 'kind': 'transition', 'step': 6, 'to_bits': 'c070a61f396ad281', 'validated': False}
- lj55 seed 2002: transition step 2; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c0702d7b2c9cf7ed', 'hash': '03ecf6d7e37cad88fdca3db4c62f1cab68a502889474b25b44b708624d195b91', 'index': 1, 'kind': 'transition', 'step': 2, 'to_bits': 'c07063cb258c70bb', 'validated': True}; B={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c0702d7b2c9cf7ed', 'hash': '4dd15e06ef25b9dea75b1320740acd3507717ee75e82b6fbf54788a928fcf3c7', 'index': 1, 'kind': 'transition', 'step': 2, 'to_bits': 'c0709c13add704ea', 'validated': True}
- lj75 seed 2000: transition step 2; A={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c077df2aed3993df', 'hash': '19d3d8244d70885c387e2c9ee7648a93775c52fa1cd8befd1d31f7ab83007bf8', 'index': 0, 'kind': 'transition', 'step': 2, 'to_bits': 'c077fa247c00c5ac', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c077df2aed3993df', 'hash': 'ba070647151998698815d0a1ea09dc5778a344c1aeaecf7353e8bbdc07a29914', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c077df2aeab05748', 'validated': False}
- lj75 seed 2001: transition step 9; A={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c077daba8b28d13b', 'hash': '335b7947d6bc31eb5eac900f2f8f64c048f8fd7bca2c817b7ad44025d93d857b', 'index': 4, 'kind': 'transition', 'step': 9, 'to_bits': 'c077eb5b89c9b26b', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c077daba8b28d13b', 'hash': 'b835b442495699ce38e6edb8afa3412fe1f2cd9aaa1f86d995e71466e7735c4c', 'index': 4, 'kind': 'transition', 'step': 7, 'to_bits': 'c077db9a7838cdac', 'validated': False}
- lj75 seed 2002: transition step 9; A={'action': 'continuous-symmetry-csm', 'adopted': False, 'from_bits': 'c077f6572afb82e4', 'hash': '804a0dcc7936e3f00149d346666edb08cc3d36bc1b2bd79ba20217276e88ece7', 'index': 5, 'kind': 'transition', 'step': 9, 'to_bits': 'c077f6572afb82b5', 'validated': True}; B={'action': 'continuous-symmetry-csm', 'adopted': False, 'from_bits': 'c077f6572afb82e4', 'hash': 'b712ff1fec1361fc4bb86b1ccf74258f0c107653387011752e74b858d3fed55a', 'index': 5, 'kind': 'transition', 'step': 9, 'to_bits': 'c077f6572afb82ec', 'validated': True}
- lj98 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c0801759cbe58f61', 'hash': '126c8c7133c2e7e676b6b7c5f4f1b0c575e5c7007fad199725ae8367bdf8ea64', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0802410bc3d7c35', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c0801759cbe58f61', 'hash': 'a77fc7ead5f85d06ac4de2645eba03e6fb41fdbb82dba9456eb5e5b4ec3f8efd', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0801759ca22f910', 'validated': False}
- lj98 seed 2001: transition step 8; A={'action': 'sym', 'adopted': True, 'from_bits': 'c08056ee71e3702e', 'hash': '604ea48472c22df9f690e0f69a9aee353b85a839fc8bdc33cd26697c5021489e', 'index': 7, 'kind': 'transition', 'step': 8, 'to_bits': 'c08056ee71e3701d', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c08056ee71e3702e', 'hash': '7f2aab9ab952213d4a42462491dcc2d73b20a66ec8d3146686a8933aa42e3002', 'index': 7, 'kind': 'transition', 'step': 8, 'to_bits': 'c0804b02833b6eac', 'validated': True}
- lj98 seed 2002: transition step 3; A={'action': 'sym', 'adopted': True, 'from_bits': 'c08013ae13ca5bd8', 'hash': 'bf7f17282178112378e14d67ddf5e6b5468196bf7ee6b1b571bcfacb870de9b3', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c0801573349013a9', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c08013ae13ca5bd8', 'hash': 'c16999ab581faada75d4e20ec0153d2f41d01be6e8fe7901554ca18bce2946bb', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c080153b5e009b4b', 'validated': False}
- lj102 seed 2000: transition step 2; A={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c080f237a7b07020', 'hash': '41689b2451b70f7903dc1a9d53c01f07921e87815a518cc6ecef1f6f292a87be', 'index': 0, 'kind': 'transition', 'step': 2, 'to_bits': 'c08106b15aed3857', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c080f237a7b07020', 'hash': '2b30e1c541e4d76d53bb0c81b8e8e50ec708ccd771fca4a249986dbd9e60fd97', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c080f237a60530e9', 'validated': False}
- lj102 seed 2001: transition step 3; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c080f291a6500dda', 'hash': '66bc10ac9456e85f64072464000e92dbd32698a133ed7c8f85ef0f56f2fea8dd', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c08134058f1e50f2', 'validated': True}; B={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c080f291a6500dda', 'hash': '60ea1fb5258ebbc8b0b0aedcdcbd3f3119fc2996913237c9d1a63635fbf77861', 'index': 2, 'kind': 'transition', 'step': 4, 'to_bits': 'c08104089d8c63ad', 'validated': False}
- lj102 seed 2002: transition step 4; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c080f79e31198d95', 'hash': 'c785e4573591b4e6154e2d274f4e9f778740db03bb0a617028834ba057c87675', 'index': 3, 'kind': 'transition', 'step': 4, 'to_bits': 'c08110701f3f833f', 'validated': True}; B={'action': 'single', 'adopted': True, 'from_bits': 'c080f79e31198d95', 'hash': 'a69b42754bcbff0e90f73e238b7ac52afb8e09441730cb8f442166fc6b141a35', 'index': 3, 'kind': 'transition', 'step': 5, 'to_bits': 'c080f72906ec4cc7', 'validated': False}
- lj103 seed 2000: transition step 2; A={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c080eb1222cb64b4', 'hash': '58f8eb9fa87e77fe4b942fe8b2bfd63ebae37686935174fd9f0f142ea3501df9', 'index': 0, 'kind': 'transition', 'step': 2, 'to_bits': 'c08102a0f17d83f6', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c080eb1222cb64b4', 'hash': '2d76404491df586f9093d254b45e4b94887d45d2e2566a5cf01eda6cd4b7e225', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c080eb122186f812', 'validated': False}
- lj103 seed 2001: transition step 2; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c081058c2d8657a4', 'hash': '03bdf96503a6deb8929dff143257d69209276a304c29243fd096070c8487ccd5', 'index': 1, 'kind': 'transition', 'step': 2, 'to_bits': 'c08156307625908f', 'validated': True}; B={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c081058c2d8657a4', 'hash': 'a8e7f9aea88ccbb7fe11d01b68f82faeb327d54b8004516a824742ee53f216d4', 'index': 1, 'kind': 'transition', 'step': 2, 'to_bits': 'c08141dfb99bc271', 'validated': True}
- lj103 seed 2002: transition step 9; A={'action': 'continuous-symmetry-csm', 'adopted': True, 'from_bits': 'c0814a3f1c2c3e58', 'hash': '51e8388e640725a6c8ed37fa036cb2e93f7a9297bdcbce8cc51e3c83558a356d', 'index': 7, 'kind': 'transition', 'step': 9, 'to_bits': 'c0814a82775d0c65', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c0814a3f1c2c3e58', 'hash': 'cd1ee9b977e7cb63148aebf53ef1e12b4d0ac3465749a08d93aebcb6b6e2e19f', 'index': 7, 'kind': 'transition', 'step': 9, 'to_bits': 'c0814a4566f6a3df', 'validated': False}
- lj104 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c0813482d973a27b', 'hash': '6a31d26d86ec8d6b7b06053f57e0571e587bc3bfbd5c86433e35b2ba83359b7e', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c08155e0fd41ec98', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c0813482d973a27b', 'hash': '8dabf4eab5788ddd482fbaea487d416984b711fb679641cc44facf5004e84f9a', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0813482d727333f', 'validated': False}
- lj104 seed 2001: transition step 8; A={'action': 'sym', 'adopted': True, 'from_bits': 'c08177d623539120', 'hash': '2d843e1dda3995564144349f70ad5d05d3fb39fd635b40f0bf533fcda4abbea1', 'index': 5, 'kind': 'transition', 'step': 8, 'to_bits': 'c08177f991fc6ddd', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c08177d623539120', 'hash': '79d362de6386869a2f6aa4d8172bacc8f5b78375c80882af4953417292d18576', 'index': 5, 'kind': 'transition', 'step': 8, 'to_bits': 'c08177f962476ba6', 'validated': False}
- lj104 seed 2002: transition step 4; A={'action': 'sym', 'adopted': True, 'from_bits': 'c0814ddfa669655e', 'hash': '86f99ef5dc13eb53e89313d03cee0dd7b77e89e2d52e334a829d55084a76a469', 'index': 2, 'kind': 'transition', 'step': 4, 'to_bits': 'c0814cb25102d312', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c0814ddfa669655e', 'hash': '0a75eeef9609620a9ec6ae6b63f82b722094094150ed5aee18b616d8e93ffd8e', 'index': 2, 'kind': 'transition', 'step': 4, 'to_bits': 'c0814ddf5cf44a93', 'validated': False}
- lj150 seed 2000: transition step 2; A={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c08a3c080dea4fd2', 'hash': '076bde6b048998e5e38305a50a8dd7581fd03753afb903359f787fb03880f5d4', 'index': 0, 'kind': 'transition', 'step': 2, 'to_bits': 'c08a4945274465e3', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c08a3c080dea4fd2', 'hash': 'fa3d27384119bc7ea0e69080f40a6433fa7d9b88c4447a1d9298b795d6576dc4', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c08a3c080d6312e2', 'validated': False}
- lj150 seed 2001: transition step 7; A={'action': 'sym', 'adopted': True, 'from_bits': 'c08a88d487fb861c', 'hash': 'fa66d1bf0f4bdf156ebb950e5e607357c12ba891f9bc9840be145375b73d8ea5', 'index': 3, 'kind': 'transition', 'step': 7, 'to_bits': 'c08a8b81f9be7134', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c08a88d487fb861c', 'hash': 'e222c69e18b937927bde6ad358dc30a5843cbfa78af51a6ff0d384cf4b0815a2', 'index': 3, 'kind': 'transition', 'step': 7, 'to_bits': 'c08a88f869945b79', 'validated': False}
- lj150 seed 2002: transition step 3; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c08a32ea72866256', 'hash': '3f113336d034289c328bb80fafa8cc69f66e203112ce3de62470598a0be76f48', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c08a48ab027e2fe6', 'validated': True}; B={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c08a32ea72866256', 'hash': '587f5a0eda0a23c79d1e81647d30757042156b07283cc09e0f28f417d000fb9e', 'index': 2, 'kind': 'transition', 'step': 3, 'to_bits': 'c08a5c1a601ab4f8', 'validated': True}
- au55 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c067c18c45d42bc0', 'hash': '47f9a7b73e9c581bfad322f4721ea5bffc17866f4c67a51328d117559013798a', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c067c6c0a35aee82', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c067c18c45d42bc0', 'hash': 'cf14fe7926f52a299226f3f9b9b1b2238ed2f4837b8a24e44029c0b3e731e238', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c067c18c412e8d43', 'validated': False}
- au55 seed 2001: transition step 9; A={'action': 'sym', 'adopted': True, 'from_bits': 'c067c1d274741c88', 'hash': '30af9dc01b93f24a43fc1cc972933a3b0dd7417ec20bfd7f216f765b49b68f37', 'index': 8, 'kind': 'transition', 'step': 9, 'to_bits': 'c067c18fcbe201d1', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c067c1d274741c88', 'hash': 'c09b0710be4ffcc23b6042b6953d4e46d591c2fe60178d3b3ce8696926b46874', 'index': 8, 'kind': 'transition', 'step': 9, 'to_bits': 'c067c1d26fa2e640', 'validated': False}
- au55 seed 2002: transition step 6; A={'action': 'sym', 'adopted': True, 'from_bits': 'c067c7af3cd4eddc', 'hash': '24eb02dbbfef15656e6c50a4f2f8bb49151fb35ef937def63c8ea53ed72deaba', 'index': 5, 'kind': 'transition', 'step': 6, 'to_bits': 'c067c71ef5743475', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c067c7af3cd4eddc', 'hash': '9748880102848a9a234fe2356675f29a5d2bb11de0486b061ae3b8ab7acaa9ec', 'index': 5, 'kind': 'transition', 'step': 6, 'to_bits': 'c067c7af37498ec9', 'validated': False}
- m80r6 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c07684e96b5e5683', 'hash': '98d3254957f2e811e862c907bf14cadd269cfad589c7c921e1dd10ed907aa809', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0767bb0285c96b0', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c07684e96b5e5683', 'hash': 'b3d2971368e8fac30c16133b74903ef4105f437cd94bebda565eeeddebf5ba54', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c07684e964572feb', 'validated': False}
- m80r6 seed 2001: transition step 7; A={'action': 'dls-kernel', 'adopted': True, 'from_bits': 'c076924341104598', 'hash': 'dc9798d5ec46cae9eb897e0c21e8b4c52634c06918965a4e87d8fc1399e6c0de', 'index': 2, 'kind': 'transition', 'step': 7, 'to_bits': 'c076a7bc45c9226f', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c076924341104598', 'hash': '22c0c7d07bfecfbc3cb78f8dac426ceb3b88d888ad756aec0a31e467c30f6ebc', 'index': 2, 'kind': 'transition', 'step': 5, 'to_bits': 'c07692433da443b8', 'validated': False}
- m80r6 seed 2002: transition step 7; A={'action': 'sym', 'adopted': True, 'from_bits': 'c076bb22da8e64ed', 'hash': 'bc4ada2b65fddf42517ea5db697b94cb1626838b4c9396ba5a49031983589de4', 'index': 6, 'kind': 'transition', 'step': 7, 'to_bits': 'c076bae136a8467b', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c076bb22da8e64ed', 'hash': 'f7608e7ed7b210e7803f54134f9a12dfd9f017250b1689dbc482d690f53b1e5c', 'index': 6, 'kind': 'transition', 'step': 7, 'to_bits': 'c076bb5eda5a36b0', 'validated': False}
- m80r14 seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c06a881fbbc657b5', 'hash': 'fe0904acef30621c20ca0d55aff09c5690335f45a9dcc31279f21a5d752c8cef', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c0705e866711954a', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c06a881fbbc657b5', 'hash': '3780bc5c9d8ed4db69b1c0aa6b8038d75fd1fc228ef57fefc6ec1de5799628fc', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c06acd7db7da4c3f', 'validated': False}
- m80r14 seed 2001: transition step 2; A={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c0713a685117326d', 'hash': '03f9c6595f16b3cc5ecd9962f70d991c51523c70cc1dc49314674f784844f1b9', 'index': 1, 'kind': 'transition', 'step': 2, 'to_bits': 'c071c8d604853853', 'validated': True}; B={'action': 'orbit-completion', 'adopted': True, 'from_bits': 'c0713a685117326d', 'hash': '8e6bdf686353b02bbf7330cbf4ed1efd8ab554697dbebfb44e55bd93ce6a9b60', 'index': 1, 'kind': 'transition', 'step': 2, 'to_bits': 'c071b0a2a76220a5', 'validated': True}
- m80r14 seed 2002: transition step 5; A={'action': 'sym', 'adopted': True, 'from_bits': 'c0716eab46583dd2', 'hash': '1437ccad224122d472069af467589ee177bcb8f5c3fde1a066d465c1c22dd060', 'index': 4, 'kind': 'transition', 'step': 5, 'to_bits': 'c0716eab46583dd6', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c0716eab46583dd2', 'hash': '31b4caaeba37f136eae33e88b3da5680508947d0ce52178856e1e6a54e9d7013', 'index': 4, 'kind': 'transition', 'step': 5, 'to_bits': 'c0716eab36613aff', 'validated': False}
- lj75-seed seed 2000: transition step 1; A={'action': 'sym', 'adopted': True, 'from_bits': 'c077d6af44961ebc', 'hash': '9283a155360ec6287fbe07b2a7375ea6de4c7ccd8f7688e778baddad022232f5', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c077fc971f260b2f', 'validated': True}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c077d6af44961ebc', 'hash': '6b4613cdc49831ed68fd79610bf927e97c139c98e49735e2c70edf142574fc78', 'index': 0, 'kind': 'transition', 'step': 1, 'to_bits': 'c07835579d138d1b', 'validated': True}
- lj75-seed seed 2001: transition step 7; A={'action': 'sym', 'adopted': True, 'from_bits': 'c0782890dcd4cd8b', 'hash': '41c12e7be95a8f93e154a0665f88ae941bb0fcb8be34e4abd5b69af3faf19650', 'index': 4, 'kind': 'transition', 'step': 7, 'to_bits': 'c0782accd07a7e26', 'validated': False}; B={'action': 'sym', 'adopted': True, 'from_bits': 'c0782890dcd4cd8b', 'hash': '52ad9f698ea6250b2065f39875c198edea8a5688bbe168fad0ef6787d84a6a2e', 'index': 4, 'kind': 'transition', 'step': 7, 'to_bits': 'c0782bda2cc423f5', 'validated': False}
- lj75-seed seed 2002: transition step 9; A={'action': 'continuous-symmetry-csm', 'adopted': False, 'from_bits': 'c0782d53bb5bb5ba', 'hash': 'ecb2f61d6a2acb4283f661fe4fe093bab2d3e80e0d1a04ef7ab6c9745204a371', 'index': 4, 'kind': 'transition', 'step': 9, 'to_bits': 'c0782d53bb4cdd83', 'validated': True}; B={'action': 'continuous-symmetry-csm', 'adopted': True, 'from_bits': 'c0782d53bb5bb5ba', 'hash': 'da107ccd644375d6304411da10a748e4e8b5589bf3c0d3340a08bb84dda3c516', 'index': 4, 'kind': 'transition', 'step': 9, 'to_bits': 'c0782d53bb9412c9', 'validated': True}

### Gate 2 per-cell results

| Cell | Driver | A ns/call | B ns/call | B/A | A2 base ns/call | A2 ns/call | A2/A | Verdict |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| lj38 | h2h_gmin_proof | 2461.291 | 2785.716 | 1.131811 | 2467.219 | 2462.434 | 0.998061 | FAIL |
| lj55 | h2h_gmin_proof | 5630.582 | 5020.929 | 0.891725 | 5621.718 | 5611.671 | 0.998213 | PASS |
| lj75 | h2h_gmin_proof | 7286.673 | 8757.526 | 1.201855 | 7562.432 | 7386.115 | 0.976685 | FAIL |
| lj98 | h2h_gmin_proof | 10622.439 | 14155.340 | 1.332588 | 10199.512 | 10033.197 | 0.983694 | FAIL |
| lj102 | h2h_gmin_proof | 10092.087 | 16240.924 | 1.609273 | 10022.230 | 10344.666 | 1.032172 | FAIL |
| lj103 | h2h_gmin_proof+lj103 | 11617.866 | 19006.849 | 1.636002 | 11287.278 | 11293.725 | 1.000571 | FAIL |
| lj104 | h2h_gmin_proof | 10366.498 | 17225.663 | 1.661667 | 10699.502 | 10358.403 | 0.968120 | FAIL |
| lj150 | h2h_gmin_proof | 20151.146 | 44370.143 | 2.201867 | 19069.629 | 18670.052 | 0.979046 | FAIL |
| au55 | h2h_metal | 9768.721 | 12325.230 | 1.261704 | 9502.016 | 9599.712 | 1.010282 | FAIL |
| m80r6 | h2h_morse | 11399.441 | 10456.806 | 0.917309 | 11392.334 | 11402.178 | 1.000864 | PASS |
| m80r14 | h2h_morse | 9003.531 | 15016.591 | 1.667856 | 8952.314 | 9018.238 | 1.007364 | FAIL |
| lj75-seed | h2h_seed | 10867.106 | 8936.147 | 0.822312 | 10925.049 | 11058.866 | 1.012249 | PASS |

Each timing entry is the pre-registered median of six raw
observations. The complete append-only rows and derived CSV
tables are adjacent to this report.
