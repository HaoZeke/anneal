<!-- rgpot-foreign: subprojects/rgpot.wrap -Dwith_rgpot -->
# eOn packaging: pkg-config install of rgpot

eOn resolves **installed** rgpot via `dependency('rgpot')` before the Meson
subproject wrap (`subprojects/rgpot.wrap`). Engines stay **runtime dlopen**
(no torch / metatomic / xtb at link time).

## Build + install profile (matches eOn wrap options)

<!-- rgpot-doc: syntax configures and installs a separate build tree -->
```bash
meson setup build-eon-export \
  -Dwith_rpc_client_only=true \
  -Dpure_lib=true \
  -Dwith_tests=false \
  -Dwith_examples=false \
  -Dwith_xtb=false \
  -Dwith_metatomic=false \
  -Dwith_rpc=false \
  --prefix=/usr/local
meson compile -C build-eon-export
meson install -C build-eon-export
# installs: libnwchempot.a, libcpmdpot.a, libpot_client_bridge,
# include/rgpot/** (with rpc/Potentials.capnp and the generated
# rpc/Potentials.capnp.h), and rgpot.pc
# (the generated Cap'n Proto code is linked into libpot_client_bridge;
# ptlrpc is an uninstalled static archive, so there is no libptlrpc)
#
# This client-only profile has no umbrella library, so the archives are the
# installed product, and rgpot.pc lists only libnwchempot and libcpmdpot.
# A full librgpot build link_wholes the same frontends into librgpot and
# does not install libnwchempot.a or libcpmdpot.a.
```

Optional engine plugins (install separately / set `*_ENGINE` env at run time):

| Engine | Env | Linked into eOn? |
|--------|-----|------------------|
| `libnwchemc.so` | `NWCHEMC_LIBRARY` / `RGPOT_NWCHEMC_ENGINE` | no |
| `libcpmdc.so` | `CPMDC_LIBRARY` / `RGPOT_CPMDC_ENGINE` | no |
| `libmetatomic_engine.so` | `RGPOT_METATOMIC_ENGINE` | no |
| `libxtb_engine.so` | `RGPOT_XTB_ENGINE` | no (engine is runtime-only) |

The thin `XTBDlopen` frontend is a static archive that is not installed.
A full build links it into `librgpot`. A client-only build links it into
the subproject dependency that the wrap fallback uses, but the installed
client-only profile above does not ship it, and `rgpot.pc` does not list
it.

## eOn configure

<!-- rgpot-doc: skip configures eOn, not this repository -->
```bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
meson setup bbdir -Dwith_rgpot=true
# uses pkg-config 'rgpot' when present; else falls back to subproject wrap
```

Wrap fallback still uses:

<!-- rgpot-doc: skip meson option list from eOn's wrap, not a program -->
```text
with_rpc_client_only=true, pure_lib=true, with_tests=false, with_examples=false
```
