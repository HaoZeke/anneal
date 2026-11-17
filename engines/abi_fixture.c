/* Stand-in for librgpot in the abi_symbols artifact-kind tests. It is
   built as a shared object and as static archives. The hidden name stays
   out of a shared object's dynamic table, so a baseline that asks for it
   must fail there. */
int rgpot_nlist_fixture_open(void) { return 1; }
int rgpot_fast_fixture_wrap(void) { return 2; }
__attribute__((visibility("hidden"))) int rgpot_nlist_fixture_hidden(void) {
  return 3;
}
