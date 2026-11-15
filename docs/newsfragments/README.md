# Towncrier news fragments

User-facing changes land here **before** a release is cut. On `cog bump`,
`uvx towncrier build` folds these into **`CHANGELOG.md`** (canonical user
changelog; see also `docs/orgmode/changelog.org`, which only includes that file
for Sphinx) and removes the fragment files.

## Discipline

- **One fragment per user-visible change** (not one per commit).
- Prefer issue-linked names (`42.added.md`) when there is a tracking issue; use
  `+slug.fixed.md` for internal/no-issue fixes.
- Write for **users/integrators**, not implementers (implementation detail goes
  in the PR body or `dev` type only when relevant to contributors).
- **Required** when the PR touches shipped surfaces (CI: `towncrier` workflow
  runs `towncrier check` vs the PR base for `CppCore/`, `rgpot-core/`, meson,
  CMake, pixi workspace, etc.).
- The `towncrier` job runs on every pull request with no path filter, so a
  `docs` / `ci` / `chore` / `style` change also needs a fragment (`dev` type
  when it does not alter library/CLI/crate behaviour).

Preview: `uvx towncrier build --draft` or `pixi r towncrier-draft` if defined.

## Naming

```
<issue-or-slug>.<type>.md
```

| `<type>` directory suffix | Section in CHANGELOG |
|---------------------------|----------------------|
| `breaking` | Breaking |
| `security` | Security |
| `removed` | Removed |
| `deprecated` | Deprecated |
| `added` | Added |
| `dev` | Developer |
| `changed` | Changed |
| `fixed` | Fixed |
| `misc` | Miscellaneous |

Examples:

```bash
uvx towncrier create --content "MetatomicPot loads TorchScript models directly." 37.added.md
uvx towncrier create --content "Vesin 0.5+ device struct compatibility." +vesin05.fixed.md
```

After merge to `main`, maintainers cut the release with `cog bump` (see
`docs/orgmode/contributing/developer/release.org`). Do **not** hand-edit a
`## [X.Y.Z]` block above the towncrier marker in `CHANGELOG.md`.
