# Anneal

This is a `python` package for simulated annealing of various types. The design
decisions are described in the corresponding InTechOpen article.

## Development

We use `micromamba` to manage system dependencies, with `meson-python` (backend),
`build` (frontend) and `twine` for managing the upload to PyPI.

``` sh
micromamba create -f environment.yml
micromamba activate anneal-dev
```

# License
MIT.
