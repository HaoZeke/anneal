# Anneal

This is a `python` package for simulated annealing (and quenching) in all its
many guises. The design decisions are described in the corresponding InTechOpen
article.

## Development

We use `micromamba` to manage system dependencies, with `meson-python` (backend),
`build` (frontend) and `twine` for managing the upload to PyPI.

``` sh
micromamba create -f environment.yml
micromamba activate anneal-dev
```

## Contributing

> All contributions are welcome!!

- We follow the [NumPy commit guidelines](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message).
- Please run `pdm all` and ensure no linting or test errors exist
- [Co-author commits](https://github.blog/2018-01-29-commit-together-with-co-authors/) generously

# License
MIT.
