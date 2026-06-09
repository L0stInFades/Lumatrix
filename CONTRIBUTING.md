# Contributing

Lumatrix is intended to keep numerical linear algebra algorithms explicit and easy
to inspect. Contributions should prefer readable textbook implementations over
low-level performance tricks.

## Development Checks

Run these before opening a pull request:

```sh
gleam format --check src test
gleam test
gleam docs build
```

The same checks run in GitHub Actions on push and pull request. The release
workflow in `.github/workflows/release.yml` validates release tags and manual
release requests by checking the package version and rerunning the project
checks.

## Guidelines

- Keep APIs small and typed with existing `lumatrix/error` errors.
- Add focused tests for new algorithms or changed numerical behavior.
- Prefer existing `lumatrix/vector`, `lumatrix/matrix`, `lumatrix/direct`, and
  `lumatrix/orthogonal` helpers before introducing new abstractions.
- Document numerical assumptions such as symmetry, positive definiteness,
  dimensions, tolerance, and convergence behavior in function names or tests.

## License

By contributing, you agree that your contributions are licensed under the
Apache License, Version 2.0.
