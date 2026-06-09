# Lumatrix

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![Release](https://github.com/L0stInFades/Lumatrix/actions/workflows/release.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

中文文档: [README.zh.md](README.zh.md)

Lumatrix is a pure Gleam numerical linear algebra library for people who want
the algorithms to stay close to the mathematics on the page, so they can be
read, checked, and gently extended.

It is not trying to replace BLAS or LAPACK. Its center of gravity is different:
clear textbook-style routines, careful API boundaries, and a small surface that
is comfortable for learning, testing, and lightweight numerical work in Gleam.

## What It Offers

- Dense vectors and row-major matrices with checked construction and opaque
  internals.
- Direct solvers, orthogonal transformations, least-squares solvers, iterative
  methods, Krylov methods, eigenvalue routines, and basic error analysis.
- Small implementations that are meant to be inspected, reasoned about, and
  improved without hiding the mathematics.

## API Notes

`Matrix` and `Vector` constructors are intentionally hidden. Build values with
`matrix.from_rows`, `matrix.from_columns`, `matrix.from_flat`, `matrix.from_fn`,
`vector.from_list`, `vector.zeros`, or `vector.basis`; inspect dimensions with
`matrix.rows`, `matrix.cols`, and `vector.dimension`.

Vectors are coordinate arrays, not separate row-vector or column-vector types.
In `matrix.mul_vec(a, x)`, `x` is interpreted as the column vector in `A * x`.
Use `matrix.row_matrix` or `matrix.column_matrix` when orientation needs to be
represented as a matrix.

QR results carry a `form` tag. Householder and Givens QR return `FullQR`
(`q` is m-by-m, `r` is m-by-n); classical and modified Gram-Schmidt return
`ThinQR` (`q` is m-by-n, `r` is n-by-n).

Prefer `matrix.get` for user-provided indices. `matrix.unsafe_get` is available
for internal-style code that has already proved bounds.

Least-squares solvers return the solution and residual norm. Condition numbers
and normal-equation residuals live in `least_squares.stability_diagnostics`.

## Modules

- `lumatrix/vector` and `lumatrix/matrix`: dense coordinate vectors and row-major
  matrices.
- `lumatrix/direct`: Gaussian elimination, LU with partial pivoting, Cholesky,
  triangular solves, determinant, and inverse.
- `lumatrix/orthogonal`: Householder transformations, Givens rotations, and QR
  factorizations.
- `lumatrix/least_squares`: Householder-QR default `solve`, normal equations,
  Givens QR, Gram-Schmidt QR, and diagnostics.
- `lumatrix/error_analysis`: residuals, iterative refinement, error bounds, and
  infinity-norm condition estimates.
- `lumatrix/iterative`: Jacobi, Gauss-Seidel, SOR, steepest descent, CG, and
  preconditioned CG variants.
- `lumatrix/krylov`: Arnoldi, Lanczos, GMRES, and restarted GMRES.
- `lumatrix/eigen`: power methods, Hessenberg and tridiagonal reductions,
  Jacobi eigen iteration, QR iterations, Schur block helpers, and symmetric
  eigendecomposition.

## Example

```gleam
import lumatrix/direct
import lumatrix/matrix
import lumatrix/vector

pub fn main() -> Nil {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(x) = direct.solve(a, b)

  assert vector.approx_equal(x, vector.from_list([0.2, 0.6]), 1.0e-8)
}
```

## Development

```sh
gleam format --check src test
gleam test
gleam docs build
```

## CI/CD

The CI workflow runs on pushes, pull requests, and manual dispatch. It downloads
dependencies, checks formatting, runs the test suite, and builds the generated
documentation.

The release workflow publishes to Hex.pm from either a `vX.Y.Z` tag or a manual
dispatch. The release version must match `gleam.toml`, and publishing requires a
`HEXPM_API_KEY` secret on the repository or the protected `hexpm` environment.

Before the first release, add a Hex.pm API key:

```sh
gh secret set HEXPM_API_KEY --repo L0stInFades/Lumatrix
```

```sh
git tag v1.0.0
git push origin v1.0.0
```

## Repository Layout

- `src/lumatrix/*.gleam`: library modules.
- `test/lumatrix_test.gleam`: unit and algorithm behavior tests.
- `gleam.toml` and `manifest.toml`: package metadata and lockfile.
- `.github/workflows/ci.yml`: formatting, test, and docs checks.
- `.github/workflows/release.yml`: checked Hex.pm publishing on release tags or
  manual dispatch.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
