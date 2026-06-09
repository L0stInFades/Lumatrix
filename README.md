# Lumatrix

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

中文文档: [README.zh.md](README.zh.md)

Lumatrix is a pure Gleam numerical linear algebra library focused on small,
inspectable numerical kernels, checked data shapes, residual-aware APIs, and
stable default algorithms.

It is not trying to replace BLAS or LAPACK. Its center of gravity is a
Gleam-native layer that is easy to audit, gentle to extend, and dependable
enough for lightweight numerical work inside Gleam applications and tools.

## What It Offers

- Dense vectors and row-major matrices with checked construction and opaque
  internals.
- Direct solvers, orthogonal transformations, least-squares solvers, iterative
  methods, Krylov methods, eigenvalue routines, and basic error analysis.
- Stability-oriented building blocks: pivoted direct solvers, Householder and
  Givens QR paths, residual diagnostics, refinement tools, and Krylov solvers
  with explicit breakdown handling.

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
- `lumatrix/krylov`: Arnoldi, Lanczos, GMRES, restarted GMRES, BiCG, BiCGSTAB,
  and MINRES.
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

## Quality Checks

The local and CI checks use the same core loop: formatting, tests, and generated
documentation. Numerical routines are expected to expose convergence state and
residual quality rather than hiding failure behind unchecked values.

## Repository Layout

- `src/lumatrix/*.gleam`: library modules.
- `test/lumatrix_test.gleam`: unit and algorithm behavior tests.
- `gleam.toml` and `manifest.toml`: package metadata and lockfile.
- `.github/workflows/*.yml`: repository automation for checks.

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
