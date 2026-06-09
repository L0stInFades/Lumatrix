# Lumatrix

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

Lumatrix is a pure Gleam numerical linear algebra package aimed at textbook-style
algorithms and readable implementations.

It is designed for learning, testing, and reference implementations rather than
being a BLAS/LAPACK replacement.

## Modules

- `lumatrix/vector` and `lumatrix/matrix`: dynamic-size dense vectors and row-major matrices.
- `lumatrix/direct`: Gauss transforms, Gaussian elimination, LU with partial pivoting, Cholesky factorization for SPD systems, triangular solves, determinant and inverse.
- `lumatrix/orthogonal`: Householder transformations, Givens rotations, Householder QR, Givens QR, classical Gram-Schmidt QR and modified Gram-Schmidt QR.
- `lumatrix/error_analysis`: residuals, iterative refinement, backward/forward error helpers and infinity-norm condition estimates.
- `lumatrix/least_squares`: normal equations, Householder QR, Givens QR, classical Gram-Schmidt QR, modified Gram-Schmidt QR and stability diagnostics for least-squares solvers.
- `lumatrix/iterative`: Jacobi, Gauss-Seidel, SOR, stationary-iteration convergence diagnostics, steepest descent, CG, practical CG with residual replacement, general preconditioned CG and Jacobi-preconditioned CG.
- `lumatrix/krylov`: Arnoldi iteration, Lanczos tridiagonalization, GMRES and restarted GMRES for Krylov subspace methods.
- `lumatrix/eigen`: power method, inverse power method, Hessenberg reduction, real Schur block classification and eigenvalue extraction, symmetric tridiagonal reduction, symmetric Jacobi eigenvalue iteration, symmetric QR eigendecomposition, basic QR iteration, QR convergence histories, Rayleigh/Wilkinson shifted QR, Givens-based implicit single-shift QR and explicit double-shift QR.

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

## Repository Layout

- `src/lumatrix/*.gleam`: library modules.
- `test/lumatrix_test.gleam`: unit and algorithm behavior tests.
- `gleam.toml` and `manifest.toml`: package metadata and dependency lockfile.
- `.github/workflows/ci.yml`: formatting, test, and docs checks.

## Development

```sh
gleam format --check src test
gleam test
gleam docs build
```

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE).
