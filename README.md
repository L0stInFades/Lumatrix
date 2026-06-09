# nla

[![Package Version](https://img.shields.io/hexpm/v/nla)](https://hex.pm/packages/nla)
[![Hex Docs](https://img.shields.io/badge/hex-docs-ffaff3)](https://hexdocs.pm/nla/)

`nla` is a pure Gleam numerical linear algebra package aimed at textbook-style
algorithms and readable implementations.

Current modules:

- `nla/vector` and `nla/matrix`: dynamic-size dense vectors and row-major matrices.
- `nla/direct`: Gauss transforms, Gaussian elimination, LU with partial pivoting, triangular solves, determinant and inverse.
- `nla/orthogonal`: Householder transformations, Givens rotations, Householder QR, Givens QR, classical Gram-Schmidt QR and modified Gram-Schmidt QR.
- `nla/error_analysis`: residuals, iterative refinement, backward/forward error helpers and infinity-norm condition estimates.
- `nla/least_squares`: normal equations, Householder QR, classical Gram-Schmidt QR, modified Gram-Schmidt QR and stability diagnostics for least-squares solvers.
- `nla/iterative`: Jacobi, Gauss-Seidel, SOR, stationary-iteration convergence diagnostics, steepest descent, CG, practical CG with residual replacement, general preconditioned CG and Jacobi-preconditioned CG.
- `nla/krylov`: Arnoldi iteration, Lanczos tridiagonalization, GMRES and restarted GMRES for Krylov subspace methods.
- `nla/eigen`: power method, inverse power method, Hessenberg reduction, real Schur block classification and eigenvalue extraction, symmetric tridiagonal reduction, symmetric Jacobi eigenvalue iteration, symmetric QR eigendecomposition, basic QR iteration, QR convergence histories, Rayleigh/Wilkinson shifted QR, Givens-based implicit single-shift QR and explicit double-shift QR.

Example:

```gleam
import nla/direct
import nla/matrix
import nla/vector

pub fn main() -> Nil {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(x) = direct.solve(a, b)

  assert vector.approx_equal(x, vector.from_list([0.2, 0.6]), 1.0e-8)
}
```

This is not a BLAS/LAPACK replacement. It favors explicit algorithms and tests
over low-level performance.

## Development

```sh
gleam run   # Run the project
gleam test  # Run the tests
```
