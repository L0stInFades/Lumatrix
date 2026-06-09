<div align="center">

# 🧮 Lumatrix

**Numerically stable linear algebra kernels in pure Gleam.**

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Pure Gleam](https://img.shields.io/badge/pure%20Gleam-no%20FFI-ffaff3.svg)](https://gleam.run)
[![Targets](https://img.shields.io/badge/targets-Erlang%20%7C%20JavaScript-ffaff3.svg)](https://gleam.run)

**English** · [简体中文](README.zh.md)

</div>

---

Lumatrix is a numerical linear algebra library written entirely in Gleam — no FFI, no NIFs, no C to babysit. It runs wherever Gleam runs, on both the Erlang and JavaScript targets.

It is not trying to outrace BLAS or LAPACK, and it doesn't pretend to. Its job is to be the honest `solve` inside your Gleam app: kernels small enough to actually read, shapes checked at construction, stable algorithms as the defaults, and solvers that report how good their answers really are — instead of handing you a number and wishing you luck.

## ✨ What's in the box

- **Checked-at-construction dense and sparse types** — coordinate vectors, row-major matrices, and canonical CSR sparse matrices. Ragged rows, out-of-bounds indices, and malformed data are rejected with errors at build time; the internals stay opaque.
- **The classic toolbox** — pivoted LU and Cholesky direct solvers, Householder / Givens / Gram-Schmidt QR, least squares, one-sided Jacobi SVD, real and complex eigenvalue routines, stationary iterative methods, and Krylov solvers.
- **Stability gear everywhere** — pivoting by default, residual diagnostics, iterative refinement, condition-number estimates, and explicit breakdown handling in Krylov methods. Failure is a value you can pattern match on, not a silently wrong answer.

## 🚀 Quick start

> [!NOTE]
> Lumatrix isn't on Hex yet. Pull it straight from GitHub (git dependencies need Gleam ≥ 1.12), or clone the repo and use a path dependency.

```toml
[dependencies]
lumatrix = { git = "https://github.com/L0stInFades/Lumatrix.git", ref = "main" }
# or, after cloning locally:
# lumatrix = { path = "../Lumatrix" }
```

Solve a linear system:

```math
\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 1 \\ 2 \end{bmatrix}
\quad\Longrightarrow\quad
x = \begin{bmatrix} 0.2 \\ 0.6 \end{bmatrix}
```

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

Least-squares solvers return the solution *and* how far off it is:

```gleam
import lumatrix/least_squares

let assert Ok(fit) = least_squares.solve(a, b)
// fit.solution      — the least-squares solution x̂
// fit.residual_norm — ‖A·x̂ − b‖₂, so you know how good the fit is
```

> [!TIP]
> The plain `solve` in each module is the stable default path (`direct.solve` is LU with partial pivoting, `least_squares.solve` is Householder QR). The fancier variants are there for when you know exactly why you want them.

## 🗺️ Module map

| Module | What's inside |
| --- | --- |
| 🧱 `lumatrix/vector` · `lumatrix/matrix` | Dense coordinate vectors and row-major matrices |
| 🚨 `lumatrix/error` | The shared `NlaError` type returned by every fallible function |
| 🌀 `lumatrix/complex` | Complex scalars and complex coordinate vectors |
| 🕸️ `lumatrix/sparse` | Canonical CSR sparse matrices: dense conversion, matrix-vector products, transpose, scaling, ∞-norm |
| 🔨 `lumatrix/direct` | Gaussian elimination, LU with partial/complete pivoting, Cholesky, triangular solves, determinant, inverse |
| 📐 `lumatrix/orthogonal` | Householder transformations, Givens rotations, QR factorizations |
| 🎯 `lumatrix/least_squares` | Householder-QR `solve`, plus normal equations, Givens QR, Gram-Schmidt QR, SVD least squares, and diagnostics |
| 💎 `lumatrix/svd` | Thin SVD via one-sided Jacobi, pseudoinverse, numerical rank, 2-norm, 2-norm condition number |
| 🩺 `lumatrix/error_analysis` | Residuals, iterative refinement, error bounds, ∞-norm condition estimates |
| 🔁 `lumatrix/iterative` | Jacobi, Gauss-Seidel, SOR, steepest descent, CG, preconditioned CG |
| 🚄 `lumatrix/krylov` | Arnoldi, Lanczos, GMRES (plain and restarted), BiCG, BiCGSTAB, MINRES |
| λ `lumatrix/eigen` | Power methods, Hessenberg/tridiagonal reduction, Jacobi and QR iterations, Schur helpers, symmetric eigendecomposition, generalized eigenvalues, complex eigenpairs from real Schur form |

## 📐 Conventions worth knowing

- **Constructors are checked; internals are opaque.** Build values with `matrix.from_rows` / `from_columns` / `from_flat` / `from_fn` and `vector.from_list` / `zeros` / `basis`; read shapes with `matrix.rows`, `matrix.cols`, and `vector.dimension`.
- **Vectors are plain coordinates.** There is no row/column vector distinction: `matrix.mul_vec(a, x)` treats `x` as the column in `A·x`. When orientation matters, use `matrix.row_matrix` or `matrix.column_matrix`.
- **QR results say what shape they are.** Householder and Givens QR return `FullQR` (`q` is m×m, `r` is m×n); classical/modified Gram-Schmidt return `ThinQR` (`q` is m×n, `r` is n×n). Check the `form` tag instead of guessing.
- **Two ways to index.** `matrix.get` for indices you don't trust; `matrix.unsafe_get` for internal-style code where bounds are already proved.
- **Solvers report quality.** Least-squares results carry their residual norm, and deeper diagnostics (condition numbers, normal-equation residuals) live in `least_squares.stability_diagnostics`.
- **SVD never forms `AᵀA`.** One-sided Jacobi sidesteps the condition-number squaring that normal equations bring, and `svd.rank`, `svd.pseudoinverse`, and `svd.condition_number` share one singular-value cutoff rule.
- **Sparse is its own type.** Canonical CSR storage: entries bounds-checked, sorted by row and column, duplicate coordinates summed, stored zeros dropped — all at construction time, not "eventually".
- **Complex and generalized eigenproblems are covered.** Real eigen routines extract complex eigenpairs from the real Schur form, with residual checks on `A·v = λ·v`. Generalized problems with invertible `B` are reduced to the standard problem on `B⁻¹·A` via the complete-pivoting solver, and residuals are reported against the original pencil.

## 🔬 Tests & quality

Two layers of tests keep the kernels honest:

- `test/` — unit and algorithm-behavior tests for every module (gleeunit).
- `nla_weird_matrix_tests/` — a separate package that treats the library as read-only and attacks the public API from the outside, using deterministic "weird" matrix fixtures generated with Python + NumPy.

```sh
cd nla_weird_matrix_tests
python3 tools/generate_weird_cases.py   # regenerate fixtures
gleam test
```

The standing rule: numerical routines must expose convergence state and residual quality. No hiding failure behind unchecked values.

## 🛠️ Development

```sh
gleam format --check src test
gleam test
gleam docs build
```

CI runs exactly the same loop. Contributions welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

- `src/lumatrix/*.gleam` — library modules
- `test/lumatrix_test.gleam` — unit and algorithm-behavior tests
- `nla_weird_matrix_tests/` — external adversarial test package
- `gleam.toml` / `manifest.toml` — package metadata and lockfile
- `.github/workflows/*.yml` — CI

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE).

---

<div align="center">

⎡ pure Gleam · careful pivoting · honest residuals ⎤

(=^･ω･^=)

</div>
