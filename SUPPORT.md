# Support policy and tested sizes

Lumatrix is a pure Gleam library for small and medium dense numerical problems.
It does not call BLAS or LAPACK at runtime, so large dense workloads should use
a native accelerated library instead.

## Runtime targets

| Target | Continuous integration | Dense storage |
| --- | --- | --- |
| Erlang | OTP 27, Gleam 1.17.0 | immutable tuple |
| JavaScript | Node.js 24 on `ubuntu-latest`, Gleam 1.17.0 | copied `Float64Array` |

Both storage implementations provide constant-time indexed reads. Operations
that return a modified matrix or vector preserve value semantics by copying the
underlying storage.

## Benchmarked tiers

These are the largest shapes exercised by the reproducible benchmark suite,
not hard limits or performance guarantees.

| Operation | Largest benchmark shape | Intended tier |
| --- | ---: | --- |
| Dense matrix-vector product | 512 x 512 | medium |
| Dense matrix-matrix product | 64 x 64 | small/medium |
| LU solve | 64 x 64 | small/medium |
| Householder-QR least squares | 64 x 32 | small/medium |
| Thin one-sided Jacobi SVD | 32 x 16 | small |
| Conjugate gradient | 256 x 256 | medium |

The recorded run used macOS 15.7.4, an Intel Core i7-9750H, and 32 GiB RAM on
2026-07-10. Raw Erlang and JavaScript results are in `benchmarks/results/`.
Run the suite on deployment hardware before setting latency budgets.

## Correctness coverage

- Every root test and external-consumer test runs on Erlang and JavaScript.
- The external suite includes 64 deterministic randomized property cases.
- NumPy 2.4.4 supplies LAPACK-backed differential oracles for dense solve,
  least squares, SVD singular values, symmetric eigenvalues, and Cholesky.
- Iterative solvers are regression-tested under global scales `1e-150`, `1`,
  and `1e150`, with the absolute tolerance scaled by the same factor. This
  covers stationary, CG-family, GMRES, BiCG, BiCGSTAB, and MINRES solvers.

## Compatibility

The public `Matrix` and `Vector` types are opaque. Internal storage may change
without changing their construction, conversion, or indexing APIs. Semantic
versioning applies to public modules; modules under `lumatrix/internal` are
annotated internal and are not part of the compatibility contract.

See `NUMERICAL_CONTRACT.md` for accepted numeric inputs, error variants, and
the panic boundary.
