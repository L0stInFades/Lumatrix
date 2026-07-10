# Changelog

All notable changes to this project are documented in this file. The format is
based on Keep a Changelog and this project follows semantic versioning.

## [Unreleased]

## [1.0.0] - 2026-07-10

### Added

- Dense linear algebra, direct and iterative solvers, QR, least squares, SVD,
  eigenvalue routines, sparse CSR operations, complex vectors, and error
  analysis.
- Structured `NonFiniteInput`, `ArithmeticOverflow`, and `InternalInvariant`
  failures plus checked dense/sparse scaling and real/complex construction APIs.
- Erlang/JavaScript CI, external-consumer stress tests, deterministic randomized
  properties, and NumPy/LAPACK differential fixtures.
- Reproducible cross-target benchmarks, a tested-size support policy, Hex
  tarball artifacts, and GitHub build-provenance attestations.

### Changed

- Dense matrices and vectors now use constant-time target-native indexed storage
  while retaining opaque public types and existing conversion APIs.
- Iterative solvers normalize the complete linear system and tolerance, use
  overflow-safe dot singularity checks, and validate every iteration option;
  the same scale contract covers GMRES, BiCG, BiCGSTAB, and MINRES.
- Dense dot products and matrix products use scaled checked accumulation to
  prevent avoidable underflow and overflow.
- Krylov breakdown tests use normalized dot products without multiplying vector
  norms, scalar recurrences report structured overflow, and symmetry validation
  is relative to the complete matrix scale.

[Unreleased]: https://github.com/L0stInFades/Lumatrix/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/L0stInFades/Lumatrix/releases/tag/v1.0.0
