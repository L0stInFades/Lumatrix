# Lumatrix benchmarks

This package benchmarks the public API as an external path dependency. It uses
the same deterministic inputs on both supported targets and reports average
wall-clock milliseconds as CSV.

```sh
cd benchmarks
gleam run --target erlang --no-print-progress
gleam run --target javascript --no-print-progress
```

The suite covers dense matrix-vector and matrix-matrix multiplication, LU
solves, Householder-QR least squares, thin SVD, and conjugate gradient. Results
are evidence for the tested size tiers in `../SUPPORT.md`; they are not a
machine-independent performance guarantee.
