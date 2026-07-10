# Numerical and error contract

## Accepted inputs

Safe numerical APIs accept finite IEEE-754 values, valid dimensions, valid
algorithm options, and workloads whose requested outputs are representable as
finite floats. A tolerance of zero is allowed where documented; iteration
counts must be non-negative.

Symmetry checks are relative to the infinity norm of the complete matrix, not
to an individual off-diagonal pair. Cholesky uses a `1e-12` symmetry tolerance
independently of its stricter pivot-singularity tolerance. This accepts
roundoff-sized asymmetry consistently at tiny and huge global scales while
still rejecting structurally non-symmetric inputs.

`matrix.from_rows`, `matrix.from_flat`, and `matrix.from_fn` reject non-finite
matrix entries. `vector.try_from_list` is the checked vector constructor.
`vector.from_list` remains unchecked for compatibility, but algorithms that
return `Result` validate such vectors before arithmetic. Complex values have
the analogous `complex.try_new` and `complex.vector_try_from_list` entry points.

Callbacks passed to `from_fn` or `zip_with` must themselves be total and return
finite values. The library cannot intercept a runtime exception raised inside
user code.

## Error meanings

| Error | Meaning |
| --- | --- |
| `DimensionMismatch` | Shapes do not satisfy the operation contract. |
| `InvalidInput` | A finite option or domain value is invalid. |
| `NonFiniteInput` | NaN or infinity reached a checked boundary. |
| `ArithmeticOverflow` | A finite operation or result cannot be represented. |
| `InternalInvariant` | Validated state contradicted a library invariant; report this as a bug. |
| `NotSquare` | A square matrix was required. |
| `OutOfBounds` | A checked index was outside the value. |
| `SingularMatrix` | A required pivot or search direction is numerically singular. |
| `ZeroNorm` | Normalization was requested for a zero vector. |
| `NoConvergence` | An algorithm whose API treats non-convergence as an error exhausted its budget. |

Iterative linear solvers return `Ok(IterationResult(converged: False, ...))`
when a valid non-negative iteration budget is exhausted. Their tolerance is an
absolute residual tolerance. Scaling `A`, `b`, and the tolerance by the same
positive factor preserves the iteration problem; the implementation normalizes
all three together and restores the reported residual to caller units.

## Panic boundary

For accepted inputs, public functions returning `Result` are expected to return
a structured error rather than panic. `matrix.unsafe_get` and
`sparse.unsafe_get` are deliberately outside that guarantee and may panic on an
invalid index. Assertions inside tests, benchmarks, and user callbacks are also
outside the library guarantee.

Non-`Result` compatibility helpers such as `matrix.scale`, `vector.scale`, and
`sparse.scale`, complex scalar arithmetic, or generic `zip_with` cannot report
overflow. Use `matrix.checked_scale`, `vector.checked_scale`,
`sparse.checked_scale`, and the named checked arithmetic operations when values
may approach the floating-point range.
