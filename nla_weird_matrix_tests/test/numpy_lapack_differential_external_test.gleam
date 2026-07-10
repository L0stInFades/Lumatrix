import gleam/float
import gleam/list
import lumatrix/direct
import lumatrix/eigen
import lumatrix/least_squares
import lumatrix/matrix
import lumatrix/svd
import lumatrix/vector
import nla_weird_matrix_tests/generated_cases
import nla_weird_support

pub fn dense_solve_matches_numpy_lapack_test() {
  list.each(generated_cases.solve_oracles(), fn(sample) {
    let assert Ok(a) = matrix.from_rows(sample.rows)
    let b = vector.from_list(sample.rhs)
    let expected = vector.from_list(sample.solution)
    let assert Ok(actual) = direct.solve(a, b)

    nla_weird_support.assert_vector_close(actual, expected, 1.0e-9)
    nla_weird_support.assert_residual_small(a, actual, b, 1.0e-9)
  })
}

pub fn least_squares_matches_numpy_lapack_test() {
  list.each(generated_cases.least_squares_oracles(), fn(sample) {
    let assert Ok(a) = matrix.from_rows(sample.rows)
    let b = vector.from_list(sample.rhs)
    let expected = vector.from_list(sample.solution)
    let assert Ok(actual) = least_squares.solve(a, b)

    nla_weird_support.assert_vector_close(actual.solution, expected, 1.0e-8)
    nla_weird_support.assert_close_to(
      actual.residual_norm,
      sample.residual_norm,
      1.0e-8,
    )
  })
}

pub fn singular_values_match_numpy_lapack_test() {
  list.each(generated_cases.svd_oracles(), fn(sample) {
    let assert Ok(a) = matrix.from_rows(sample.rows)
    let expected = vector.from_list(sample.singular_values)
    let assert Ok(actual) = svd.singular_values(a)

    nla_weird_support.assert_vector_close(actual, expected, 1.0e-7)
  })
}

pub fn symmetric_eigenvalues_match_numpy_lapack_test() {
  list.each(generated_cases.symmetric_eigen_oracles(), fn(sample) {
    let assert Ok(a) = matrix.from_rows(sample.rows)
    let assert Ok(actual) = eigen.jacobi_eigen(a, 120, 1.0e-12)
    let actual_values =
      vector.to_list(actual.diagonal)
      |> list.sort(by: float.compare)

    assert actual.converged
    assert_float_lists_close(actual_values, sample.eigenvalues, 1.0e-7)
  })
}

pub fn cholesky_factor_matches_numpy_lapack_test() {
  list.each(generated_cases.cholesky_oracles(), fn(sample) {
    let assert Ok(a) = matrix.from_rows(sample.rows)
    let assert Ok(expected) = matrix.from_rows(sample.lower)
    let assert Ok(actual) = direct.cholesky_factor(a)

    nla_weird_support.assert_matrix_close(actual.l, expected, 1.0e-8)
  })
}

fn assert_float_lists_close(
  actual: List(Float),
  expected: List(Float),
  tolerance: Float,
) -> Nil {
  assert list.length(actual) == list.length(expected)
  list.each(list.zip(actual, with: expected), fn(pair) {
    assert float.absolute_value(pair.0 -. pair.1) <=. tolerance
  })
}
