import gleam/list
import lumatrix
import lumatrix/error
import lumatrix/matrix
import lumatrix/vector
import nla_weird_matrix_tests/generated_cases
import nla_weird_support

pub fn root_main_smoke_test() {
  lumatrix.main()
}

pub fn error_type_constructors_are_public_test() {
  let errors = [
    error.DimensionMismatch(expected: "2", actual: "3"),
    error.InvalidInput("bad input"),
    error.NotSquare(rows: 2, cols: 3),
    error.OutOfBounds(row: 9, col: 1),
    error.SingularMatrix(pivot: 4),
    error.ZeroNorm,
    error.NoConvergence(iterations: 7, residual: 0.125),
  ]

  assert list.length(errors) == 7
}

pub fn generated_vectors_round_trip_and_normalize_test() {
  nla_weird_support.each_vector_case(generated_cases.vector_cases(), fn(sample) {
    let v = nla_weird_support.vector_from_case(sample)
    assert vector.dimension(v) == list.length(sample.data)
    assert vector.to_list(v) == sample.data

    let assert Ok(n2) = vector.norm2(v)
    assert n2 >=. 0.0
    assert vector.norm_inf(v) >=. 0.0

    case vector.normalize(v) {
      Ok(unit) -> {
        let assert Ok(unit_norm) = vector.norm2(unit)
        nla_weird_support.assert_close_to(
          unit_norm,
          1.0,
          nla_weird_support.tolerance(),
        )
      }
      Error(error.ZeroNorm) -> {
        assert vector.norm_inf(v) <=. 0.0
      }
      Error(_) -> panic as "unexpected vector normalization error"
    }
  })
}

pub fn generated_vector_arithmetic_identities_test() {
  nla_weird_support.each_vector_case(generated_cases.vector_cases(), fn(sample) {
    let v = nla_weird_support.vector_from_case(sample)
    let assert Ok(zero) = vector.zeros(vector.dimension(v))
    let assert Ok(sum) = vector.add(v, zero)
    let assert Ok(delta) = vector.sub(v, v)
    let twice = vector.scale(v, 2.0)
    let assert Ok(axpy_zero) = vector.axpy(-2.0, v, twice)
    let assert Ok(dot) = vector.dot(v, v)
    let assert Ok(norm) = vector.norm2(v)

    nla_weird_support.assert_vector_close(sum, v, nla_weird_support.tolerance())
    nla_weird_support.assert_vector_close(
      delta,
      zero,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_vector_close(
      axpy_zero,
      zero,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_close_to(dot, norm *. norm, 1.0e-3)
  })
}

pub fn vector_error_paths_test() {
  let short = vector.from_list([1.0, 2.0])
  let long = vector.from_list([1.0, 2.0, 3.0])
  let assert Ok(zero) = vector.zeros(0)

  case vector.zeros(-1) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "negative vector size should be invalid"
  }
  case vector.basis(3, 4) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "basis index outside size should be invalid"
  }
  case vector.get(short, 9) {
    Error(error.OutOfBounds(row: 0, col: 9)) -> Nil
    _ -> panic as "vector get should report out of bounds"
  }
  case vector.dot(short, long) {
    Error(error.DimensionMismatch(expected: "2", actual: "3")) -> Nil
    _ -> panic as "dot product should reject mismatched dimensions"
  }
  case vector.normalize(zero) {
    Error(error.ZeroNorm) -> Nil
    _ -> panic as "zero vector should not normalize"
  }
}

pub fn generated_matrices_round_trip_and_accessors_test() {
  nla_weird_support.each_matrix_case(all_matrix_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let rows = list.length(sample.rows)
    let assert [first_row, ..] = sample.rows
    let cols = list.length(first_row)
    let assert Ok(row0) = matrix.row(a, 0)
    let assert Ok(col0) = matrix.col(a, 0)
    let assert Ok(same) = matrix.set(a, 0, 0, matrix.unsafe_get(a, 0, 0))
    let transposed_twice = matrix.transpose(matrix.transpose(a))

    assert matrix.rows(a) == rows
    assert matrix.cols(a) == cols
    assert matrix.to_rows(a) == sample.rows
    assert vector.to_list(row0) == first_row
    assert vector.to_list(col0) == first_column(sample.rows)
    nla_weird_support.assert_matrix_close(
      same,
      a,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      transposed_twice,
      a,
      nla_weird_support.tolerance(),
    )
  })
}

pub fn generated_matrix_algebra_identities_test() {
  nla_weird_support.each_matrix_case(generated_cases.square_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let assert Ok(zero) =
      matrix.zeros(rows: matrix.rows(a), cols: matrix.cols(a))
    let assert Ok(identity) = matrix.identity(matrix.rows(a))
    let assert Ok(sum) = matrix.add(a, zero)
    let assert Ok(delta) = matrix.sub(a, a)
    let assert Ok(left_identity) = matrix.mul(identity, a)
    let assert Ok(right_identity) = matrix.mul(a, identity)
    let assert Ok(trace) = matrix.trace(a)
    let assert Ok(frobenius) = matrix.frobenius_norm(a)

    nla_weird_support.assert_matrix_close(sum, a, nla_weird_support.tolerance())
    nla_weird_support.assert_matrix_close(
      delta,
      zero,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      left_identity,
      a,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      right_identity,
      a,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_close_to(
      trace,
      nla_weird_support.trace_of_rows(sample.rows),
      nla_weird_support.tolerance(),
    )
    assert frobenius >=. 0.0
    assert matrix.norm_inf(a) >=. 0.0
  })
}

pub fn matrix_product_outer_swap_and_set_row_test() {
  nla_weird_support.each_matrix_case(all_matrix_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let x = nla_weird_support.ramp_vector(matrix.cols(a))
    let assert Ok(y) = matrix.mul_vec(a, x)
    let assert Ok(outer) = matrix.outer(y, x)
    let assert Ok(row0) = matrix.row(a, 0)
    let assert Ok(row_same) = matrix.set_row(a, 0, row0)
    let assert Ok(swapped) = matrix.swap_rows(a, 0, matrix.rows(a) - 1)
    let assert Ok(restored) = matrix.swap_rows(swapped, 0, matrix.rows(a) - 1)

    assert matrix.rows(outer) == vector.dimension(y)
    assert matrix.cols(outer) == vector.dimension(x)
    nla_weird_support.assert_matrix_vector_consistent(
      a,
      x,
      y,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      row_same,
      a,
      nla_weird_support.tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      restored,
      a,
      nla_weird_support.tolerance(),
    )
  })
}

pub fn matrix_zip_with_and_indices_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, -2.0], [3.0, 4.0]])
  let assert Ok(b) = matrix.from_rows([[0.5, 1.0], [-1.0, 2.0]])
  let assert Ok(zipped) = matrix.zip_with(a, b, fn(x, y) { 2.0 *. x -. y })
  let assert Ok(expected) = matrix.from_rows([[1.5, -5.0], [7.0, 6.0]])

  assert matrix.indices(4) == [0, 1, 2, 3]
  nla_weird_support.assert_matrix_close(
    zipped,
    expected,
    nla_weird_support.tolerance(),
  )
}

pub fn matrix_error_paths_test() {
  let assert Ok(square) = matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
  let assert Ok(rectangle) = matrix.from_rows([[1.0, 2.0, 3.0]])
  let assert Ok(short_row) = vector.zeros(1)
  let assert Ok(wrong_cols) = matrix.from_rows([[1.0, 2.0, 3.0]])
  let wrong_vector = vector.from_list([1.0, 2.0, 3.0])

  case matrix.from_rows([]) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "empty matrix rows should be invalid"
  }
  case matrix.from_rows([[1.0], [2.0, 3.0]]) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "ragged matrix rows should be invalid"
  }
  case matrix.from_flat(rows: 2, cols: 2, data: [1.0, 2.0, 3.0]) {
    Error(error.DimensionMismatch(expected: "4", actual: "3")) -> Nil
    _ -> panic as "flat matrix data length mismatch should be rejected"
  }
  case matrix.zeros(rows: 0, cols: 3) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "zero matrix rows should be invalid"
  }
  case matrix.get(square, -1, 0) {
    Error(error.OutOfBounds(row: -1, col: 0)) -> Nil
    _ -> panic as "negative matrix get should be out of bounds"
  }
  case matrix.set(square, 0, 7, 1.0) {
    Error(error.OutOfBounds(row: 0, col: 7)) -> Nil
    _ -> panic as "matrix set should reject bad column"
  }
  case matrix.row(square, 5) {
    Error(error.OutOfBounds(row: 5, col: 0)) -> Nil
    _ -> panic as "matrix row should reject bad row"
  }
  case matrix.col(square, 5) {
    Error(error.OutOfBounds(row: 0, col: 5)) -> Nil
    _ -> panic as "matrix col should reject bad column"
  }
  case matrix.mul(square, wrong_cols) {
    Error(error.DimensionMismatch(expected: "2", actual: "1")) -> Nil
    _ -> panic as "matrix multiply should reject incompatible shapes"
  }
  case matrix.mul_vec(square, wrong_vector) {
    Error(error.DimensionMismatch(expected: "2", actual: "3")) -> Nil
    _ -> panic as "matrix-vector multiply should reject incompatible vector"
  }
  case matrix.trace(rectangle) {
    Error(error.NotSquare(rows: 1, cols: 3)) -> Nil
    _ -> panic as "trace should require a square matrix"
  }
  case matrix.set_row(square, 0, short_row) {
    Error(error.DimensionMismatch(expected: "2", actual: "1")) -> Nil
    _ -> panic as "set_row should reject wrong vector length"
  }
}

fn all_matrix_cases() -> List(generated_cases.MatrixCase) {
  list.append(
    generated_cases.square_cases(),
    list.append(
      generated_cases.spd_cases(),
      list.append(
        generated_cases.tall_cases(),
        list.append(
          generated_cases.stationary_cases(),
          generated_cases.eigen_cases(),
        ),
      ),
    ),
  )
}

fn first_column(rows: List(List(Float))) -> List(Float) {
  case rows {
    [] -> []
    [row, ..rest] -> [first_float(row), ..first_column(rest)]
  }
}

fn first_float(values: List(Float)) -> Float {
  case values {
    [value, ..] -> value
    [] -> 0.0
  }
}
