import gleam/float
import gleam/int
import gleam/list
import lumatrix/eigen
import lumatrix/error_analysis
import lumatrix/matrix
import lumatrix/orthogonal
import lumatrix/vector
import nla_weird_matrix_tests/generated_cases.{type MatrixCase, type VectorCase}

pub fn tolerance() -> Float {
  1.0e-7
}

pub fn loose_tolerance() -> Float {
  1.0e-5
}

pub fn close_to(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

pub fn assert_close_to(a: Float, b: Float, tol: Float) -> Nil {
  assert close_to(a, b, tol)
}

pub fn assert_vector_close(
  actual: vector.Vector,
  expected: vector.Vector,
  tol: Float,
) -> Nil {
  assert vector.approx_equal(actual, expected, tol)
}

pub fn assert_matrix_close(
  actual: matrix.Matrix,
  expected: matrix.Matrix,
  tol: Float,
) -> Nil {
  assert matrix.approx_equal(actual, expected, tol)
}

pub fn matrix_from_case(sample: MatrixCase) -> matrix.Matrix {
  let assert Ok(value) = matrix.from_rows(sample.rows)
  value
}

pub fn vector_from_case(sample: VectorCase) -> vector.Vector {
  vector.from_list(sample.data)
}

pub fn each_matrix_case(
  samples: List(MatrixCase),
  run: fn(MatrixCase) -> Nil,
) -> Nil {
  case samples {
    [] -> Nil
    [sample, ..rest] -> {
      run(sample)
      each_matrix_case(rest, run)
    }
  }
}

pub fn each_vector_case(
  samples: List(VectorCase),
  run: fn(VectorCase) -> Nil,
) -> Nil {
  case samples {
    [] -> Nil
    [sample, ..rest] -> {
      run(sample)
      each_vector_case(rest, run)
    }
  }
}

pub fn ramp_vector(size: Int) -> vector.Vector {
  vector.from_list(
    list.map(matrix.indices(size), fn(i) { int.to_float(i + 1) }),
  )
}

pub fn alternating_vector(size: Int) -> vector.Vector {
  vector.from_list(
    list.map(matrix.indices(size), fn(i) {
      case i % 2 == 0 {
        True -> int.to_float(i + 1)
        False -> 0.0 -. int.to_float(i + 1)
      }
    }),
  )
}

pub fn diagonal_matrix_from_vector(values: vector.Vector) -> matrix.Matrix {
  let assert Ok(result) =
    matrix.from_fn(
      rows: vector.dimension(values),
      cols: vector.dimension(values),
      with: fn(i, j) {
        case i == j {
          True -> unsafe_vector_get(values, i)
          False -> 0.0
        }
      },
    )
  result
}

pub fn assert_qr_reconstructs(
  original: matrix.Matrix,
  qr: orthogonal.QR,
  tol: Float,
) -> Nil {
  let assert Ok(reconstructed) = matrix.mul(qr.q, qr.r)
  let assert Ok(qtq) = matrix.mul(matrix.transpose(qr.q), qr.q)
  let assert Ok(identity) = matrix.identity(matrix.cols(qr.q))

  assert_matrix_close(reconstructed, original, tol)
  assert_matrix_close(qtq, identity, tol)
}

pub fn assert_residual_small(
  a: matrix.Matrix,
  x: vector.Vector,
  b: vector.Vector,
  tol: Float,
) -> Nil {
  let assert Ok(residual) = error_analysis.residual_norm2(a, x, b)
  assert residual <=. tol
}

pub fn assert_matrix_vector_consistent(
  a: matrix.Matrix,
  x: vector.Vector,
  expected: vector.Vector,
  tol: Float,
) -> Nil {
  let assert Ok(actual) = matrix.mul_vec(a, x)
  assert_vector_close(actual, expected, tol)
}

pub fn last_qr_step(
  steps: List(eigen.QrConvergenceStep),
) -> eigen.QrConvergenceStep {
  case steps {
    [step] -> step
    [_, ..rest] -> last_qr_step(rest)
    [] ->
      eigen.QrConvergenceStep(iteration: 0, shift: 0.0, off_diagonal_norm: 0.0)
  }
}

pub fn unsafe_vector_get(values: vector.Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
}

pub fn trace_of_rows(rows: List(List(Float))) -> Float {
  trace_loop(rows, 0, 0.0)
}

fn trace_loop(rows: List(List(Float)), index: Int, sum: Float) -> Float {
  case rows {
    [] -> sum
    [row, ..rest] -> trace_loop(rest, index + 1, sum +. list_at(row, index))
  }
}

fn list_at(values: List(Float), index: Int) -> Float {
  let #(left, right) = list.split(values, at: index)
  case right {
    [value, ..] -> value
    [] -> {
      let _ = left
      0.0
    }
  }
}
