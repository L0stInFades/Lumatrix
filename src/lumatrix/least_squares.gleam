import gleam/int
import gleam/list
import lumatrix/direct
import lumatrix/error.{type NlaError, DimensionMismatch}
import lumatrix/error_analysis
import lumatrix/matrix.{type Matrix}
import lumatrix/orthogonal
import lumatrix/vector.{type Vector}

/// Core least-squares solve output.
///
/// Use `stability_diagnostics` when condition numbers or normal-equation
/// residuals are needed.
pub type LeastSquaresSolution {
  LeastSquaresSolution(solution: Vector, residual_norm: Float)
}

/// Diagnostic quantities for a least-squares solution.
pub type LeastSquaresDiagnostics {
  LeastSquaresDiagnostics(
    residual_norm: Float,
    relative_residual: Float,
    normal_matrix_condition_inf: Float,
    normal_equation_residual_norm: Float,
  )
}

pub fn solve(a: Matrix, b: Vector) -> Result(LeastSquaresSolution, NlaError) {
  householder_qr(a, b)
}

pub fn normal_equations(
  a: Matrix,
  b: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case validate_least_squares_system(a, b) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let at = matrix.transpose(a)
      case matrix.mul(at, a) {
        Error(e) -> Error(e)
        Ok(ata) ->
          case matrix.mul_vec(at, b) {
            Error(e) -> Error(e)
            Ok(atb) ->
              case direct.solve(ata, atb) {
                Error(e) -> Error(e)
                Ok(x) -> finish_solution(a, b, x)
              }
          }
      }
    }
  }
}

pub fn householder_qr(
  a: Matrix,
  b: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case validate_least_squares_system(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case orthogonal.householder_qr(a) {
        Error(e) -> Error(e)
        Ok(qr) -> solve_full_qr(a, b, qr)
      }
  }
}

pub fn givens_qr(
  a: Matrix,
  b: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case validate_least_squares_system(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case orthogonal.givens_qr(a) {
        Error(e) -> Error(e)
        Ok(qr) -> solve_full_qr(a, b, qr)
      }
  }
}

pub fn classical_gram_schmidt_qr(
  a: Matrix,
  b: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case validate_least_squares_system(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case orthogonal.classical_gram_schmidt_qr(a) {
        Error(e) -> Error(e)
        Ok(qr) -> solve_thin_qr(a, b, qr)
      }
  }
}

pub fn modified_gram_schmidt_qr(
  a: Matrix,
  b: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case validate_least_squares_system(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case orthogonal.modified_gram_schmidt_qr(a) {
        Error(e) -> Error(e)
        Ok(qr) -> solve_thin_qr(a, b, qr)
      }
  }
}

pub fn residual_norm(
  a: Matrix,
  x: Vector,
  b: Vector,
) -> Result(Float, NlaError) {
  error_analysis.residual_norm2(a, x, b)
}

pub fn stability_diagnostics(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(LeastSquaresDiagnostics, NlaError) {
  case validate_least_squares_solution(a, b, x) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let at = matrix.transpose(a)
      case error_analysis.residual(a, x, b) {
        Error(e) -> Error(e)
        Ok(r) ->
          case vector.norm2(r) {
            Error(e) -> Error(e)
            Ok(r_norm) ->
              case error_analysis.normwise_relative_residual(a, x, b) {
                Error(e) -> Error(e)
                Ok(relative_residual) ->
                  case matrix.mul(at, a) {
                    Error(e) -> Error(e)
                    Ok(ata) ->
                      case error_analysis.condition_number_inf(ata) {
                        Error(e) -> Error(e)
                        Ok(condition) ->
                          case matrix.mul_vec(at, r) {
                            Error(e) -> Error(e)
                            Ok(normal_residual) ->
                              case vector.norm2(normal_residual) {
                                Error(e) -> Error(e)
                                Ok(normal_residual_norm) ->
                                  Ok(LeastSquaresDiagnostics(
                                    residual_norm: r_norm,
                                    relative_residual: relative_residual,
                                    normal_matrix_condition_inf: condition,
                                    normal_equation_residual_norm: normal_residual_norm,
                                  ))
                              }
                          }
                      }
                  }
              }
          }
      }
    }
  }
}

fn solve_full_qr(
  a: Matrix,
  b: Vector,
  qr: orthogonal.QR,
) -> Result(LeastSquaresSolution, NlaError) {
  let qt = matrix.transpose(qr.q)
  case matrix.mul_vec(qt, b) {
    Error(e) -> Error(e)
    Ok(qtb) -> {
      let r1 = leading_square(qr.r, matrix.cols(a))
      let c1 = leading_vector(qtb, matrix.cols(a))
      case direct.back_substitution(r1, c1) {
        Error(e) -> Error(e)
        Ok(x) -> {
          finish_solution(a, b, x)
        }
      }
    }
  }
}

fn solve_thin_qr(
  a: Matrix,
  b: Vector,
  qr: orthogonal.QR,
) -> Result(LeastSquaresSolution, NlaError) {
  let qt = matrix.transpose(qr.q)
  case matrix.mul_vec(qt, b) {
    Error(e) -> Error(e)
    Ok(qtb) ->
      case direct.back_substitution(qr.r, qtb) {
        Error(e) -> Error(e)
        Ok(x) -> {
          finish_solution(a, b, x)
        }
      }
  }
}

fn finish_solution(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case residual_norm(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) -> Ok(LeastSquaresSolution(solution: x, residual_norm: r_norm))
  }
}

fn validate_least_squares_system(
  a: Matrix,
  b: Vector,
) -> Result(Nil, NlaError) {
  case
    matrix.rows(a) == vector.dimension(b) && matrix.rows(a) >= matrix.cols(a)
  {
    True -> Ok(Nil)
    False ->
      Error(DimensionMismatch(
        expected: "m >= n and b length m",
        actual: int.to_string(matrix.rows(a))
          <> "x"
          <> int.to_string(matrix.cols(a))
          <> ", b="
          <> int.to_string(vector.dimension(b)),
      ))
  }
}

fn validate_least_squares_solution(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(Nil, NlaError) {
  case
    matrix.rows(a) == vector.dimension(b)
    && matrix.cols(a) == vector.dimension(x)
    && matrix.rows(a) >= matrix.cols(a)
  {
    True -> Ok(Nil)
    False ->
      Error(DimensionMismatch(
        expected: "m >= n, b length m and x length n",
        actual: int.to_string(matrix.rows(a))
          <> "x"
          <> int.to_string(matrix.cols(a))
          <> ", b="
          <> int.to_string(vector.dimension(b))
          <> ", x="
          <> int.to_string(vector.dimension(x)),
      ))
  }
}

fn leading_square(a: Matrix, size: Int) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: size, cols: size, with: fn(i, j) {
      matrix.unsafe_get(a, i, j)
    })
  result
}

fn leading_vector(x: Vector, size: Int) -> Vector {
  vector.from_list(
    list.map(matrix.indices(size), fn(i) { unsafe_vector_get(x, i) }),
  )
}

fn unsafe_vector_get(x: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(x, index)
  value
}
