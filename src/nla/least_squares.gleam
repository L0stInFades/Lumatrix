import gleam/int
import gleam/list
import nla/direct
import nla/error.{type NlaError, DimensionMismatch}
import nla/error_analysis
import nla/matrix.{type Matrix}
import nla/orthogonal
import nla/vector.{type Vector}

pub type LeastSquaresSolution {
  LeastSquaresSolution(
    solution: Vector,
    residual_norm: Float,
    normal_matrix_condition_inf: Float,
  )
}

pub type LeastSquaresDiagnostics {
  LeastSquaresDiagnostics(
    residual_norm: Float,
    relative_residual: Float,
    normal_matrix_condition_inf: Float,
    normal_equation_residual_norm: Float,
  )
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
                Ok(x) -> finish_solution(a, b, ata, x)
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
    Ok(_) -> {
      case orthogonal.qr_householder(a) {
        Error(e) -> Error(e)
        Ok(qr) -> {
          let qt = matrix.transpose(qr.q)
          case matrix.mul_vec(qt, b) {
            Error(e) -> Error(e)
            Ok(qtb) -> {
              let r1 = leading_square(qr.r, a.cols)
              let c1 = leading_vector(qtb, a.cols)
              case direct.back_substitution(r1, c1) {
                Error(e) -> Error(e)
                Ok(x) -> {
                  let at = matrix.transpose(a)
                  case matrix.mul(at, a) {
                    Error(e) -> Error(e)
                    Ok(ata) -> finish_solution(a, b, ata, x)
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

pub fn classical_gram_schmidt_qr(
  a: Matrix,
  b: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case validate_least_squares_system(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case orthogonal.qr_classical_gram_schmidt(a) {
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
      case orthogonal.qr_modified_gram_schmidt(a) {
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
          let at = matrix.transpose(a)
          case matrix.mul(at, a) {
            Error(e) -> Error(e)
            Ok(ata) -> finish_solution(a, b, ata, x)
          }
        }
      }
  }
}

fn finish_solution(
  a: Matrix,
  b: Vector,
  normal_matrix: Matrix,
  x: Vector,
) -> Result(LeastSquaresSolution, NlaError) {
  case residual_norm(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) ->
      case error_analysis.condition_number_inf(normal_matrix) {
        Error(e) -> Error(e)
        Ok(condition) ->
          Ok(LeastSquaresSolution(
            solution: x,
            residual_norm: r_norm,
            normal_matrix_condition_inf: condition,
          ))
      }
  }
}

fn validate_least_squares_system(
  a: Matrix,
  b: Vector,
) -> Result(Nil, NlaError) {
  case a.rows == b.size && a.rows >= a.cols {
    True -> Ok(Nil)
    False ->
      Error(DimensionMismatch(
        expected: "m >= n and b length m",
        actual: int.to_string(a.rows)
          <> "x"
          <> int.to_string(a.cols)
          <> ", b="
          <> int.to_string(b.size),
      ))
  }
}

fn validate_least_squares_solution(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(Nil, NlaError) {
  case a.rows == b.size && a.cols == x.size && a.rows >= a.cols {
    True -> Ok(Nil)
    False ->
      Error(DimensionMismatch(
        expected: "m >= n, b length m and x length n",
        actual: int.to_string(a.rows)
          <> "x"
          <> int.to_string(a.cols)
          <> ", b="
          <> int.to_string(b.size)
          <> ", x="
          <> int.to_string(x.size),
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
  let #(left, right) = list.split(x.data, at: index)
  case right {
    [value, ..] -> value
    [] -> {
      let _ = left
      0.0
    }
  }
}
