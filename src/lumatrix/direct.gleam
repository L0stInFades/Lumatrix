import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, DimensionMismatch, InvalidInput,
  NonFiniteInput, NotSquare, SingularMatrix,
}
import lumatrix/matrix.{type Matrix}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

const pivot_tolerance = 1.0e-14

const symmetry_tolerance = 1.0e-12

pub type LU {
  LU(l: Matrix, u: Matrix, p: Matrix, swaps: Int)
}

pub type CompleteLU {
  CompleteLU(l: Matrix, u: Matrix, p: Matrix, q: Matrix, swaps: Int)
}

pub type Cholesky {
  Cholesky(l: Matrix)
}

pub fn gauss_transform(
  matrix a: Matrix,
  pivot k: Int,
  row i: Int,
) -> Result(Matrix, NlaError) {
  case
    matrix.is_square(a)
    && k >= 0
    && k < matrix.rows(a)
    && i >= 0
    && i < matrix.rows(a)
  {
    False -> Error(InvalidInput("invalid Gauss transform indices"))
    True ->
      case matrix.is_finite(a) {
        False -> Error(NonFiniteInput("Gauss transform matrix"))
        True -> {
          let pivot = matrix.unsafe_get(a, k, k)
          case
            singular_magnitude(float.absolute_value(pivot), matrix.norm_inf(a))
          {
            True -> Error(SingularMatrix(k))
            False -> {
              let factor = 0.0 -. matrix.unsafe_get(a, i, k) /. pivot
              matrix.from_fn(
                rows: matrix.rows(a),
                cols: matrix.cols(a),
                with: fn(row, col) {
                  case row == col {
                    True -> 1.0
                    False ->
                      case row == i && col == k {
                        True -> factor
                        False -> 0.0
                      }
                  }
                },
              )
            }
          }
        }
      }
  }
}

pub fn lu_factor(matrix a: Matrix) -> Result(LU, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.is_finite(a) {
        False -> Error(NonFiniteInput("LU matrix"))
        True -> {
          let n = matrix.rows(a)
          let assert Ok(l) = matrix.identity(n)
          let assert Ok(p) = matrix.identity(n)
          lu_loop(0, n, a, l, p, 0, matrix.norm_inf(a))
        }
      }
  }
}

pub fn complete_lu_factor(matrix a: Matrix) -> Result(CompleteLU, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.is_finite(a) {
        False -> Error(NonFiniteInput("complete-pivoting LU matrix"))
        True -> {
          let n = matrix.rows(a)
          let assert Ok(l) = matrix.identity(n)
          let assert Ok(p) = matrix.identity(n)
          let assert Ok(q) = matrix.identity(n)
          complete_lu_loop(0, n, a, l, p, q, 0, matrix.norm_inf(a))
        }
      }
  }
}

pub fn cholesky_factor(matrix a: Matrix) -> Result(Cholesky, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.is_finite(a) {
        False -> Error(NonFiniteInput("Cholesky matrix"))
        True ->
          case is_symmetric(a, symmetry_tolerance) {
            False -> Error(InvalidInput("matrix must be symmetric"))
            True -> {
              let assert Ok(l) =
                matrix.zeros(rows: matrix.rows(a), cols: matrix.cols(a))
              cholesky_loop(a, l, 0, 0, matrix.norm_inf(a))
            }
          }
      }
  }
}

pub fn solve(a: Matrix, b: Vector) -> Result(Vector, NlaError) {
  case lu_factor(a) {
    Ok(factors) -> lu_solve(factors, b)
    Error(e) -> Error(e)
  }
}

pub fn solve_complete_pivoting(
  a: Matrix,
  b: Vector,
) -> Result(Vector, NlaError) {
  case complete_lu_factor(a) {
    Ok(factors) -> complete_lu_solve(factors, b)
    Error(e) -> Error(e)
  }
}

pub fn gaussian_elimination(a: Matrix, b: Vector) -> Result(Vector, NlaError) {
  solve(a, b)
}

pub fn solve_spd(a: Matrix, b: Vector) -> Result(Vector, NlaError) {
  case cholesky_factor(a) {
    Error(e) -> Error(e)
    Ok(factors) -> cholesky_solve(factors, b)
  }
}

pub fn cholesky_solve(
  factors: Cholesky,
  b: Vector,
) -> Result(Vector, NlaError) {
  case matrix.rows(factors.l) == vector.dimension(b) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows(factors.l)),
        actual: int.to_string(vector.dimension(b)),
      ))
    True ->
      case forward_substitution(factors.l, b) {
        Error(e) -> Error(e)
        Ok(y) -> back_substitution(matrix.transpose(factors.l), y)
      }
  }
}

pub fn lu_solve(factors: LU, b: Vector) -> Result(Vector, NlaError) {
  case matrix.rows(factors.l) == vector.dimension(b) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows(factors.l)),
        actual: int.to_string(vector.dimension(b)),
      ))
    True -> {
      case matrix.mul_vec(factors.p, b) {
        Error(e) -> Error(e)
        Ok(pb) ->
          case forward_substitution(factors.l, pb) {
            Error(e) -> Error(e)
            Ok(y) -> back_substitution(factors.u, y)
          }
      }
    }
  }
}

pub fn complete_lu_solve(
  factors: CompleteLU,
  b: Vector,
) -> Result(Vector, NlaError) {
  case matrix.rows(factors.l) == vector.dimension(b) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows(factors.l)),
        actual: int.to_string(vector.dimension(b)),
      ))
    True ->
      case matrix.mul_vec(factors.p, b) {
        Error(e) -> Error(e)
        Ok(pb) ->
          case forward_substitution(factors.l, pb) {
            Error(e) -> Error(e)
            Ok(y) ->
              case back_substitution(factors.u, y) {
                Error(e) -> Error(e)
                Ok(permuted_x) -> matrix.mul_vec(factors.q, permuted_x)
              }
          }
      }
  }
}

pub fn forward_substitution(l: Matrix, b: Vector) -> Result(Vector, NlaError) {
  case
    matrix.rows(l) == matrix.cols(l) && matrix.rows(l) == vector.dimension(b)
  {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows(l))
          <> "x"
          <> int.to_string(matrix.cols(l)),
        actual: int.to_string(vector.dimension(b)),
      ))
    True ->
      case matrix.is_finite(l) && vector.is_finite(b) {
        False -> Error(NonFiniteInput("forward substitution system"))
        True -> forward_loop(l, b, 0, [], matrix.norm_inf(l))
      }
  }
}

pub fn back_substitution(u: Matrix, b: Vector) -> Result(Vector, NlaError) {
  case
    matrix.rows(u) == matrix.cols(u) && matrix.rows(u) == vector.dimension(b)
  {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows(u))
          <> "x"
          <> int.to_string(matrix.cols(u)),
        actual: int.to_string(vector.dimension(b)),
      ))
    True ->
      case matrix.is_finite(u) && vector.is_finite(b) {
        False -> Error(NonFiniteInput("back substitution system"))
        True -> back_loop(u, b, matrix.rows(u) - 1, [], matrix.norm_inf(u))
      }
  }
}

pub fn determinant(a: Matrix) -> Result(Float, NlaError) {
  case lu_factor(a) {
    Error(e) -> Error(e)
    Ok(factors) ->
      case checked_diagonal_product(factors.u) {
        Error(e) -> Error(e)
        Ok(product) ->
          case factors.swaps % 2 == 0 {
            True -> Ok(product)
            False -> Ok(0.0 -. product)
          }
      }
  }
}

pub fn determinant_complete_pivoting(a: Matrix) -> Result(Float, NlaError) {
  case complete_lu_factor(a) {
    Error(e) -> Error(e)
    Ok(factors) ->
      case checked_diagonal_product(factors.u) {
        Error(e) -> Error(e)
        Ok(product) ->
          case factors.swaps % 2 == 0 {
            True -> Ok(product)
            False -> Ok(0.0 -. product)
          }
      }
  }
}

pub fn inverse(a: Matrix) -> Result(Matrix, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      case lu_factor(a) {
        Error(e) -> Error(e)
        Ok(factors) -> inverse_columns(factors, 0, matrix.rows(a), [])
      }
    }
  }
}

pub fn inverse_complete_pivoting(a: Matrix) -> Result(Matrix, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case complete_lu_factor(a) {
        Error(e) -> Error(e)
        Ok(factors) -> inverse_complete_columns(factors, 0, matrix.rows(a), [])
      }
  }
}

fn cholesky_loop(
  a: Matrix,
  l: Matrix,
  i: Int,
  j: Int,
  scale: Float,
) -> Result(Cholesky, NlaError) {
  case i >= matrix.rows(a) {
    True -> Ok(Cholesky(l: l))
    False ->
      case j > i {
        True -> cholesky_loop(a, l, i + 1, 0, scale)
        False -> cholesky_entry(a, l, i, j, scale)
      }
  }
}

fn cholesky_entry(
  a: Matrix,
  l: Matrix,
  i: Int,
  j: Int,
  scale: Float,
) -> Result(Cholesky, NlaError) {
  case cholesky_dot(l, i, j) {
    Error(e) -> Error(e)
    Ok(sum) ->
      case i == j {
        True ->
          case numerics.checked_subtract(matrix.unsafe_get(a, i, i), sum) {
            Error(_) -> Error(ArithmeticOverflow("Cholesky diagonal update"))
            Ok(value) ->
              case value <=. 0.0 || singular_magnitude(value, scale) {
                True -> Error(SingularMatrix(i))
                False ->
                  case float.square_root(value) {
                    Error(_) -> Error(SingularMatrix(i))
                    Ok(root) ->
                      case matrix.set(l, i, j, root) {
                        Error(e) -> Error(e)
                        Ok(next_l) -> cholesky_loop(a, next_l, i, j + 1, scale)
                      }
                  }
              }
          }
        False -> {
          let diagonal = matrix.unsafe_get(l, j, j)
          case singular_magnitude(diagonal *. diagonal, scale) {
            True -> Error(SingularMatrix(j))
            False ->
              case numerics.checked_subtract(matrix.unsafe_get(a, i, j), sum) {
                Error(_) ->
                  Error(ArithmeticOverflow("Cholesky off-diagonal update"))
                Ok(numerator) ->
                  case numerics.checked_divide(numerator, diagonal) {
                    Error(_) -> Error(ArithmeticOverflow("Cholesky division"))
                    Ok(value) ->
                      case matrix.set(l, i, j, value) {
                        Error(e) -> Error(e)
                        Ok(next_l) -> cholesky_loop(a, next_l, i, j + 1, scale)
                      }
                  }
              }
          }
        }
      }
  }
}

fn cholesky_dot(l: Matrix, i: Int, j: Int) -> Result(Float, NlaError) {
  case
    numerics.checked_dot_pairs(
      list.map(matrix.indices(j), fn(k) {
        #(matrix.unsafe_get(l, i, k), matrix.unsafe_get(l, j, k))
      }),
    )
  {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow("Cholesky inner product"))
  }
}

fn lu_loop(
  k: Int,
  n: Int,
  u: Matrix,
  l: Matrix,
  p: Matrix,
  swaps: Int,
  scale: Float,
) -> Result(LU, NlaError) {
  case k >= n {
    True -> Ok(LU(l: l, u: u, p: p, swaps: swaps))
    False -> {
      let #(pivot, magnitude) = pivot_row(u, k, n)
      case singular_magnitude(magnitude, scale) {
        True -> Error(SingularMatrix(k))
        False -> {
          let #(u, l, p, swaps) = case pivot == k {
            True -> #(u, l, p, swaps)
            False -> {
              let assert Ok(next_u) = matrix.swap_rows(u, k, pivot)
              let next_l = swap_l_prefix(l, k, pivot, k)
              let assert Ok(next_p) = matrix.swap_rows(p, k, pivot)
              #(next_u, next_l, next_p, swaps + 1)
            }
          }
          case eliminate_below(u, l, k, k + 1, n) {
            Error(e) -> Error(e)
            Ok(#(u, l)) -> lu_loop(k + 1, n, u, l, p, swaps, scale)
          }
        }
      }
    }
  }
}

fn complete_lu_loop(
  k: Int,
  n: Int,
  u: Matrix,
  l: Matrix,
  p: Matrix,
  q: Matrix,
  swaps: Int,
  scale: Float,
) -> Result(CompleteLU, NlaError) {
  case k >= n {
    True -> Ok(CompleteLU(l: l, u: u, p: p, q: q, swaps: swaps))
    False -> {
      let #(pivot_i, pivot_j, magnitude) = complete_pivot(u, k, n)
      case singular_magnitude(magnitude, scale) {
        True -> Error(SingularMatrix(k))
        False -> {
          let #(u, l, p, swaps) = swap_complete_rows(u, l, p, k, pivot_i, swaps)
          let #(u, q, swaps) = swap_complete_cols(u, q, k, pivot_j, swaps)
          case eliminate_below(u, l, k, k + 1, n) {
            Error(e) -> Error(e)
            Ok(#(u, l)) -> complete_lu_loop(k + 1, n, u, l, p, q, swaps, scale)
          }
        }
      }
    }
  }
}

fn pivot_row(u: Matrix, k: Int, n: Int) -> #(Int, Float) {
  list.fold(list.drop(matrix.indices(n), up_to: k), #(k, 0.0), fn(best, i) {
    let value = float.absolute_value(matrix.unsafe_get(u, i, k))
    case value >. best.1 {
      True -> #(i, value)
      False -> best
    }
  })
}

fn complete_pivot(u: Matrix, k: Int, n: Int) -> #(Int, Int, Float) {
  list.fold(list.drop(matrix.indices(n), up_to: k), #(k, k, 0.0), fn(best, i) {
    list.fold(list.drop(matrix.indices(n), up_to: k), best, fn(inner_best, j) {
      let value = float.absolute_value(matrix.unsafe_get(u, i, j))
      case value >. inner_best.2 {
        True -> #(i, j, value)
        False -> inner_best
      }
    })
  })
}

fn swap_complete_rows(
  u: Matrix,
  l: Matrix,
  p: Matrix,
  k: Int,
  pivot: Int,
  swaps: Int,
) -> #(Matrix, Matrix, Matrix, Int) {
  case pivot == k {
    True -> #(u, l, p, swaps)
    False -> {
      let assert Ok(next_u) = matrix.swap_rows(u, k, pivot)
      let next_l = swap_l_prefix(l, k, pivot, k)
      let assert Ok(next_p) = matrix.swap_rows(p, k, pivot)
      #(next_u, next_l, next_p, swaps + 1)
    }
  }
}

fn swap_complete_cols(
  u: Matrix,
  q: Matrix,
  k: Int,
  pivot: Int,
  swaps: Int,
) -> #(Matrix, Matrix, Int) {
  case pivot == k {
    True -> #(u, q, swaps)
    False -> {
      let next_u = swap_columns(u, k, pivot)
      let next_q = swap_columns(q, k, pivot)
      #(next_u, next_q, swaps + 1)
    }
  }
}

fn eliminate_below(
  u: Matrix,
  l: Matrix,
  k: Int,
  i: Int,
  n: Int,
) -> Result(#(Matrix, Matrix), NlaError) {
  case i >= n {
    True -> Ok(#(u, l))
    False -> {
      let pivot = matrix.unsafe_get(u, k, k)
      case numerics.checked_divide(matrix.unsafe_get(u, i, k), pivot) {
        Error(_) -> Error(ArithmeticOverflow("LU elimination factor"))
        Ok(factor) ->
          case matrix.set(l, i, k, factor) {
            Error(e) -> Error(e)
            Ok(next_l) ->
              case eliminate_row(u, i, k, k, n, factor) {
                Error(e) -> Error(e)
                Ok(next_u) -> eliminate_below(next_u, next_l, k, i + 1, n)
              }
          }
      }
    }
  }
}

fn eliminate_row(
  u: Matrix,
  row: Int,
  pivot_row: Int,
  col: Int,
  n: Int,
  factor: Float,
) -> Result(Matrix, NlaError) {
  case col >= n {
    True -> Ok(u)
    False ->
      case
        numerics.checked_multiply(factor, matrix.unsafe_get(u, pivot_row, col))
      {
        Error(_) -> Error(ArithmeticOverflow("LU row update"))
        Ok(product) ->
          case
            numerics.checked_subtract(matrix.unsafe_get(u, row, col), product)
          {
            Error(_) -> Error(ArithmeticOverflow("LU row update"))
            Ok(updated) ->
              case matrix.set(u, row, col, updated) {
                Error(e) -> Error(e)
                Ok(next) ->
                  eliminate_row(next, row, pivot_row, col + 1, n, factor)
              }
          }
      }
  }
}

fn swap_l_prefix(l: Matrix, a: Int, b: Int, width: Int) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: matrix.rows(l), cols: matrix.cols(l), with: fn(i, j) {
      case j < width {
        True ->
          case i == a {
            True -> matrix.unsafe_get(l, b, j)
            False ->
              case i == b {
                True -> matrix.unsafe_get(l, a, j)
                False -> matrix.unsafe_get(l, i, j)
              }
          }
        False -> matrix.unsafe_get(l, i, j)
      }
    })
  result
}

fn swap_columns(a: Matrix, left: Int, right: Int) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: matrix.rows(a), cols: matrix.cols(a), with: fn(i, j) {
      case j == left {
        True -> matrix.unsafe_get(a, i, right)
        False ->
          case j == right {
            True -> matrix.unsafe_get(a, i, left)
            False -> matrix.unsafe_get(a, i, j)
          }
      }
    })
  result
}

fn is_symmetric(a: Matrix, tolerance: Float) -> Bool {
  let scale = matrix.norm_inf(a)
  list.all(matrix.indices(matrix.rows(a)), satisfying: fn(i) {
    list.all(matrix.indices(i), satisfying: fn(j) {
      numerics.relative_close_at_scale(
        matrix.unsafe_get(a, i, j),
        matrix.unsafe_get(a, j, i),
        scale,
        tolerance,
      )
    })
  })
}

fn forward_loop(
  l: Matrix,
  b: Vector,
  i: Int,
  solved: List(Float),
  scale: Float,
) -> Result(Vector, NlaError) {
  case i >= matrix.rows(l) {
    True -> Ok(vector.from_list(solved))
    False -> {
      let diagonal = matrix.unsafe_get(l, i, i)
      case singular_magnitude(float.absolute_value(diagonal), scale) {
        True -> Error(SingularMatrix(i))
        False -> {
          let rhs = unsafe_vector_get(b, i)
          let pairs =
            list.map(matrix.indices(i), fn(j) {
              #(matrix.unsafe_get(l, i, j), unsafe_at(solved, j))
            })
          case checked_substitution_value(rhs, pairs, diagonal, "forward") {
            Error(e) -> Error(e)
            Ok(x) -> forward_loop(l, b, i + 1, list.append(solved, [x]), scale)
          }
        }
      }
    }
  }
}

fn back_loop(
  u: Matrix,
  b: Vector,
  i: Int,
  solved_tail: List(Float),
  scale: Float,
) -> Result(Vector, NlaError) {
  case i < 0 {
    True -> Ok(vector.from_list(solved_tail))
    False -> {
      let diagonal = matrix.unsafe_get(u, i, i)
      case singular_magnitude(float.absolute_value(diagonal), scale) {
        True -> Error(SingularMatrix(i))
        False -> {
          let rhs = unsafe_vector_get(b, i)
          let pairs =
            list.map(
              list.drop(matrix.indices(matrix.rows(u)), up_to: i + 1),
              fn(j) {
                #(matrix.unsafe_get(u, i, j), unsafe_at(solved_tail, j - i - 1))
              },
            )
          case checked_substitution_value(rhs, pairs, diagonal, "back") {
            Error(e) -> Error(e)
            Ok(x) -> back_loop(u, b, i - 1, [x, ..solved_tail], scale)
          }
        }
      }
    }
  }
}

fn inverse_columns(
  factors: LU,
  i: Int,
  n: Int,
  columns: List(Vector),
) -> Result(Matrix, NlaError) {
  case i >= n {
    True -> {
      let columns = list.reverse(columns)
      matrix.from_fn(rows: n, cols: n, with: fn(row, col) {
        let column = unsafe_vector_at(columns, col)
        unsafe_vector_get(column, row)
      })
    }
    False -> {
      case vector.basis(n, i) {
        Error(e) -> Error(e)
        Ok(e_i) ->
          case lu_solve(factors, e_i) {
            Error(e) -> Error(e)
            Ok(column) ->
              inverse_columns(factors, i + 1, n, [column, ..columns])
          }
      }
    }
  }
}

fn inverse_complete_columns(
  factors: CompleteLU,
  i: Int,
  n: Int,
  columns: List(Vector),
) -> Result(Matrix, NlaError) {
  case i >= n {
    True -> {
      let columns = list.reverse(columns)
      matrix.from_fn(rows: n, cols: n, with: fn(row, col) {
        let column = unsafe_vector_at(columns, col)
        unsafe_vector_get(column, row)
      })
    }
    False ->
      case vector.basis(n, i) {
        Error(e) -> Error(e)
        Ok(e_i) ->
          case complete_lu_solve(factors, e_i) {
            Error(e) -> Error(e)
            Ok(column) ->
              inverse_complete_columns(factors, i + 1, n, [column, ..columns])
          }
      }
  }
}

fn checked_diagonal_product(a: Matrix) -> Result(Float, NlaError) {
  list.try_fold(
    over: matrix.indices(matrix.rows(a)),
    from: 1.0,
    with: fn(product, index) {
      case
        numerics.checked_multiply(product, matrix.unsafe_get(a, index, index))
      {
        Ok(value) -> Ok(value)
        Error(_) -> Error(ArithmeticOverflow("determinant"))
      }
    },
  )
}

fn checked_substitution_value(
  rhs: Float,
  pairs: List(#(Float, Float)),
  diagonal: Float,
  direction: String,
) -> Result(Float, NlaError) {
  case numerics.checked_dot_pairs(pairs) {
    Error(_) -> Error(ArithmeticOverflow(direction <> " substitution sum"))
    Ok(sum) ->
      case numerics.checked_subtract(rhs, sum) {
        Error(_) ->
          Error(ArithmeticOverflow(direction <> " substitution update"))
        Ok(numerator) ->
          case numerics.checked_divide(numerator, diagonal) {
            Ok(value) -> Ok(value)
            Error(_) ->
              Error(ArithmeticOverflow(direction <> " substitution division"))
          }
      }
  }
}

fn unsafe_vector_get(values: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
}

fn singular_magnitude(magnitude: Float, scale: Float) -> Bool {
  numerics.relative_near_zero(magnitude, scale, pivot_tolerance)
}

fn unsafe_vector_at(vectors: List(Vector), index: Int) -> Vector {
  let #(left, right) = list.split(vectors, at: index)
  case right {
    [value, ..] -> value
    [] -> {
      let _ = left
      vector.from_list([])
    }
  }
}

fn unsafe_at(data: List(Float), index: Int) -> Float {
  let #(left, right) = list.split(data, at: index)
  case right {
    [value, ..] -> value
    [] -> {
      let _ = left
      0.0
    }
  }
}
