import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, NotSquare, SingularMatrix,
}
import lumatrix/matrix.{type Matrix}
import lumatrix/vector.{type Vector}

const pivot_tolerance = 1.0e-12

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
    True -> {
      let pivot = matrix.unsafe_get(a, k, k)
      case float.absolute_value(pivot) <=. pivot_tolerance {
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

pub fn lu_factor(matrix a: Matrix) -> Result(LU, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let n = matrix.rows(a)
      let assert Ok(l) = matrix.identity(n)
      let assert Ok(p) = matrix.identity(n)
      lu_loop(0, n, a, l, p, 0)
    }
  }
}

pub fn complete_lu_factor(matrix a: Matrix) -> Result(CompleteLU, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let n = matrix.rows(a)
      let assert Ok(l) = matrix.identity(n)
      let assert Ok(p) = matrix.identity(n)
      let assert Ok(q) = matrix.identity(n)
      complete_lu_loop(0, n, a, l, p, q, 0)
    }
  }
}

pub fn cholesky_factor(matrix a: Matrix) -> Result(Cholesky, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case is_symmetric(a, pivot_tolerance) {
        False -> Error(InvalidInput("matrix must be symmetric"))
        True -> {
          let assert Ok(l) =
            matrix.zeros(rows: matrix.rows(a), cols: matrix.cols(a))
          cholesky_loop(a, l, 0, 0)
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
    True -> forward_loop(l, b, 0, [])
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
    True -> back_loop(u, b, matrix.rows(u) - 1, [])
  }
}

pub fn determinant(a: Matrix) -> Result(Float, NlaError) {
  case lu_factor(a) {
    Error(e) -> Error(e)
    Ok(factors) -> {
      let product =
        list.fold(matrix.indices(matrix.rows(factors.u)), 1.0, fn(acc, i) {
          acc *. matrix.unsafe_get(factors.u, i, i)
        })
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
    Ok(factors) -> {
      let product =
        list.fold(matrix.indices(matrix.rows(factors.u)), 1.0, fn(acc, i) {
          acc *. matrix.unsafe_get(factors.u, i, i)
        })
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
) -> Result(Cholesky, NlaError) {
  case i >= matrix.rows(a) {
    True -> Ok(Cholesky(l: l))
    False ->
      case j > i {
        True -> cholesky_loop(a, l, i + 1, 0)
        False -> cholesky_entry(a, l, i, j)
      }
  }
}

fn cholesky_entry(
  a: Matrix,
  l: Matrix,
  i: Int,
  j: Int,
) -> Result(Cholesky, NlaError) {
  let sum = cholesky_dot(l, i, j, 0, 0.0)
  case i == j {
    True -> {
      let value = matrix.unsafe_get(a, i, i) -. sum
      case value <=. pivot_tolerance {
        True -> Error(SingularMatrix(i))
        False ->
          case float.square_root(value) {
            Error(_) -> Error(SingularMatrix(i))
            Ok(root) -> {
              let assert Ok(next_l) = matrix.set(l, i, j, root)
              cholesky_loop(a, next_l, i, j + 1)
            }
          }
      }
    }
    False -> {
      let diagonal = matrix.unsafe_get(l, j, j)
      case float.absolute_value(diagonal) <=. pivot_tolerance {
        True -> Error(SingularMatrix(j))
        False -> {
          let value = { matrix.unsafe_get(a, i, j) -. sum } /. diagonal
          let assert Ok(next_l) = matrix.set(l, i, j, value)
          cholesky_loop(a, next_l, i, j + 1)
        }
      }
    }
  }
}

fn cholesky_dot(l: Matrix, i: Int, j: Int, k: Int, sum: Float) -> Float {
  case k >= j {
    True -> sum
    False ->
      cholesky_dot(
        l,
        i,
        j,
        k + 1,
        sum +. matrix.unsafe_get(l, i, k) *. matrix.unsafe_get(l, j, k),
      )
  }
}

fn lu_loop(
  k: Int,
  n: Int,
  u: Matrix,
  l: Matrix,
  p: Matrix,
  swaps: Int,
) -> Result(LU, NlaError) {
  case k >= n {
    True -> Ok(LU(l: l, u: u, p: p, swaps: swaps))
    False -> {
      let #(pivot, magnitude) = pivot_row(u, k, n)
      case magnitude <=. pivot_tolerance {
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
          let #(u, l) = eliminate_below(u, l, k, k + 1, n)
          lu_loop(k + 1, n, u, l, p, swaps)
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
) -> Result(CompleteLU, NlaError) {
  case k >= n {
    True -> Ok(CompleteLU(l: l, u: u, p: p, q: q, swaps: swaps))
    False -> {
      let #(pivot_i, pivot_j, magnitude) = complete_pivot(u, k, n)
      case magnitude <=. pivot_tolerance {
        True -> Error(SingularMatrix(k))
        False -> {
          let #(u, l, p, swaps) = swap_complete_rows(u, l, p, k, pivot_i, swaps)
          let #(u, q, swaps) = swap_complete_cols(u, q, k, pivot_j, swaps)
          let #(u, l) = eliminate_below(u, l, k, k + 1, n)
          complete_lu_loop(k + 1, n, u, l, p, q, swaps)
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
) -> #(Matrix, Matrix) {
  case i >= n {
    True -> #(u, l)
    False -> {
      let pivot = matrix.unsafe_get(u, k, k)
      let factor = matrix.unsafe_get(u, i, k) /. pivot
      let assert Ok(next_l) = matrix.set(l, i, k, factor)
      let next_u =
        list.fold(list.drop(matrix.indices(n), up_to: k), u, fn(acc, j) {
          let updated =
            matrix.unsafe_get(acc, i, j)
            -. factor
            *. matrix.unsafe_get(acc, k, j)
          let assert Ok(next) = matrix.set(acc, i, j, updated)
          next
        })
      eliminate_below(next_u, next_l, k, i + 1, n)
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
  list.all(matrix.indices(matrix.rows(a)), satisfying: fn(i) {
    list.all(matrix.indices(i), satisfying: fn(j) {
      float.absolute_value(
        matrix.unsafe_get(a, i, j) -. matrix.unsafe_get(a, j, i),
      )
      <=. tolerance
    })
  })
}

fn forward_loop(
  l: Matrix,
  b: Vector,
  i: Int,
  solved: List(Float),
) -> Result(Vector, NlaError) {
  case i >= matrix.rows(l) {
    True -> Ok(vector.from_list(solved))
    False -> {
      let diagonal = matrix.unsafe_get(l, i, i)
      case float.absolute_value(diagonal) <=. pivot_tolerance {
        True -> Error(SingularMatrix(i))
        False -> {
          let rhs = unsafe_vector_get(b, i)
          let lower_sum =
            list.fold(matrix.indices(i), 0.0, fn(acc, j) {
              acc +. matrix.unsafe_get(l, i, j) *. unsafe_at(solved, j)
            })
          let x = { rhs -. lower_sum } /. diagonal
          forward_loop(l, b, i + 1, list.append(solved, [x]))
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
) -> Result(Vector, NlaError) {
  case i < 0 {
    True -> Ok(vector.from_list(solved_tail))
    False -> {
      let diagonal = matrix.unsafe_get(u, i, i)
      case float.absolute_value(diagonal) <=. pivot_tolerance {
        True -> Error(SingularMatrix(i))
        False -> {
          let rhs = unsafe_vector_get(b, i)
          let upper_sum =
            list.fold(
              list.drop(matrix.indices(matrix.rows(u)), up_to: i + 1),
              0.0,
              fn(acc, j) {
                acc
                +. matrix.unsafe_get(u, i, j)
                *. unsafe_at(solved_tail, j - i - 1)
              },
            )
          let x = { rhs -. upper_sum } /. diagonal
          back_loop(u, b, i - 1, [x, ..solved_tail])
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

fn unsafe_vector_get(values: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
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
