import gleam/float
import gleam/int
import gleam/list
import nla/error.{
  type NlaError, DimensionMismatch, InvalidInput, NotSquare, SingularMatrix,
}
import nla/matrix.{type Matrix}
import nla/vector.{type Vector}

const pivot_tolerance = 1.0e-12

pub type LU {
  LU(l: Matrix, u: Matrix, p: Matrix, swaps: Int)
}

pub type Cholesky {
  Cholesky(l: Matrix)
}

pub fn gauss_transform(
  matrix a: Matrix,
  pivot k: Int,
  row i: Int,
) -> Result(Matrix, NlaError) {
  case matrix.is_square(a) && k >= 0 && k < a.rows && i >= 0 && i < a.rows {
    False -> Error(InvalidInput("invalid Gauss transform indices"))
    True -> {
      let pivot = matrix.unsafe_get(a, k, k)
      case float.absolute_value(pivot) <=. pivot_tolerance {
        True -> Error(SingularMatrix(k))
        False -> {
          let factor = 0.0 -. matrix.unsafe_get(a, i, k) /. pivot
          matrix.from_fn(rows: a.rows, cols: a.cols, with: fn(row, col) {
            case row == col {
              True -> 1.0
              False ->
                case row == i && col == k {
                  True -> factor
                  False -> 0.0
                }
            }
          })
        }
      }
    }
  }
}

pub fn lu_factor(matrix a: Matrix) -> Result(LU, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(a.rows, a.cols))
    True -> {
      let n = a.rows
      let assert Ok(l) = matrix.identity(n)
      let assert Ok(p) = matrix.identity(n)
      lu_loop(0, n, a, l, p, 0)
    }
  }
}

pub fn cholesky_factor(matrix a: Matrix) -> Result(Cholesky, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(a.rows, a.cols))
    True ->
      case is_symmetric(a, pivot_tolerance) {
        False -> Error(InvalidInput("matrix must be symmetric"))
        True -> {
          let assert Ok(l) = matrix.zeros(rows: a.rows, cols: a.cols)
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
  case factors.l.rows == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(factors.l.rows),
        actual: int.to_string(b.size),
      ))
    True ->
      case forward_substitution(factors.l, b) {
        Error(e) -> Error(e)
        Ok(y) -> back_substitution(matrix.transpose(factors.l), y)
      }
  }
}

pub fn lu_solve(factors: LU, b: Vector) -> Result(Vector, NlaError) {
  case factors.l.rows == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(factors.l.rows),
        actual: int.to_string(b.size),
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

pub fn forward_substitution(l: Matrix, b: Vector) -> Result(Vector, NlaError) {
  case l.rows == l.cols && l.rows == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(l.rows) <> "x" <> int.to_string(l.cols),
        actual: int.to_string(b.size),
      ))
    True -> forward_loop(l, b, 0, [])
  }
}

pub fn back_substitution(u: Matrix, b: Vector) -> Result(Vector, NlaError) {
  case u.rows == u.cols && u.rows == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(u.rows) <> "x" <> int.to_string(u.cols),
        actual: int.to_string(b.size),
      ))
    True -> back_loop(u, b, u.rows - 1, [])
  }
}

pub fn determinant(a: Matrix) -> Result(Float, NlaError) {
  case lu_factor(a) {
    Error(e) -> Error(e)
    Ok(factors) -> {
      let product =
        list.fold(matrix.indices(factors.u.rows), 1.0, fn(acc, i) {
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
    False -> Error(NotSquare(a.rows, a.cols))
    True -> {
      case lu_factor(a) {
        Error(e) -> Error(e)
        Ok(factors) -> inverse_columns(factors, 0, a.rows, [])
      }
    }
  }
}

fn cholesky_loop(
  a: Matrix,
  l: Matrix,
  i: Int,
  j: Int,
) -> Result(Cholesky, NlaError) {
  case i >= a.rows {
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

fn pivot_row(u: Matrix, k: Int, n: Int) -> #(Int, Float) {
  list.fold(list.drop(matrix.indices(n), up_to: k), #(k, 0.0), fn(best, i) {
    let value = float.absolute_value(matrix.unsafe_get(u, i, k))
    case value >. best.1 {
      True -> #(i, value)
      False -> best
    }
  })
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
    matrix.from_fn(rows: l.rows, cols: l.cols, with: fn(i, j) {
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

fn is_symmetric(a: Matrix, tolerance: Float) -> Bool {
  list.all(matrix.indices(a.rows), satisfying: fn(i) {
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
  case i >= l.rows {
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
              list.drop(matrix.indices(u.rows), up_to: i + 1),
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

fn unsafe_vector_get(vector: Vector, index: Int) -> Float {
  unsafe_at(vector.data, index)
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
