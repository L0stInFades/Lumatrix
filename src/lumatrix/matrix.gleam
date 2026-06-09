import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, NotSquare, OutOfBounds,
}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

/// A dense row-major matrix.
///
/// Matrix dimensions and data are validated at construction time. Use
/// `from_rows`, `from_columns`, `from_flat`, or `from_fn` to create values.
pub opaque type Matrix {
  Matrix(rows: Int, cols: Int, data: List(Float))
}

pub fn from_flat(
  rows rows_count: Int,
  cols cols_count: Int,
  data data_values: List(Float),
) -> Result(Matrix, NlaError) {
  case rows_count > 0 && cols_count > 0 {
    False -> Error(InvalidInput("matrix dimensions must be positive"))
    True -> {
      let expected = rows_count * cols_count
      let actual = list.length(data_values)
      case actual == expected {
        True ->
          Ok(Matrix(rows: rows_count, cols: cols_count, data: data_values))
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(expected),
            actual: int.to_string(actual),
          ))
      }
    }
  }
}

pub fn from_rows(
  rows rows_data: List(List(Float)),
) -> Result(Matrix, NlaError) {
  case rows_data {
    [] -> Error(InvalidInput("matrix must have at least one row"))
    [first, ..] -> {
      let rows_count = list.length(rows_data)
      let cols_count = list.length(first)
      case cols_count > 0 {
        False -> Error(InvalidInput("matrix must have at least one column"))
        True -> {
          let rectangular =
            list.all(rows_data, satisfying: fn(row) {
              list.length(row) == cols_count
            })
          case rectangular {
            True ->
              from_flat(
                rows: rows_count,
                cols: cols_count,
                data: list.flatten(rows_data),
              )
            False -> Error(InvalidInput("matrix rows must have equal length"))
          }
        }
      }
    }
  }
}

pub fn from_fn(
  rows rows_count: Int,
  cols cols_count: Int,
  with f: fn(Int, Int) -> Float,
) -> Result(Matrix, NlaError) {
  case rows_count > 0 && cols_count > 0 {
    False -> Error(InvalidInput("matrix dimensions must be positive"))
    True ->
      Ok(Matrix(
        rows: rows_count,
        cols: cols_count,
        data: list.flat_map(indices(rows_count), fn(i) {
          list.map(indices(cols_count), fn(j) { f(i, j) })
        }),
      ))
  }
}

pub fn zeros(
  rows rows_count: Int,
  cols cols_count: Int,
) -> Result(Matrix, NlaError) {
  from_fn(rows: rows_count, cols: cols_count, with: fn(_, _) { 0.0 })
}

pub fn identity(size: Int) -> Result(Matrix, NlaError) {
  from_fn(rows: size, cols: size, with: fn(i, j) {
    case i == j {
      True -> 1.0
      False -> 0.0
    }
  })
}

pub fn diagonal(values: List(Float)) -> Result(Matrix, NlaError) {
  let n = list.length(values)
  from_fn(rows: n, cols: n, with: fn(i, j) {
    case i == j {
      True -> unsafe_at(values, i)
      False -> 0.0
    }
  })
}

pub fn rows(matrix: Matrix) -> Int {
  matrix.rows
}

pub fn cols(matrix: Matrix) -> Int {
  matrix.cols
}

pub fn is_square(matrix: Matrix) -> Bool {
  matrix.rows == matrix.cols
}

pub fn get(matrix: Matrix, row: Int, col: Int) -> Result(Float, NlaError) {
  case in_bounds(matrix, row, col) {
    True -> Ok(unsafe_get(matrix, row, col))
    False -> Error(OutOfBounds(row, col))
  }
}

pub fn set(
  matrix: Matrix,
  row: Int,
  col: Int,
  value: Float,
) -> Result(Matrix, NlaError) {
  case in_bounds(matrix, row, col) {
    True ->
      Ok(Matrix(
        rows: matrix.rows,
        cols: matrix.cols,
        data: set_at(matrix.data, flat_index(matrix, row, col), value),
      ))
    False -> Error(OutOfBounds(row, col))
  }
}

pub fn to_rows(matrix: Matrix) -> List(List(Float)) {
  list.map(indices(matrix.rows), fn(i) {
    list.map(indices(matrix.cols), fn(j) { unsafe_get(matrix, i, j) })
  })
}

pub fn from_columns(columns: List(Vector)) -> Result(Matrix, NlaError) {
  case columns {
    [] -> Error(InvalidInput("matrix must have at least one column"))
    [first, ..] -> {
      let rows_count = vector.dimension(first)
      case rows_count > 0 {
        False -> Error(InvalidInput("matrix must have at least one row"))
        True -> {
          let rectangular =
            list.all(columns, satisfying: fn(column) {
              vector.dimension(column) == rows_count
            })
          case rectangular {
            True ->
              from_rows(
                list.map(indices(rows_count), fn(i) {
                  list.map(columns, fn(column) { unsafe_vector_get(column, i) })
                }),
              )
            False ->
              Error(InvalidInput("matrix columns must have equal length"))
          }
        }
      }
    }
  }
}

pub fn to_columns(matrix: Matrix) -> List(Vector) {
  list.map(indices(matrix.cols), fn(j) {
    let assert Ok(column) = column(matrix, j)
    column
  })
}

pub fn row(matrix: Matrix, row_index: Int) -> Result(Vector, NlaError) {
  case row_index >= 0 && row_index < matrix.rows {
    True ->
      Ok(
        vector.from_list(
          list.map(indices(matrix.cols), fn(j) {
            unsafe_get(matrix, row_index, j)
          }),
        ),
      )
    False -> Error(OutOfBounds(row_index, 0))
  }
}

pub fn column(matrix: Matrix, col_index: Int) -> Result(Vector, NlaError) {
  case col_index >= 0 && col_index < matrix.cols {
    True ->
      Ok(
        vector.from_list(
          list.map(indices(matrix.rows), fn(i) {
            unsafe_get(matrix, i, col_index)
          }),
        ),
      )
    False -> Error(OutOfBounds(0, col_index))
  }
}

pub fn col(matrix: Matrix, col_index: Int) -> Result(Vector, NlaError) {
  column(matrix, col_index)
}

pub fn column_matrix(values: Vector) -> Result(Matrix, NlaError) {
  from_columns([values])
}

pub fn row_matrix(values: Vector) -> Result(Matrix, NlaError) {
  from_rows([vector.to_list(values)])
}

pub fn transpose(matrix: Matrix) -> Matrix {
  let assert Ok(result) =
    from_fn(rows: matrix.cols, cols: matrix.rows, with: fn(i, j) {
      unsafe_get(matrix, j, i)
    })
  result
}

pub fn add(a: Matrix, b: Matrix) -> Result(Matrix, NlaError) {
  zip_with(a, b, fn(x, y) { x +. y })
}

pub fn sub(a: Matrix, b: Matrix) -> Result(Matrix, NlaError) {
  zip_with(a, b, fn(x, y) { x -. y })
}

pub fn scale(matrix: Matrix, scalar: Float) -> Matrix {
  Matrix(
    rows: matrix.rows,
    cols: matrix.cols,
    data: list.map(matrix.data, fn(x) { scalar *. x }),
  )
}

/// Multiply a matrix by a coordinate vector, interpreting the vector as the
/// column vector `x` in `A * x`.
pub fn mul_vec(matrix: Matrix, x: Vector) -> Result(Vector, NlaError) {
  let x_size = vector.dimension(x)
  case matrix.cols == x_size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.cols),
        actual: int.to_string(x_size),
      ))
    True ->
      Ok(
        vector.from_list(
          list.map(indices(matrix.rows), fn(i) {
            numerics.compensated_sum_map(indices(matrix.cols), fn(j) {
              unsafe_get(matrix, i, j) *. unsafe_vector_get(x, j)
            })
          }),
        ),
      )
  }
}

/// Multiply the transpose by a coordinate vector, computing `A^T * x` without
/// making callers spell out `matrix.mul_vec(matrix.transpose(a), x)`.
pub fn transpose_mul_vec(
  matrix: Matrix,
  x: Vector,
) -> Result(Vector, NlaError) {
  let x_size = vector.dimension(x)
  case matrix.rows == x_size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows),
        actual: int.to_string(x_size),
      ))
    True ->
      Ok(
        vector.from_list(
          list.map(indices(matrix.cols), fn(j) {
            numerics.compensated_sum_map(indices(matrix.rows), fn(i) {
              unsafe_get(matrix, i, j) *. unsafe_vector_get(x, i)
            })
          }),
        ),
      )
  }
}

pub fn mul(a: Matrix, b: Matrix) -> Result(Matrix, NlaError) {
  case a.cols == b.rows {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.cols),
        actual: int.to_string(b.rows),
      ))
    True ->
      from_fn(rows: a.rows, cols: b.cols, with: fn(i, j) {
        numerics.compensated_sum_map(indices(a.cols), fn(k) {
          unsafe_get(a, i, k) *. unsafe_get(b, k, j)
        })
      })
  }
}

pub fn outer(x: Vector, y: Vector) -> Result(Matrix, NlaError) {
  from_fn(rows: vector.dimension(x), cols: vector.dimension(y), with: fn(i, j) {
    unsafe_vector_get(x, i) *. unsafe_vector_get(y, j)
  })
}

pub fn trace(matrix: Matrix) -> Result(Float, NlaError) {
  case is_square(matrix) {
    True ->
      Ok(
        numerics.compensated_sum_map(indices(matrix.rows), fn(i) {
          unsafe_get(matrix, i, i)
        }),
      )
    False -> Error(NotSquare(matrix.rows, matrix.cols))
  }
}

pub fn norm_inf(matrix: Matrix) -> Float {
  list.fold(indices(matrix.rows), 0.0, fn(best, i) {
    let row_sum =
      numerics.compensated_sum_map(indices(matrix.cols), fn(j) {
        float.absolute_value(unsafe_get(matrix, i, j))
      })
    float.max(best, row_sum)
  })
}

pub fn frobenius_norm(matrix: Matrix) -> Result(Float, NlaError) {
  case numerics.norm2(matrix.data) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(InvalidInput("cannot take square root of norm"))
  }
}

pub fn swap_rows(matrix: Matrix, a: Int, b: Int) -> Result(Matrix, NlaError) {
  case a >= 0 && a < matrix.rows && b >= 0 && b < matrix.rows {
    True ->
      from_fn(rows: matrix.rows, cols: matrix.cols, with: fn(i, j) {
        case i == a {
          True -> unsafe_get(matrix, b, j)
          False ->
            case i == b {
              True -> unsafe_get(matrix, a, j)
              False -> unsafe_get(matrix, i, j)
            }
        }
      })
    False -> Error(OutOfBounds(a, b))
  }
}

pub fn set_row(
  matrix: Matrix,
  row_index: Int,
  values: Vector,
) -> Result(Matrix, NlaError) {
  let values_size = vector.dimension(values)
  case row_index < 0 || row_index >= matrix.rows {
    True -> Error(OutOfBounds(row_index, 0))
    False ->
      case values_size == matrix.cols {
        True ->
          from_fn(rows: matrix.rows, cols: matrix.cols, with: fn(i, j) {
            case i == row_index {
              True -> unsafe_vector_get(values, j)
              False -> unsafe_get(matrix, i, j)
            }
          })
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(matrix.cols),
            actual: int.to_string(values_size),
          ))
      }
  }
}

pub fn set_column(
  matrix: Matrix,
  col_index: Int,
  values: Vector,
) -> Result(Matrix, NlaError) {
  let values_size = vector.dimension(values)
  case col_index < 0 || col_index >= matrix.cols {
    True -> Error(OutOfBounds(0, col_index))
    False ->
      case values_size == matrix.rows {
        True ->
          from_fn(rows: matrix.rows, cols: matrix.cols, with: fn(i, j) {
            case j == col_index {
              True -> unsafe_vector_get(values, i)
              False -> unsafe_get(matrix, i, j)
            }
          })
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(matrix.rows),
            actual: int.to_string(values_size),
          ))
      }
  }
}

pub fn zip_with(
  a: Matrix,
  b: Matrix,
  f: fn(Float, Float) -> Float,
) -> Result(Matrix, NlaError) {
  case a.rows == b.rows && a.cols == b.cols {
    True ->
      Ok(Matrix(
        rows: a.rows,
        cols: a.cols,
        data: list.map(list.zip(a.data, with: b.data), fn(pair) {
          let #(x, y) = pair
          f(x, y)
        }),
      ))
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.rows) <> "x" <> int.to_string(a.cols),
        actual: int.to_string(b.rows) <> "x" <> int.to_string(b.cols),
      ))
  }
}

pub fn approx_equal(a: Matrix, b: Matrix, tolerance: Float) -> Bool {
  case sub(a, b) {
    Ok(delta) -> norm_inf(delta) <=. tolerance
    Error(_) -> False
  }
}

/// Get an entry without returning a `Result`.
///
/// Prefer `get` for user-provided indices. This function panics if the index is
/// outside the matrix bounds.
pub fn unsafe_get(matrix: Matrix, row: Int, col: Int) -> Float {
  case in_bounds(matrix, row, col) {
    True -> unsafe_at(matrix.data, flat_index(matrix, row, col))
    False -> panic as "matrix.unsafe_get index out of bounds"
  }
}

pub fn indices(size: Int) -> List(Int) {
  int.range(from: 0, to: size, with: [], run: fn(acc, i) { [i, ..acc] })
  |> list.reverse
}

fn in_bounds(matrix: Matrix, row: Int, col: Int) -> Bool {
  row >= 0 && row < matrix.rows && col >= 0 && col < matrix.cols
}

fn flat_index(matrix: Matrix, row: Int, col: Int) -> Int {
  row * matrix.cols + col
}

fn unsafe_vector_get(vector: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(vector, index)
  value
}

fn unsafe_at(data: List(Float), index: Int) -> Float {
  let #(_, right) = list.split(data, at: index)
  case right {
    [value, ..] -> value
    [] -> panic as "matrix internal index out of bounds"
  }
}

fn set_at(data: List(Float), index: Int, value: Float) -> List(Float) {
  case data {
    [] -> []
    [first, ..rest] -> {
      case index == 0 {
        True -> [value, ..rest]
        False -> [first, ..set_at(rest, index - 1, value)]
      }
    }
  }
}
