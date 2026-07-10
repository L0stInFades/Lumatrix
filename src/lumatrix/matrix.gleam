import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, DimensionMismatch, InvalidInput,
  NonFiniteInput, NotSquare, OutOfBounds,
}
import lumatrix/internal/storage.{type Storage}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

/// A dense row-major matrix.
///
/// Matrix dimensions and data are validated at construction time. Use
/// `from_rows`, `from_columns`, `from_flat`, or `from_fn` to create values.
pub opaque type Matrix {
  Matrix(rows: Int, cols: Int, data: Storage)
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
          case list.all(data_values, satisfying: numerics.is_finite) {
            True ->
              Ok(Matrix(
                rows: rows_count,
                cols: cols_count,
                data: storage.from_list(data_values),
              ))
            False -> Error(NonFiniteInput("matrix data"))
          }
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
      from_flat(
        rows: rows_count,
        cols: cols_count,
        data: list.flat_map(indices(rows_count), fn(i) {
          list.map(indices(cols_count), fn(j) { f(i, j) })
        }),
      )
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
  let diagonal_values = storage.from_list(values)
  from_fn(rows: n, cols: n, with: fn(i, j) {
    case i == j {
      True -> storage.unsafe_get(diagonal_values, i)
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

pub fn is_finite(matrix: Matrix) -> Bool {
  list.all(storage.to_list(matrix.data), satisfying: numerics.is_finite)
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
  case numerics.is_finite(value) {
    False -> Error(NonFiniteInput("matrix entry"))
    True ->
      case in_bounds(matrix, row, col) {
        True ->
          Ok(Matrix(
            rows: matrix.rows,
            cols: matrix.cols,
            data: storage.set(matrix.data, flat_index(matrix, row, col), value),
          ))
        False -> Error(OutOfBounds(row, col))
      }
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
  checked_zip_with(a, b, "matrix addition", numerics.checked_add)
}

pub fn sub(a: Matrix, b: Matrix) -> Result(Matrix, NlaError) {
  checked_zip_with(a, b, "matrix subtraction", numerics.checked_subtract)
}

pub fn scale(matrix: Matrix, scalar: Float) -> Matrix {
  Matrix(
    rows: matrix.rows,
    cols: matrix.cols,
    data: storage.map(matrix.data, with: fn(x) { scalar *. x }),
  )
}

pub fn checked_scale(
  matrix: Matrix,
  scalar: Float,
) -> Result(Matrix, NlaError) {
  case numerics.is_finite(scalar) {
    False -> Error(NonFiniteInput("matrix scale"))
    True ->
      checked_map(matrix, "matrix scaling", fn(value) {
        numerics.checked_multiply(value, scalar)
      })
  }
}

pub fn divide(matrix: Matrix, scalar: Float) -> Result(Matrix, NlaError) {
  case numerics.is_finite(scalar) {
    False -> Error(NonFiniteInput("matrix divisor"))
    True if scalar == 0.0 ->
      Error(InvalidInput("matrix divisor must be non-zero"))
    True ->
      checked_map(matrix, "matrix division", fn(value) {
        numerics.checked_divide(value, scalar)
      })
  }
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
      case is_finite(matrix) && vector.is_finite(x) {
        False -> Error(NonFiniteInput("matrix-vector product operands"))
        True ->
          case
            list.try_map(indices(matrix.rows), fn(i) {
              checked_dot_at(
                matrix.cols,
                fn(j) { unsafe_get(matrix, i, j) },
                fn(j) { unsafe_vector_get(x, j) },
                "matrix-vector product",
              )
            })
          {
            Ok(values) -> Ok(vector.from_list(values))
            Error(e) -> Error(e)
          }
      }
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
      case is_finite(matrix) && vector.is_finite(x) {
        False ->
          Error(NonFiniteInput("transpose matrix-vector product operands"))
        True ->
          case
            list.try_map(indices(matrix.cols), fn(j) {
              checked_dot_at(
                matrix.rows,
                fn(i) { unsafe_get(matrix, i, j) },
                fn(i) { unsafe_vector_get(x, i) },
                "transpose matrix-vector product",
              )
            })
          {
            Ok(values) -> Ok(vector.from_list(values))
            Error(e) -> Error(e)
          }
      }
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
      case is_finite(a) && is_finite(b) {
        False -> Error(NonFiniteInput("matrix product operands"))
        True ->
          case
            list.try_map(indices(a.rows), fn(i) {
              list.try_map(indices(b.cols), fn(j) {
                checked_dot_at(
                  a.cols,
                  fn(k) { unsafe_get(a, i, k) },
                  fn(k) { unsafe_get(b, k, j) },
                  "matrix product",
                )
              })
            })
          {
            Ok(rows) -> from_rows(rows)
            Error(e) -> Error(e)
          }
      }
  }
}

pub fn outer(x: Vector, y: Vector) -> Result(Matrix, NlaError) {
  case vector.is_finite(x) && vector.is_finite(y) {
    False -> Error(NonFiniteInput("outer product operands"))
    True ->
      case
        list.try_map(indices(vector.dimension(x)), fn(i) {
          list.try_map(indices(vector.dimension(y)), fn(j) {
            case
              numerics.checked_multiply(
                unsafe_vector_get(x, i),
                unsafe_vector_get(y, j),
              )
            {
              Ok(value) -> Ok(value)
              Error(_) -> Error(ArithmeticOverflow("outer product"))
            }
          })
        })
      {
        Ok(rows) -> from_rows(rows)
        Error(e) -> Error(e)
      }
  }
}

pub fn trace(matrix: Matrix) -> Result(Float, NlaError) {
  case is_square(matrix) {
    True ->
      case is_finite(matrix) {
        False -> Error(NonFiniteInput("matrix trace operand"))
        True ->
          case
            numerics.checked_sum(
              list.map(indices(matrix.rows), fn(i) { unsafe_get(matrix, i, i) }),
            )
          {
            Ok(value) -> Ok(value)
            Error(_) -> Error(ArithmeticOverflow("matrix trace"))
          }
      }
    False -> Error(NotSquare(matrix.rows, matrix.cols))
  }
}

pub fn norm_inf(matrix: Matrix) -> Float {
  list.fold(indices(matrix.rows), 0.0, fn(best, i) {
    let row_sum =
      numerics.saturating_nonnegative_sum_map(indices(matrix.cols), fn(j) {
        float.absolute_value(unsafe_get(matrix, i, j))
      })
    float.max(best, row_sum)
  })
}

pub fn frobenius_norm(matrix: Matrix) -> Result(Float, NlaError) {
  case is_finite(matrix) {
    False -> Error(NonFiniteInput("matrix norm operand"))
    True ->
      case numerics.norm2(storage.to_list(matrix.data)) {
        Ok(value) -> Ok(value)
        Error(_) -> Error(ArithmeticOverflow("matrix norm"))
      }
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
        data: storage.from_list(
          list.map(
            list.zip(storage.to_list(a.data), with: storage.to_list(b.data)),
            fn(pair) {
              let #(x, y) = pair
              f(x, y)
            },
          ),
        ),
      ))
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.rows) <> "x" <> int.to_string(a.cols),
        actual: int.to_string(b.rows) <> "x" <> int.to_string(b.cols),
      ))
  }
}

pub fn approx_equal(a: Matrix, b: Matrix, tolerance: Float) -> Bool {
  case a.rows == b.rows && a.cols == b.cols {
    False -> False
    True ->
      list.all(
        list.zip(storage.to_list(a.data), with: storage.to_list(b.data)),
        satisfying: fn(pair) {
          numerics.absolute_close(pair.0, pair.1, tolerance)
        },
      )
  }
}

/// Get an entry without returning a `Result`.
///
/// Prefer `get` for user-provided indices. This function panics if the index is
/// outside the matrix bounds.
pub fn unsafe_get(matrix: Matrix, row: Int, col: Int) -> Float {
  case in_bounds(matrix, row, col) {
    True -> storage.unsafe_get(matrix.data, flat_index(matrix, row, col))
    False -> panic as "matrix.unsafe_get index out of bounds"
  }
}

pub fn indices(size: Int) -> List(Int) {
  int.range(from: 0, to: size, with: [], run: fn(acc, i) { [i, ..acc] })
  |> list.reverse
}

fn checked_map(
  matrix: Matrix,
  operation: String,
  f: fn(Float) -> Result(Float, Nil),
) -> Result(Matrix, NlaError) {
  case is_finite(matrix) {
    False -> Error(NonFiniteInput(operation <> " operand"))
    True ->
      case list.try_map(storage.to_list(matrix.data), f) {
        Ok(values) ->
          Ok(Matrix(
            rows: matrix.rows,
            cols: matrix.cols,
            data: storage.from_list(values),
          ))
        Error(_) -> Error(ArithmeticOverflow(operation))
      }
  }
}

fn checked_zip_with(
  a: Matrix,
  b: Matrix,
  operation: String,
  f: fn(Float, Float) -> Result(Float, Nil),
) -> Result(Matrix, NlaError) {
  case a.rows == b.rows && a.cols == b.cols {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.rows) <> "x" <> int.to_string(a.cols),
        actual: int.to_string(b.rows) <> "x" <> int.to_string(b.cols),
      ))
    True ->
      case is_finite(a) && is_finite(b) {
        False -> Error(NonFiniteInput(operation <> " operands"))
        True ->
          case
            list.try_map(
              list.zip(storage.to_list(a.data), with: storage.to_list(b.data)),
              fn(pair) { f(pair.0, pair.1) },
            )
          {
            Ok(values) ->
              Ok(Matrix(
                rows: a.rows,
                cols: a.cols,
                data: storage.from_list(values),
              ))
            Error(_) -> Error(ArithmeticOverflow(operation))
          }
      }
  }
}

fn checked_dot_at(
  count: Int,
  left: fn(Int) -> Float,
  right: fn(Int) -> Float,
  operation: String,
) -> Result(Float, NlaError) {
  case
    numerics.checked_dot_pairs(
      list.map(indices(count), fn(index) { #(left(index), right(index)) }),
    )
  {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
  }
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
