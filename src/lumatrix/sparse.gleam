import gleam/float
import gleam/int
import gleam/list
import gleam/order
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, OutOfBounds,
}
import lumatrix/matrix.{type Matrix}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

/// A sparse matrix entry in zero-based coordinates.
pub type Entry {
  Entry(row: Int, col: Int, value: Float)
}

/// A compressed sparse row matrix.
///
/// Values are stored in canonical CSR form: row-major order, no duplicate
/// coordinates, and no stored zero entries after construction.
pub opaque type SparseMatrix {
  SparseMatrix(
    rows: Int,
    cols: Int,
    row_offsets: List(Int),
    column_indices: List(Int),
    values: List(Float),
  )
}

pub fn from_entries(
  rows rows_count: Int,
  cols cols_count: Int,
  entries entries: List(Entry),
) -> Result(SparseMatrix, NlaError) {
  from_entries_with_tolerance(
    rows: rows_count,
    cols: cols_count,
    entries: entries,
    drop_tolerance: 0.0,
  )
}

pub fn from_entries_with_tolerance(
  rows rows_count: Int,
  cols cols_count: Int,
  entries entries: List(Entry),
  drop_tolerance drop_tolerance: Float,
) -> Result(SparseMatrix, NlaError) {
  case validate_shape(rows_count, cols_count) {
    Error(e) -> Error(e)
    Ok(_) ->
      case validate_drop_tolerance(drop_tolerance) {
        Error(e) -> Error(e)
        Ok(_) ->
          case validate_entries(entries, rows_count, cols_count) {
            Error(e) -> Error(e)
            Ok(_) -> {
              let canonical = canonical_entries(entries, drop_tolerance)
              Ok(SparseMatrix(
                rows: rows_count,
                cols: cols_count,
                row_offsets: build_row_offsets(rows_count, canonical),
                column_indices: list.map(canonical, fn(entry) { entry.col }),
                values: list.map(canonical, fn(entry) { entry.value }),
              ))
            }
          }
      }
  }
}

pub fn from_dense(
  matrix dense: Matrix,
  drop_tolerance drop_tolerance: Float,
) -> Result(SparseMatrix, NlaError) {
  case validate_drop_tolerance(drop_tolerance) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let entries =
        list.flat_map(indices(matrix.rows(dense)), fn(i) {
          list.filter_map(indices(matrix.cols(dense)), fn(j) {
            let value = matrix.unsafe_get(dense, i, j)
            case float.absolute_value(value) >. drop_tolerance {
              True -> Ok(Entry(row: i, col: j, value: value))
              False -> Error(Nil)
            }
          })
        })
      from_entries_with_tolerance(
        rows: matrix.rows(dense),
        cols: matrix.cols(dense),
        entries: entries,
        drop_tolerance: drop_tolerance,
      )
    }
  }
}

pub fn from_rows(
  rows rows_data: List(List(Float)),
  drop_tolerance drop_tolerance: Float,
) -> Result(SparseMatrix, NlaError) {
  case matrix.from_rows(rows_data) {
    Error(e) -> Error(e)
    Ok(dense) -> from_dense(dense, drop_tolerance)
  }
}

pub fn to_dense(sparse: SparseMatrix) -> Matrix {
  let assert Ok(dense) =
    matrix.from_fn(rows: sparse.rows, cols: sparse.cols, with: fn(i, j) {
      unsafe_get(sparse, i, j)
    })
  dense
}

pub fn rows(sparse: SparseMatrix) -> Int {
  sparse.rows
}

pub fn cols(sparse: SparseMatrix) -> Int {
  sparse.cols
}

pub fn is_square(sparse: SparseMatrix) -> Bool {
  sparse.rows == sparse.cols
}

pub fn nnz(sparse: SparseMatrix) -> Int {
  list.length(sparse.values)
}

pub fn row_offsets(sparse: SparseMatrix) -> List(Int) {
  sparse.row_offsets
}

pub fn column_indices(sparse: SparseMatrix) -> List(Int) {
  sparse.column_indices
}

pub fn values(sparse: SparseMatrix) -> List(Float) {
  sparse.values
}

pub fn to_entries(sparse: SparseMatrix) -> List(Entry) {
  list.flat_map(indices(sparse.rows), fn(row) {
    let start = unsafe_int_at(sparse.row_offsets, row)
    let stop = unsafe_int_at(sparse.row_offsets, row + 1)
    list.map(interval(start, stop), fn(position) {
      Entry(
        row: row,
        col: unsafe_int_at(sparse.column_indices, position),
        value: unsafe_float_at(sparse.values, position),
      )
    })
  })
}

pub fn get(
  sparse: SparseMatrix,
  row: Int,
  col: Int,
) -> Result(Float, NlaError) {
  case in_bounds(sparse, row, col) {
    True -> Ok(unsafe_get(sparse, row, col))
    False -> Error(OutOfBounds(row, col))
  }
}

pub fn unsafe_get(sparse: SparseMatrix, row: Int, col: Int) -> Float {
  case in_bounds(sparse, row, col) {
    False -> panic as "sparse.unsafe_get index out of bounds"
    True -> {
      let start = unsafe_int_at(sparse.row_offsets, row)
      let stop = unsafe_int_at(sparse.row_offsets, row + 1)
      row_value(sparse, col, start, stop)
    }
  }
}

pub fn mul_vec(sparse: SparseMatrix, x: Vector) -> Result(Vector, NlaError) {
  let x_size = vector.dimension(x)
  case sparse.cols == x_size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(sparse.cols),
        actual: int.to_string(x_size),
      ))
    True ->
      Ok(
        vector.from_list(
          list.map(indices(sparse.rows), fn(row) { row_dot(sparse, row, x) }),
        ),
      )
  }
}

pub fn transpose_mul_vec(
  sparse: SparseMatrix,
  x: Vector,
) -> Result(Vector, NlaError) {
  case sparse.rows == vector.dimension(x) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(sparse.rows),
        actual: int.to_string(vector.dimension(x)),
      ))
    True -> mul_vec(transpose(sparse), x)
  }
}

pub fn transpose(sparse: SparseMatrix) -> SparseMatrix {
  let transposed_entries =
    list.map(to_entries(sparse), fn(entry) {
      Entry(row: entry.col, col: entry.row, value: entry.value)
    })
  let assert Ok(transposed) =
    from_entries(
      rows: sparse.cols,
      cols: sparse.rows,
      entries: transposed_entries,
    )
  transposed
}

pub fn scale(sparse: SparseMatrix, scalar: Float) -> SparseMatrix {
  let entries =
    list.map(to_entries(sparse), fn(entry) {
      Entry(row: entry.row, col: entry.col, value: entry.value *. scalar)
    })
  let assert Ok(result) =
    from_entries(rows: sparse.rows, cols: sparse.cols, entries: entries)
  result
}

pub fn norm_inf(sparse: SparseMatrix) -> Float {
  list.fold(indices(sparse.rows), 0.0, fn(best, row) {
    float.max(best, row_abs_sum(sparse, row))
  })
}

fn validate_shape(rows_count: Int, cols_count: Int) -> Result(Nil, NlaError) {
  case rows_count > 0 && cols_count > 0 {
    True -> Ok(Nil)
    False -> Error(InvalidInput("sparse matrix dimensions must be positive"))
  }
}

fn validate_drop_tolerance(drop_tolerance: Float) -> Result(Nil, NlaError) {
  case drop_tolerance >=. 0.0 {
    True -> Ok(Nil)
    False -> Error(InvalidInput("drop_tolerance must be non-negative"))
  }
}

fn validate_entries(
  entries: List(Entry),
  rows_count: Int,
  cols_count: Int,
) -> Result(Nil, NlaError) {
  list.try_fold(over: entries, from: Nil, with: fn(_, entry) {
    case
      entry.row >= 0
      && entry.row < rows_count
      && entry.col >= 0
      && entry.col < cols_count
    {
      True -> Ok(Nil)
      False -> Error(OutOfBounds(entry.row, entry.col))
    }
  })
}

fn canonical_entries(
  entries: List(Entry),
  drop_tolerance: Float,
) -> List(Entry) {
  entries
  |> list.sort(by: compare_entries)
  |> combine_sorted_entries(drop_tolerance, [])
}

fn combine_sorted_entries(
  entries: List(Entry),
  drop_tolerance: Float,
  acc: List(Entry),
) -> List(Entry) {
  case entries {
    [] -> list.reverse(acc)
    [first, ..rest] ->
      combine_same_coordinate(
        rest,
        first.row,
        first.col,
        first.value,
        drop_tolerance,
        acc,
      )
  }
}

fn combine_same_coordinate(
  rest: List(Entry),
  row: Int,
  col: Int,
  sum: Float,
  drop_tolerance: Float,
  acc: List(Entry),
) -> List(Entry) {
  case rest {
    [] -> {
      let acc = append_if_stored(acc, row, col, sum, drop_tolerance)
      list.reverse(acc)
    }
    [next, ..tail] ->
      case next.row == row && next.col == col {
        True ->
          combine_same_coordinate(
            tail,
            row,
            col,
            sum +. next.value,
            drop_tolerance,
            acc,
          )
        False -> {
          let acc = append_if_stored(acc, row, col, sum, drop_tolerance)
          combine_sorted_entries(rest, drop_tolerance, acc)
        }
      }
  }
}

fn append_if_stored(
  acc: List(Entry),
  row: Int,
  col: Int,
  value: Float,
  drop_tolerance: Float,
) -> List(Entry) {
  case float.absolute_value(value) >. drop_tolerance {
    True -> [Entry(row: row, col: col, value: value), ..acc]
    False -> acc
  }
}

fn build_row_offsets(rows_count: Int, entries: List(Entry)) -> List(Int) {
  list.map(indices(rows_count + 1), fn(row) {
    list.count(entries, where: fn(entry) { entry.row < row })
  })
}

fn row_value(
  sparse: SparseMatrix,
  col: Int,
  position: Int,
  stop: Int,
) -> Float {
  case position >= stop {
    True -> 0.0
    False -> {
      let current_col = unsafe_int_at(sparse.column_indices, position)
      case current_col == col {
        True -> unsafe_float_at(sparse.values, position)
        False ->
          case current_col > col {
            True -> 0.0
            False -> row_value(sparse, col, position + 1, stop)
          }
      }
    }
  }
}

fn row_dot(sparse: SparseMatrix, row: Int, x: Vector) -> Float {
  let start = unsafe_int_at(sparse.row_offsets, row)
  let stop = unsafe_int_at(sparse.row_offsets, row + 1)
  numerics.compensated_sum_map(interval(start, stop), fn(position) {
    let col = unsafe_int_at(sparse.column_indices, position)
    unsafe_float_at(sparse.values, position) *. unsafe_vector_get(x, col)
  })
}

fn row_abs_sum(sparse: SparseMatrix, row: Int) -> Float {
  let start = unsafe_int_at(sparse.row_offsets, row)
  let stop = unsafe_int_at(sparse.row_offsets, row + 1)
  numerics.compensated_sum_map(interval(start, stop), fn(position) {
    float.absolute_value(unsafe_float_at(sparse.values, position))
  })
}

fn compare_entries(a: Entry, b: Entry) -> order.Order {
  case int.compare(a.row, with: b.row) {
    order.Eq -> int.compare(a.col, with: b.col)
    order.Lt -> order.Lt
    order.Gt -> order.Gt
  }
}

fn in_bounds(sparse: SparseMatrix, row: Int, col: Int) -> Bool {
  row >= 0 && row < sparse.rows && col >= 0 && col < sparse.cols
}

fn unsafe_vector_get(vector: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(vector, index)
  value
}

fn unsafe_int_at(data: List(Int), index: Int) -> Int {
  let #(_, right) = list.split(data, at: index)
  case right {
    [value, ..] -> value
    [] -> panic as "sparse internal int index out of bounds"
  }
}

fn unsafe_float_at(data: List(Float), index: Int) -> Float {
  let #(_, right) = list.split(data, at: index)
  case right {
    [value, ..] -> value
    [] -> panic as "sparse internal float index out of bounds"
  }
}

fn indices(size: Int) -> List(Int) {
  int.range(from: 0, to: size, with: [], run: fn(acc, i) { [i, ..acc] })
  |> list.reverse
}

fn interval(start: Int, stop: Int) -> List(Int) {
  int.range(from: start, to: stop, with: [], run: fn(acc, i) { [i, ..acc] })
  |> list.reverse
}
