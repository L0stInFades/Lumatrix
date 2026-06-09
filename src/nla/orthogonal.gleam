import gleam/float
import gleam/list
import nla/error.{type NlaError, DimensionMismatch, InvalidInput, ZeroNorm}
import nla/matrix.{type Matrix}
import nla/vector.{type Vector}

const zero_tolerance = 1.0e-12

pub type Householder {
  Householder(v: Vector, beta: Float, target_norm: Float)
}

pub type Givens {
  Givens(c: Float, s: Float, r: Float)
}

pub type QR {
  QR(q: Matrix, r: Matrix)
}

pub fn householder(vector x: Vector) -> Result(Householder, NlaError) {
  case vector.norm2(x) {
    Error(e) -> Error(e)
    Ok(a) if a <=. zero_tolerance -> Error(ZeroNorm)
    Ok(a) -> {
      case vector.basis(x.size, 0) {
        Error(e) -> Error(e)
        Ok(e1) -> {
          let target = vector.scale(e1, a)
          case vector.sub(x, target) {
            Error(e) -> Error(e)
            Ok(v) -> {
              case vector.dot(v, v) {
                Error(e) -> Error(e)
                Ok(denom) if denom <=. zero_tolerance ->
                  Ok(Householder(v: e1, beta: 0.0, target_norm: a))
                Ok(denom) ->
                  Ok(Householder(v: v, beta: 2.0 /. denom, target_norm: a))
              }
            }
          }
        }
      }
    }
  }
}

pub fn householder_matrix(
  vector x: Vector,
) -> Result(#(Matrix, Float), NlaError) {
  case householder(x) {
    Error(e) -> Error(e)
    Ok(h) -> {
      let assert Ok(identity) = matrix.identity(x.size)
      case matrix.outer(h.v, h.v) {
        Error(e) -> Error(e)
        Ok(vvt) ->
          case matrix.sub(identity, matrix.scale(vvt, h.beta)) {
            Ok(h_matrix) -> Ok(#(h_matrix, h.target_norm))
            Error(e) -> Error(e)
          }
      }
    }
  }
}

pub fn apply_householder(
  h: Householder,
  x: Vector,
) -> Result(Vector, NlaError) {
  case vector.dot(h.v, x) {
    Error(e) -> Error(e)
    Ok(projection) -> vector.axpy(0.0 -. h.beta *. projection, h.v, x)
  }
}

pub fn givens(a: Float, b: Float) -> Result(Givens, NlaError) {
  case float.square_root(a *. a +. b *. b) {
    Error(_) -> Error(InvalidInput("cannot compute Givens radius"))
    Ok(r) if r <=. zero_tolerance -> Ok(Givens(c: 1.0, s: 0.0, r: 0.0))
    Ok(r) -> Ok(Givens(c: a /. r, s: b /. r, r: r))
  }
}

pub fn givens_matrix(
  size: Int,
  i: Int,
  k: Int,
  rotation: Givens,
) -> Result(Matrix, NlaError) {
  case size > 0 && i >= 0 && k >= 0 && i < size && k < size && i != k {
    False -> Error(InvalidInput("invalid Givens plane"))
    True ->
      matrix.from_fn(rows: size, cols: size, with: fn(row, col) {
        case row == col {
          True ->
            case row == i || row == k {
              True -> rotation.c
              False -> 1.0
            }
          False ->
            case row == i && col == k {
              True -> rotation.s
              False ->
                case row == k && col == i {
                  True -> 0.0 -. rotation.s
                  False -> 0.0
                }
            }
        }
      })
  }
}

pub fn apply_givens_left(
  a: Matrix,
  i: Int,
  k: Int,
  rotation: Givens,
) -> Result(Matrix, NlaError) {
  case i >= 0 && k >= 0 && i < a.rows && k < a.rows && i != k {
    False -> Error(InvalidInput("invalid Givens rows"))
    True ->
      matrix.from_fn(rows: a.rows, cols: a.cols, with: fn(row, col) {
        case row == i {
          True ->
            rotation.c
            *. matrix.unsafe_get(a, i, col)
            +. rotation.s
            *. matrix.unsafe_get(a, k, col)
          False ->
            case row == k {
              True ->
                { 0.0 -. rotation.s }
                *. matrix.unsafe_get(a, i, col)
                +. rotation.c
                *. matrix.unsafe_get(a, k, col)
              False -> matrix.unsafe_get(a, row, col)
            }
        }
      })
  }
}

pub fn qr_householder(a: Matrix) -> Result(QR, NlaError) {
  case a.rows >= a.cols {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> {
      let assert Ok(q0) = matrix.identity(a.rows)
      qr_householder_loop(0, min_int(a.rows - 1, a.cols), q0, a)
    }
  }
}

pub fn qr_givens(a: Matrix) -> Result(QR, NlaError) {
  case a.rows >= a.cols {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> {
      let assert Ok(q0) = matrix.identity(a.rows)
      qr_givens_column(0, a.cols, q0, a)
    }
  }
}

pub fn qr_classical_gram_schmidt(a: Matrix) -> Result(QR, NlaError) {
  case a.rows >= a.cols {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> classical_gs_loop(a, 0, [], [])
  }
}

pub fn qr_modified_gram_schmidt(a: Matrix) -> Result(QR, NlaError) {
  case a.rows >= a.cols {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> modified_gs_loop(a, 0, [], [])
  }
}

fn qr_householder_loop(
  k: Int,
  stop: Int,
  q: Matrix,
  r: Matrix,
) -> Result(QR, NlaError) {
  case k >= stop {
    True -> Ok(QR(q: q, r: r))
    False -> {
      let x =
        vector.from_list(
          list.map(list.drop(matrix.indices(r.rows), up_to: k), fn(i) {
            matrix.unsafe_get(r, i, k)
          }),
        )
      case householder_matrix(x) {
        Error(ZeroNorm) -> qr_householder_loop(k + 1, stop, q, r)
        Error(e) -> Error(e)
        Ok(#(small_h, _)) -> {
          let h = embed_householder(r.rows, k, small_h)
          case matrix.mul(h, r) {
            Error(e) -> Error(e)
            Ok(next_r) ->
              case matrix.mul(q, h) {
                Error(e) -> Error(e)
                Ok(next_q) -> qr_householder_loop(k + 1, stop, next_q, next_r)
              }
          }
        }
      }
    }
  }
}

fn classical_gs_loop(
  a: Matrix,
  k: Int,
  q_vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(QR, NlaError) {
  case k >= a.cols {
    True -> build_thin_qr(a.rows, a.cols, q_vectors, entries)
    False -> {
      case matrix.col(a, k) {
        Error(e) -> Error(e)
        Ok(original) ->
          case
            classical_orthogonalize(
              q_vectors,
              k,
              0,
              original,
              original,
              entries,
            )
          {
            Error(e) -> Error(e)
            Ok(#(v, entries)) ->
              case finish_gs_column(v, k, q_vectors, entries) {
                Error(e) -> Error(e)
                Ok(#(next_q_vectors, next_entries)) ->
                  classical_gs_loop(a, k + 1, next_q_vectors, next_entries)
              }
          }
      }
    }
  }
}

fn modified_gs_loop(
  a: Matrix,
  k: Int,
  q_vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(QR, NlaError) {
  case k >= a.cols {
    True -> build_thin_qr(a.rows, a.cols, q_vectors, entries)
    False -> {
      case matrix.col(a, k) {
        Error(e) -> Error(e)
        Ok(v) ->
          case modified_orthogonalize(q_vectors, k, 0, v, entries) {
            Error(e) -> Error(e)
            Ok(#(v, entries)) ->
              case finish_gs_column(v, k, q_vectors, entries) {
                Error(e) -> Error(e)
                Ok(#(next_q_vectors, next_entries)) ->
                  modified_gs_loop(a, k + 1, next_q_vectors, next_entries)
              }
          }
      }
    }
  }
}

fn classical_orthogonalize(
  q_vectors: List(Vector),
  k: Int,
  j: Int,
  original: Vector,
  v: Vector,
  entries: List(#(Int, Int, Float)),
) -> Result(#(Vector, List(#(Int, Int, Float))), NlaError) {
  case j >= k {
    True -> Ok(#(v, entries))
    False -> {
      let qj = unsafe_vector_at(q_vectors, j)
      case vector.dot(qj, original) {
        Error(e) -> Error(e)
        Ok(rjk) ->
          case vector.axpy(0.0 -. rjk, qj, v) {
            Error(e) -> Error(e)
            Ok(next_v) ->
              classical_orthogonalize(q_vectors, k, j + 1, original, next_v, [
                #(j, k, rjk),
                ..entries
              ])
          }
      }
    }
  }
}

fn modified_orthogonalize(
  q_vectors: List(Vector),
  k: Int,
  j: Int,
  v: Vector,
  entries: List(#(Int, Int, Float)),
) -> Result(#(Vector, List(#(Int, Int, Float))), NlaError) {
  case j >= k {
    True -> Ok(#(v, entries))
    False -> {
      let qj = unsafe_vector_at(q_vectors, j)
      case vector.dot(qj, v) {
        Error(e) -> Error(e)
        Ok(rjk) ->
          case vector.axpy(0.0 -. rjk, qj, v) {
            Error(e) -> Error(e)
            Ok(next_v) ->
              modified_orthogonalize(q_vectors, k, j + 1, next_v, [
                #(j, k, rjk),
                ..entries
              ])
          }
      }
    }
  }
}

fn finish_gs_column(
  v: Vector,
  k: Int,
  q_vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(#(List(Vector), List(#(Int, Int, Float))), NlaError) {
  case vector.norm2(v) {
    Error(e) -> Error(e)
    Ok(rkk) if rkk <=. zero_tolerance -> Error(ZeroNorm)
    Ok(rkk) -> {
      let qk = vector.scale(v, 1.0 /. rkk)
      Ok(#(list.append(q_vectors, [qk]), [#(k, k, rkk), ..entries]))
    }
  }
}

fn build_thin_qr(
  rows: Int,
  cols: Int,
  q_vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(QR, NlaError) {
  let assert Ok(q) =
    matrix.from_fn(rows: rows, cols: cols, with: fn(i, j) {
      unsafe_vector_get(unsafe_vector_at(q_vectors, j), i)
    })
  let assert Ok(r) =
    matrix.from_fn(rows: cols, cols: cols, with: fn(i, j) {
      entry_value(entries, i, j)
    })
  Ok(QR(q: q, r: r))
}

fn entry_value(entries: List(#(Int, Int, Float)), row: Int, col: Int) -> Float {
  case entries {
    [] -> 0.0
    [entry, ..rest] -> {
      case entry.0 == row && entry.1 == col {
        True -> entry.2
        False -> entry_value(rest, row, col)
      }
    }
  }
}

fn qr_givens_column(
  col: Int,
  stop: Int,
  q: Matrix,
  r: Matrix,
) -> Result(QR, NlaError) {
  case col >= stop {
    True -> Ok(QR(q: q, r: r))
    False -> {
      case qr_givens_row(col, r.rows - 1, q, r) {
        Error(e) -> Error(e)
        Ok(QR(q: next_q, r: next_r)) ->
          qr_givens_column(col + 1, stop, next_q, next_r)
      }
    }
  }
}

fn qr_givens_row(
  col: Int,
  row: Int,
  q: Matrix,
  r: Matrix,
) -> Result(QR, NlaError) {
  case row <= col {
    True -> Ok(QR(q: q, r: r))
    False -> {
      let a = matrix.unsafe_get(r, col, col)
      let b = matrix.unsafe_get(r, row, col)
      case givens(a, b) {
        Error(e) -> Error(e)
        Ok(rotation) ->
          case apply_givens_left(r, col, row, rotation) {
            Error(e) -> Error(e)
            Ok(next_r) ->
              case givens_matrix(r.rows, col, row, rotation) {
                Error(e) -> Error(e)
                Ok(g) ->
                  case matrix.mul(q, matrix.transpose(g)) {
                    Error(e) -> Error(e)
                    Ok(next_q) -> qr_givens_row(col, row - 1, next_q, next_r)
                  }
              }
          }
      }
    }
  }
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

fn unsafe_vector_get(vector: Vector, index: Int) -> Float {
  let #(left, right) = list.split(vector.data, at: index)
  case right {
    [value, ..] -> value
    [] -> {
      let _ = left
      0.0
    }
  }
}

fn embed_householder(size: Int, offset: Int, small: Matrix) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: size, cols: size, with: fn(i, j) {
      case i < offset || j < offset {
        True ->
          case i == j {
            True -> 1.0
            False -> 0.0
          }
        False -> matrix.unsafe_get(small, i - offset, j - offset)
      }
    })
  result
}

fn min_int(a: Int, b: Int) -> Int {
  case a < b {
    True -> a
    False -> b
  }
}
