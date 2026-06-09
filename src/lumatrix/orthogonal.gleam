import gleam/list
import lumatrix/error.{type NlaError, DimensionMismatch, InvalidInput, ZeroNorm}
import lumatrix/matrix.{type Matrix}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

const zero_tolerance = 1.0e-12

pub type Householder {
  Householder(v: Vector, beta: Float, target_norm: Float)
}

pub type Givens {
  Givens(c: Float, s: Float, r: Float)
}

/// Shape convention used by a QR factorization result.
pub type QRForm {
  /// `q` is m-by-m and `r` is m-by-n.
  FullQR
  /// `q` is m-by-n and `r` is n-by-n.
  ThinQR
}

/// A QR factorization result.
///
/// Householder and Givens QR currently return `FullQR`; Gram-Schmidt routines
/// return `ThinQR`.
pub type QR {
  QR(q: Matrix, r: Matrix, form: QRForm)
}

pub fn householder(vector x: Vector) -> Result(Householder, NlaError) {
  case vector.norm2(x) {
    Error(e) -> Error(e)
    Ok(a) if a <=. 0.0 -> Error(ZeroNorm)
    Ok(a) -> householder_from_values(vector.to_list(x), a)
  }
}

pub fn householder_matrix(
  vector x: Vector,
) -> Result(#(Matrix, Float), NlaError) {
  case householder(x) {
    Error(e) -> Error(e)
    Ok(h) -> {
      let assert Ok(identity) = matrix.identity(vector.dimension(x))
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
  case numerics.hypot(a, b) {
    Error(_) -> Error(InvalidInput("cannot compute Givens radius"))
    Ok(r) if r <=. 0.0 -> Ok(Givens(c: 1.0, s: 0.0, r: 0.0))
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
  case i >= 0 && k >= 0 && i < matrix.rows(a) && k < matrix.rows(a) && i != k {
    False -> Error(InvalidInput("invalid Givens rows"))
    True ->
      matrix.from_fn(
        rows: matrix.rows(a),
        cols: matrix.cols(a),
        with: fn(row, col) {
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
        },
      )
  }
}

pub fn qr_householder(a: Matrix) -> Result(QR, NlaError) {
  case matrix.rows(a) >= matrix.cols(a) {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_householder_loop(0, min_int(matrix.rows(a) - 1, matrix.cols(a)), q0, a)
    }
  }
}

pub fn householder_qr(a: Matrix) -> Result(QR, NlaError) {
  qr_householder(a)
}

pub fn qr_givens(a: Matrix) -> Result(QR, NlaError) {
  case matrix.rows(a) >= matrix.cols(a) {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_givens_column(0, matrix.cols(a), q0, a)
    }
  }
}

pub fn givens_qr(a: Matrix) -> Result(QR, NlaError) {
  qr_givens(a)
}

pub fn qr_classical_gram_schmidt(a: Matrix) -> Result(QR, NlaError) {
  case matrix.rows(a) >= matrix.cols(a) {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> classical_gs_loop(a, 0, [], [])
  }
}

pub fn classical_gram_schmidt_qr(a: Matrix) -> Result(QR, NlaError) {
  qr_classical_gram_schmidt(a)
}

pub fn qr_modified_gram_schmidt(a: Matrix) -> Result(QR, NlaError) {
  case matrix.rows(a) >= matrix.cols(a) {
    False ->
      Error(DimensionMismatch(
        expected: "rows >= columns",
        actual: "rows < columns",
      ))
    True -> modified_gs_loop(a, 0, [], [])
  }
}

pub fn modified_gram_schmidt_qr(a: Matrix) -> Result(QR, NlaError) {
  qr_modified_gram_schmidt(a)
}

fn qr_householder_loop(
  k: Int,
  stop: Int,
  q: Matrix,
  r: Matrix,
) -> Result(QR, NlaError) {
  case k >= stop {
    True -> Ok(QR(q: q, r: r, form: FullQR))
    False -> {
      let x =
        vector.from_list(
          list.map(list.drop(matrix.indices(matrix.rows(r)), up_to: k), fn(i) {
            matrix.unsafe_get(r, i, k)
          }),
        )
      case householder_matrix(x) {
        Error(ZeroNorm) -> qr_householder_loop(k + 1, stop, q, r)
        Error(e) -> Error(e)
        Ok(#(small_h, _)) -> {
          let h = embed_householder(matrix.rows(r), k, small_h)
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
  case k >= matrix.cols(a) {
    True -> build_thin_qr(matrix.rows(a), matrix.cols(a), q_vectors, entries)
    False -> {
      case matrix.column(a, k) {
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
  case k >= matrix.cols(a) {
    True -> build_thin_qr(matrix.rows(a), matrix.cols(a), q_vectors, entries)
    False -> {
      case matrix.column(a, k) {
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
  Ok(QR(q: q, r: r, form: ThinQR))
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
    True -> Ok(QR(q: q, r: r, form: FullQR))
    False -> {
      case qr_givens_row(col, matrix.rows(r) - 1, q, r) {
        Error(e) -> Error(e)
        Ok(qr) -> qr_givens_column(col + 1, stop, qr.q, qr.r)
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
    True -> Ok(QR(q: q, r: r, form: FullQR))
    False -> {
      let a = matrix.unsafe_get(r, col, col)
      let b = matrix.unsafe_get(r, row, col)
      case givens(a, b) {
        Error(e) -> Error(e)
        Ok(rotation) ->
          case apply_givens_left(r, col, row, rotation) {
            Error(e) -> Error(e)
            Ok(next_r) ->
              case givens_matrix(matrix.rows(r), col, row, rotation) {
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
  let #(_, right) = list.split(vectors, at: index)
  case right {
    [value, ..] -> value
    [] -> panic as "orthogonal internal vector index out of bounds"
  }
}

fn unsafe_vector_get(values: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
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

fn householder_from_values(
  values: List(Float),
  target_norm: Float,
) -> Result(Householder, NlaError) {
  case values {
    [] -> Error(ZeroNorm)
    [first, ..tail] -> {
      let v =
        vector.from_list([
          stable_householder_head(first, tail, target_norm),
          ..tail
        ])
      case vector.dot(v, v) {
        Error(e) -> Error(e)
        Ok(denom) if denom <=. 0.0 ->
          case vector.basis(list.length(values), 0) {
            Error(e) -> Error(e)
            Ok(e1) ->
              Ok(Householder(v: e1, beta: 0.0, target_norm: target_norm))
          }
        Ok(denom) ->
          Ok(Householder(v: v, beta: 2.0 /. denom, target_norm: target_norm))
      }
    }
  }
}

fn stable_householder_head(
  first: Float,
  tail: List(Float),
  norm: Float,
) -> Float {
  case first >. 0.0 {
    False -> first -. norm
    True ->
      case numerics.norm2(tail) {
        Error(_) -> first -. norm
        Ok(tail_norm) if tail_norm <=. 0.0 -> 0.0
        Ok(tail_norm) -> 0.0 -. tail_norm *. { tail_norm /. { first +. norm } }
      }
  }
}

fn min_int(a: Int, b: Int) -> Int {
  case a < b {
    True -> a
    False -> b
  }
}
