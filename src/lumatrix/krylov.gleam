import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{type NlaError, DimensionMismatch, InvalidInput, NotSquare}
import lumatrix/error_analysis
import lumatrix/least_squares
import lumatrix/matrix.{type Matrix}
import lumatrix/vector.{type Vector}

pub type ArnoldiResult {
  ArnoldiResult(q: Matrix, h: Matrix, steps: Int, happy_breakdown: Bool)
}

pub type LanczosResult {
  LanczosResult(q: Matrix, t: Matrix, steps: Int, happy_breakdown: Bool)
}

pub type GmresResult {
  GmresResult(
    solution: Vector,
    iterations: Int,
    residual_norm: Float,
    converged: Bool,
    happy_breakdown: Bool,
  )
}

pub fn arnoldi(
  a: Matrix,
  initial: Vector,
  steps: Int,
  tolerance: Float,
) -> Result(ArnoldiResult, NlaError) {
  case validate(a, initial, steps) {
    Error(e) -> Error(e)
    Ok(_) ->
      case vector.normalize(initial) {
        Error(e) -> Error(e)
        Ok(q0) -> arnoldi_loop(a, steps, tolerance, 0, [q0], [])
      }
  }
}

pub fn lanczos(
  a: Matrix,
  initial: Vector,
  steps: Int,
  tolerance: Float,
) -> Result(LanczosResult, NlaError) {
  case validate_symmetric(a, initial, steps, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) ->
      case vector.normalize(initial) {
        Error(e) -> Error(e)
        Ok(q0) -> {
          let assert Ok(q_prev) = vector.zeros(a.rows)
          lanczos_loop(a, steps, tolerance, 0, q_prev, 0.0, [q0], [])
        }
      }
  }
}

pub fn gmres(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case validate_system(a, b, initial, max_iterations) {
    Error(e) -> Error(e)
    Ok(_) -> gmres_cycle(a, b, initial, max_iterations, tolerance)
  }
}

pub fn restarted_gmres(
  a: Matrix,
  b: Vector,
  initial: Vector,
  restart: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case restart <= 0 {
    True -> Error(InvalidInput("GMRES restart must be positive"))
    False ->
      case validate_system(a, b, initial, max_iterations) {
        Error(e) -> Error(e)
        Ok(_) ->
          restarted_gmres_loop(
            a,
            b,
            initial,
            0,
            restart,
            max_iterations,
            tolerance,
            False,
          )
      }
  }
}

fn lanczos_loop(
  a: Matrix,
  requested_steps: Int,
  tolerance: Float,
  k: Int,
  q_prev: Vector,
  beta_prev: Float,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(LanczosResult, NlaError) {
  let qk = unsafe_vector_at(vectors, k)
  case matrix.mul_vec(a, qk) {
    Error(e) -> Error(e)
    Ok(aq) ->
      case vector.axpy(0.0 -. beta_prev, q_prev, aq) {
        Error(e) -> Error(e)
        Ok(w0) ->
          case vector.dot(qk, w0) {
            Error(e) -> Error(e)
            Ok(alpha) -> {
              let entries = [#(k, k, alpha), ..entries]
              case vector.axpy(0.0 -. alpha, qk, w0) {
                Error(e) -> Error(e)
                Ok(w) ->
                  case vector.norm2(w) {
                    Error(e) -> Error(e)
                    Ok(beta) -> {
                      let completed_steps = k + 1
                      case
                        beta <=. tolerance
                        || completed_steps >= requested_steps
                        || completed_steps >= a.rows
                      {
                        True ->
                          build_lanczos_result(
                            a.rows,
                            vectors,
                            entries,
                            completed_steps,
                            beta <=. tolerance,
                          )
                        False -> {
                          let next_q = vector.scale(w, 1.0 /. beta)
                          lanczos_loop(
                            a,
                            requested_steps,
                            tolerance,
                            k + 1,
                            qk,
                            beta,
                            list.append(vectors, [next_q]),
                            [#(k, k + 1, beta), #(k + 1, k, beta), ..entries],
                          )
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

fn arnoldi_loop(
  a: Matrix,
  requested_steps: Int,
  tolerance: Float,
  k: Int,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(ArnoldiResult, NlaError) {
  case k >= requested_steps || k >= a.rows {
    True -> build_result(a.rows, vectors, entries, k, False)
    False -> {
      let qk = unsafe_vector_at(vectors, k)
      case matrix.mul_vec(a, qk) {
        Error(e) -> Error(e)
        Ok(w0) -> {
          case orthogonalize(vectors, k, 0, w0, entries) {
            Error(e) -> Error(e)
            Ok(#(w, entries)) ->
              case vector.norm2(w) {
                Error(e) -> Error(e)
                Ok(h_next) -> {
                  let entries = [#(k + 1, k, h_next), ..entries]
                  case h_next <=. tolerance {
                    True -> build_result(a.rows, vectors, entries, k + 1, True)
                    False -> {
                      let next_q = vector.scale(w, 1.0 /. h_next)
                      arnoldi_loop(
                        a,
                        requested_steps,
                        tolerance,
                        k + 1,
                        list.append(vectors, [next_q]),
                        entries,
                      )
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

fn restarted_gmres_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  iterations: Int,
  restart: Int,
  max_iterations: Int,
  tolerance: Float,
  happy_breakdown: Bool,
) -> Result(GmresResult, NlaError) {
  case error_analysis.residual_norm2(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(GmresResult(
        solution: x,
        iterations: iterations,
        residual_norm: r_norm,
        converged: True,
        happy_breakdown: happy_breakdown,
      ))
    Ok(r_norm) ->
      case iterations >= max_iterations || happy_breakdown {
        True ->
          Ok(GmresResult(
            solution: x,
            iterations: iterations,
            residual_norm: r_norm,
            converged: False,
            happy_breakdown: happy_breakdown,
          ))
        False -> {
          let remaining = max_iterations - iterations
          let cycle_steps = min_int(restart, remaining)
          case gmres_cycle(a, b, x, cycle_steps, tolerance) {
            Error(e) -> Error(e)
            Ok(cycle) ->
              restarted_gmres_loop(
                a,
                b,
                cycle.solution,
                iterations + cycle.iterations,
                restart,
                max_iterations,
                tolerance,
                cycle.happy_breakdown,
              )
          }
        }
      }
  }
}

fn gmres_cycle(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case error_analysis.residual(a, initial, b) {
    Error(e) -> Error(e)
    Ok(r0) ->
      case vector.norm2(r0) {
        Error(e) -> Error(e)
        Ok(beta) if beta <=. tolerance ->
          Ok(GmresResult(
            solution: initial,
            iterations: 0,
            residual_norm: beta,
            converged: True,
            happy_breakdown: False,
          ))
        Ok(beta) ->
          case arnoldi(a, r0, max_iterations, tolerance) {
            Error(e) -> Error(e)
            Ok(arnoldi_result) ->
              case solve_gmres_subproblem(arnoldi_result, beta) {
                Error(e) -> Error(e)
                Ok(correction) ->
                  case vector.add(initial, correction) {
                    Error(e) -> Error(e)
                    Ok(next_x) ->
                      case error_analysis.residual_norm2(a, next_x, b) {
                        Error(e) -> Error(e)
                        Ok(r_norm) ->
                          Ok(GmresResult(
                            solution: next_x,
                            iterations: arnoldi_result.steps,
                            residual_norm: r_norm,
                            converged: r_norm <=. tolerance,
                            happy_breakdown: arnoldi_result.happy_breakdown,
                          ))
                      }
                  }
              }
          }
      }
  }
}

fn solve_gmres_subproblem(
  arnoldi_result: ArnoldiResult,
  beta: Float,
) -> Result(Vector, NlaError) {
  let rhs = gmres_rhs(arnoldi_result.h.rows, beta)
  case least_squares.householder_qr(arnoldi_result.h, rhs) {
    Error(e) -> Error(e)
    Ok(ls) -> {
      let basis = leading_columns(arnoldi_result.q, arnoldi_result.steps)
      matrix.mul_vec(basis, ls.solution)
    }
  }
}

fn gmres_rhs(size: Int, beta: Float) -> Vector {
  vector.from_list(
    list.map(matrix.indices(size), fn(i) {
      case i == 0 {
        True -> beta
        False -> 0.0
      }
    }),
  )
}

fn leading_columns(q: Matrix, cols: Int) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: q.rows, cols: cols, with: fn(i, j) {
      matrix.unsafe_get(q, i, j)
    })
  result
}

fn orthogonalize(
  vectors: List(Vector),
  k: Int,
  j: Int,
  w: Vector,
  entries: List(#(Int, Int, Float)),
) -> Result(#(Vector, List(#(Int, Int, Float))), NlaError) {
  case j > k {
    True -> Ok(#(w, entries))
    False -> {
      let qj = unsafe_vector_at(vectors, j)
      case vector.dot(qj, w) {
        Error(e) -> Error(e)
        Ok(hjk) ->
          case vector.axpy(0.0 -. hjk, qj, w) {
            Error(e) -> Error(e)
            Ok(next_w) ->
              orthogonalize(vectors, k, j + 1, next_w, [#(j, k, hjk), ..entries])
          }
      }
    }
  }
}

fn build_result(
  rows: Int,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
  completed_steps: Int,
  happy_breakdown: Bool,
) -> Result(ArnoldiResult, NlaError) {
  case completed_steps <= 0 {
    True -> Error(InvalidInput("Arnoldi needs at least one completed step"))
    False -> {
      let q_cols = list.length(vectors)
      let assert Ok(q) =
        matrix.from_fn(rows: rows, cols: q_cols, with: fn(i, j) {
          unsafe_vector_get(unsafe_vector_at(vectors, j), i)
        })
      let assert Ok(h) =
        matrix.from_fn(rows: q_cols, cols: completed_steps, with: fn(i, j) {
          entry_value(entries, i, j)
        })
      Ok(ArnoldiResult(
        q: q,
        h: h,
        steps: completed_steps,
        happy_breakdown: happy_breakdown,
      ))
    }
  }
}

fn build_lanczos_result(
  rows: Int,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
  completed_steps: Int,
  happy_breakdown: Bool,
) -> Result(LanczosResult, NlaError) {
  case completed_steps <= 0 {
    True -> Error(InvalidInput("Lanczos needs at least one completed step"))
    False -> {
      let assert Ok(q) =
        matrix.from_fn(rows: rows, cols: completed_steps, with: fn(i, j) {
          unsafe_vector_get(unsafe_vector_at(vectors, j), i)
        })
      let assert Ok(t) =
        matrix.from_fn(
          rows: completed_steps,
          cols: completed_steps,
          with: fn(i, j) { entry_value(entries, i, j) },
        )
      Ok(LanczosResult(
        q: q,
        t: t,
        steps: completed_steps,
        happy_breakdown: happy_breakdown,
      ))
    }
  }
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

fn validate_symmetric(
  a: Matrix,
  initial: Vector,
  steps: Int,
  tolerance: Float,
) -> Result(Nil, NlaError) {
  case validate(a, initial, steps) {
    Error(e) -> Error(e)
    Ok(_) ->
      case is_symmetric(a, tolerance) {
        True -> Ok(Nil)
        False -> Error(InvalidInput("Lanczos matrix must be symmetric"))
      }
  }
}

fn is_symmetric(a: Matrix, tolerance: Float) -> Bool {
  list.all(matrix.indices(a.rows), satisfying: fn(i) {
    list.all(matrix.indices(a.cols), satisfying: fn(j) {
      float.absolute_value(
        matrix.unsafe_get(a, i, j) -. matrix.unsafe_get(a, j, i),
      )
      <=. tolerance
    })
  })
}

fn validate(a: Matrix, initial: Vector, steps: Int) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(a.rows, a.cols))
    True ->
      case a.rows == initial.size && steps > 0 {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "square matrix dimension "
              <> int.to_string(a.rows)
              <> " and positive steps",
            actual: "vector dimension "
              <> int.to_string(initial.size)
              <> ", steps "
              <> int.to_string(steps),
          ))
      }
  }
}

fn validate_system(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(a.rows, a.cols))
    True ->
      case a.rows == b.size && b.size == initial.size && max_iterations > 0 {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "square matrix dimension "
              <> int.to_string(a.rows)
              <> ", matching vectors and positive iterations",
            actual: "b="
              <> int.to_string(b.size)
              <> ", initial="
              <> int.to_string(initial.size)
              <> ", iterations="
              <> int.to_string(max_iterations),
          ))
      }
  }
}

fn min_int(a: Int, b: Int) -> Int {
  case a < b {
    True -> a
    False -> b
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
