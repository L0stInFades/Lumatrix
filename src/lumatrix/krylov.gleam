import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{type NlaError, DimensionMismatch, InvalidInput, NotSquare}
import lumatrix/error_analysis
import lumatrix/least_squares
import lumatrix/matrix.{type Matrix}
import lumatrix/vector.{type Vector}

const breakdown_tolerance = 1.0e-12

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
          let assert Ok(q_prev) = vector.zeros(matrix.rows(a))
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

pub fn bicg(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case validate_system(a, b, initial, max_iterations) {
    Error(e) -> Error(e)
    Ok(_) ->
      case error_analysis.residual(a, initial, b) {
        Error(e) -> Error(e)
        Ok(r0) -> bicg_start(a, b, initial, r0, r0, max_iterations, tolerance)
      }
  }
}

pub fn bicg_with_shadow(
  a: Matrix,
  b: Vector,
  initial: Vector,
  shadow_residual: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case validate_bicg_system(a, b, initial, shadow_residual, max_iterations) {
    Error(e) -> Error(e)
    Ok(_) ->
      case error_analysis.residual(a, initial, b) {
        Error(e) -> Error(e)
        Ok(r0) ->
          bicg_start(
            a,
            b,
            initial,
            r0,
            shadow_residual,
            max_iterations,
            tolerance,
          )
      }
  }
}

pub fn bicgstab(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case validate_system(a, b, initial, max_iterations) {
    Error(e) -> Error(e)
    Ok(_) ->
      case error_analysis.residual(a, initial, b) {
        Error(e) -> Error(e)
        Ok(r0) ->
          case vector.zeros(vector.dimension(b)) {
            Error(e) -> Error(e)
            Ok(zero) ->
              bicgstab_loop(
                a,
                b,
                initial,
                r0,
                r0,
                zero,
                zero,
                1.0,
                1.0,
                1.0,
                0,
                max_iterations,
                tolerance,
              )
          }
      }
  }
}

pub fn minres(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case validate_symmetric_system(a, b, initial, max_iterations, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) ->
      case error_analysis.residual(a, initial, b) {
        Error(e) -> Error(e)
        Ok(r0) ->
          case vector.norm2(r0) {
            Error(e) -> Error(e)
            Ok(beta0) if beta0 <=. tolerance ->
              Ok(GmresResult(
                solution: initial,
                iterations: 0,
                residual_norm: beta0,
                converged: True,
                happy_breakdown: False,
              ))
            Ok(beta0) -> {
              let q0 = vector.scale(r0, 1.0 /. beta0)
              let assert Ok(q_prev) = vector.zeros(matrix.rows(a))
              minres_loop(
                a,
                b,
                initial,
                beta0,
                q_prev,
                0.0,
                [q0],
                [],
                0,
                max_iterations,
                tolerance,
              )
            }
          }
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
                        || completed_steps >= matrix.rows(a)
                      {
                        True ->
                          build_lanczos_result(
                            matrix.rows(a),
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

fn bicg_start(
  a: Matrix,
  b: Vector,
  initial: Vector,
  r0: Vector,
  shadow_residual: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case vector.norm2(r0) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(GmresResult(
        solution: initial,
        iterations: 0,
        residual_norm: r_norm,
        converged: True,
        happy_breakdown: False,
      ))
    Ok(_) ->
      case vector.dot(shadow_residual, r0) {
        Error(e) -> Error(e)
        Ok(rho) ->
          case float.absolute_value(rho) <=. breakdown_tolerance {
            True ->
              Error(InvalidInput(
                "BiCG shadow residual is orthogonal to the residual",
              ))
            False ->
              bicg_loop(
                a,
                b,
                initial,
                r0,
                shadow_residual,
                r0,
                shadow_residual,
                rho,
                0,
                max_iterations,
                tolerance,
              )
          }
      }
  }
}

fn bicg_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  r: Vector,
  shadow_r: Vector,
  p: Vector,
  shadow_p: Vector,
  rho: Float,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case vector.norm2(r) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      finish_solver(a, b, x, iteration, tolerance, False)
    Ok(_) ->
      case iteration >= max_iterations {
        True -> finish_solver(a, b, x, iteration, tolerance, False)
        False ->
          case bicg_step(a, x, r, shadow_r, p, shadow_p, rho, tolerance) {
            Error(e) -> Error(e)
            Ok(next) ->
              bicg_loop(
                a,
                b,
                next.0,
                next.1,
                next.2,
                next.3,
                next.4,
                next.5,
                iteration + 1,
                max_iterations,
                tolerance,
              )
          }
      }
  }
}

fn bicg_step(
  a: Matrix,
  x: Vector,
  r: Vector,
  shadow_r: Vector,
  p: Vector,
  shadow_p: Vector,
  rho: Float,
  tolerance: Float,
) -> Result(#(Vector, Vector, Vector, Vector, Vector, Float), NlaError) {
  case matrix.mul_vec(a, p) {
    Error(e) -> Error(e)
    Ok(ap) ->
      case matrix.transpose_mul_vec(a, shadow_p) {
        Error(e) -> Error(e)
        Ok(at_shadow_p) ->
          case vector.dot(shadow_p, ap) {
            Error(e) -> Error(e)
            Ok(denominator) ->
              case float.absolute_value(denominator) <=. breakdown_tolerance {
                True ->
                  Error(InvalidInput(
                    "BiCG breakdown: search directions are nearly A-orthogonal",
                  ))
                False -> {
                  let alpha = rho /. denominator
                  case vector.axpy(alpha, p, x) {
                    Error(e) -> Error(e)
                    Ok(next_x) ->
                      case vector.axpy(0.0 -. alpha, ap, r) {
                        Error(e) -> Error(e)
                        Ok(next_r) ->
                          case
                            vector.axpy(0.0 -. alpha, at_shadow_p, shadow_r)
                          {
                            Error(e) -> Error(e)
                            Ok(next_shadow_r) ->
                              bicg_finish_step(
                                next_x,
                                next_r,
                                next_shadow_r,
                                p,
                                shadow_p,
                                rho,
                                tolerance,
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

fn bicg_finish_step(
  next_x: Vector,
  next_r: Vector,
  next_shadow_r: Vector,
  p: Vector,
  shadow_p: Vector,
  rho: Float,
  tolerance: Float,
) -> Result(#(Vector, Vector, Vector, Vector, Vector, Float), NlaError) {
  case vector.dot(next_shadow_r, next_r) {
    Error(e) -> Error(e)
    Ok(next_rho) ->
      case vector.norm2(next_r) {
        Error(e) -> Error(e)
        Ok(r_norm) ->
          case float.absolute_value(next_rho) <=. breakdown_tolerance {
            True ->
              case r_norm <=. tolerance {
                True ->
                  Ok(#(
                    next_x,
                    next_r,
                    next_shadow_r,
                    next_r,
                    next_shadow_r,
                    next_rho,
                  ))
                False ->
                  Error(InvalidInput(
                    "BiCG breakdown: shadow residual became orthogonal",
                  ))
              }
            False -> {
              let beta = next_rho /. rho
              case vector.axpy(beta, p, next_r) {
                Error(e) -> Error(e)
                Ok(next_p) ->
                  case vector.axpy(beta, shadow_p, next_shadow_r) {
                    Error(e) -> Error(e)
                    Ok(next_shadow_p) ->
                      Ok(#(
                        next_x,
                        next_r,
                        next_shadow_r,
                        next_p,
                        next_shadow_p,
                        next_rho,
                      ))
                  }
              }
            }
          }
      }
  }
}

fn bicgstab_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  r: Vector,
  shadow_r0: Vector,
  p: Vector,
  v: Vector,
  rho_old: Float,
  alpha: Float,
  omega: Float,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  case vector.norm2(r) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      finish_solver(a, b, x, iteration, tolerance, False)
    Ok(_) ->
      case iteration >= max_iterations {
        True -> finish_solver(a, b, x, iteration, tolerance, False)
        False ->
          case
            bicgstab_step(
              a,
              b,
              x,
              r,
              shadow_r0,
              p,
              v,
              rho_old,
              alpha,
              omega,
              iteration,
              tolerance,
            )
          {
            Error(e) -> Error(e)
            Ok(IterationComplete(result)) -> Ok(result)
            Ok(IterationContinue(state)) ->
              bicgstab_loop(
                a,
                b,
                state.x,
                state.r,
                shadow_r0,
                state.p,
                state.v,
                state.rho,
                state.alpha,
                state.omega,
                iteration + 1,
                max_iterations,
                tolerance,
              )
          }
      }
  }
}

type BicgstabState {
  BicgstabState(
    x: Vector,
    r: Vector,
    p: Vector,
    v: Vector,
    rho: Float,
    alpha: Float,
    omega: Float,
  )
}

type IterationStep {
  IterationComplete(GmresResult)
  IterationContinue(BicgstabState)
}

fn bicgstab_step(
  a: Matrix,
  b: Vector,
  x: Vector,
  r: Vector,
  shadow_r0: Vector,
  p: Vector,
  v: Vector,
  rho_old: Float,
  alpha: Float,
  omega: Float,
  iteration: Int,
  tolerance: Float,
) -> Result(IterationStep, NlaError) {
  case vector.dot(shadow_r0, r) {
    Error(e) -> Error(e)
    Ok(rho) ->
      case float.absolute_value(rho) <=. breakdown_tolerance {
        True -> Error(InvalidInput("BiCGSTAB breakdown: rho is nearly zero"))
        False ->
          case float.absolute_value(omega) <=. breakdown_tolerance {
            True ->
              Error(InvalidInput("BiCGSTAB breakdown: omega is nearly zero"))
            False -> {
              let beta = rho /. rho_old *. alpha /. omega
              case bicgstab_search_direction(r, p, v, beta, omega) {
                Error(e) -> Error(e)
                Ok(next_p) ->
                  case matrix.mul_vec(a, next_p) {
                    Error(e) -> Error(e)
                    Ok(next_v) ->
                      bicgstab_stabilize(
                        a,
                        b,
                        x,
                        r,
                        shadow_r0,
                        next_p,
                        next_v,
                        rho,
                        iteration,
                        tolerance,
                      )
                  }
              }
            }
          }
      }
  }
}

fn bicgstab_search_direction(
  r: Vector,
  p: Vector,
  v: Vector,
  beta: Float,
  omega: Float,
) -> Result(Vector, NlaError) {
  case vector.axpy(0.0 -. omega, v, p) {
    Error(e) -> Error(e)
    Ok(p_minus_omega_v) -> vector.axpy(beta, p_minus_omega_v, r)
  }
}

fn bicgstab_stabilize(
  a: Matrix,
  b: Vector,
  x: Vector,
  r: Vector,
  shadow_r0: Vector,
  p: Vector,
  v: Vector,
  rho: Float,
  iteration: Int,
  tolerance: Float,
) -> Result(IterationStep, NlaError) {
  case vector.dot(shadow_r0, v) {
    Error(e) -> Error(e)
    Ok(denominator) ->
      case float.absolute_value(denominator) <=. breakdown_tolerance {
        True ->
          Error(InvalidInput(
            "BiCGSTAB breakdown: alpha denominator is nearly zero",
          ))
        False -> {
          let next_alpha = rho /. denominator
          case vector.axpy(0.0 -. next_alpha, v, r) {
            Error(e) -> Error(e)
            Ok(s) ->
              bicgstab_after_s(
                a,
                b,
                x,
                s,
                p,
                v,
                rho,
                next_alpha,
                iteration,
                tolerance,
              )
          }
        }
      }
  }
}

fn bicgstab_after_s(
  a: Matrix,
  b: Vector,
  x: Vector,
  s: Vector,
  p: Vector,
  v: Vector,
  rho: Float,
  alpha: Float,
  iteration: Int,
  tolerance: Float,
) -> Result(IterationStep, NlaError) {
  case vector.norm2(s) {
    Error(e) -> Error(e)
    Ok(s_norm) if s_norm <=. tolerance ->
      case vector.axpy(alpha, p, x) {
        Error(e) -> Error(e)
        Ok(next_x) ->
          case finish_solver(a, b, next_x, iteration + 1, tolerance, False) {
            Error(e) -> Error(e)
            Ok(result) -> Ok(IterationComplete(result))
          }
      }
    Ok(_) ->
      case matrix.mul_vec(a, s) {
        Error(e) -> Error(e)
        Ok(t) -> bicgstab_after_t(x, s, t, p, v, rho, alpha)
      }
  }
}

fn bicgstab_after_t(
  x: Vector,
  s: Vector,
  t: Vector,
  p: Vector,
  v: Vector,
  rho: Float,
  alpha: Float,
) -> Result(IterationStep, NlaError) {
  case vector.dot(t, t) {
    Error(e) -> Error(e)
    Ok(tt) ->
      case float.absolute_value(tt) <=. breakdown_tolerance {
        True ->
          Error(InvalidInput(
            "BiCGSTAB breakdown: stabilizing direction vanished",
          ))
        False ->
          case vector.dot(t, s) {
            Error(e) -> Error(e)
            Ok(ts) -> {
              let omega = ts /. tt
              case float.absolute_value(omega) <=. breakdown_tolerance {
                True ->
                  Error(InvalidInput("BiCGSTAB breakdown: omega is nearly zero"))
                False -> bicgstab_finish_step(x, s, t, p, v, rho, alpha, omega)
              }
            }
          }
      }
  }
}

fn bicgstab_finish_step(
  x: Vector,
  s: Vector,
  t: Vector,
  p: Vector,
  v: Vector,
  rho: Float,
  alpha: Float,
  omega: Float,
) -> Result(IterationStep, NlaError) {
  case vector.axpy(alpha, p, x) {
    Error(e) -> Error(e)
    Ok(x_alpha) ->
      case vector.axpy(omega, s, x_alpha) {
        Error(e) -> Error(e)
        Ok(next_x) ->
          case vector.axpy(0.0 -. omega, t, s) {
            Error(e) -> Error(e)
            Ok(next_r) ->
              Ok(
                IterationContinue(BicgstabState(
                  x: next_x,
                  r: next_r,
                  p: p,
                  v: v,
                  rho: rho,
                  alpha: alpha,
                  omega: omega,
                )),
              )
          }
      }
  }
}

fn minres_loop(
  a: Matrix,
  b: Vector,
  initial: Vector,
  beta0: Float,
  q_prev: Vector,
  beta_prev: Float,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
  k: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  let qk = unsafe_vector_at(vectors, k)
  case matrix.mul_vec(a, qk) {
    Error(e) -> Error(e)
    Ok(aq) ->
      case vector.axpy(0.0 -. beta_prev, q_prev, aq) {
        Error(e) -> Error(e)
        Ok(w0) ->
          case vector.dot(qk, w0) {
            Error(e) -> Error(e)
            Ok(alpha) ->
              case vector.axpy(0.0 -. alpha, qk, w0) {
                Error(e) -> Error(e)
                Ok(w) ->
                  case vector.norm2(w) {
                    Error(e) -> Error(e)
                    Ok(beta_next) ->
                      minres_finish_iteration(
                        a,
                        b,
                        initial,
                        beta0,
                        qk,
                        w,
                        alpha,
                        beta_prev,
                        beta_next,
                        vectors,
                        entries,
                        k,
                        max_iterations,
                        tolerance,
                      )
                  }
              }
          }
      }
  }
}

fn minres_finish_iteration(
  a: Matrix,
  b: Vector,
  initial: Vector,
  beta0: Float,
  qk: Vector,
  w: Vector,
  alpha: Float,
  beta_prev: Float,
  beta_next: Float,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
  k: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  let completed_steps = k + 1
  let entries = minres_entries(entries, k, alpha, beta_prev, beta_next)
  case solve_minres_subproblem(matrix.rows(a), vectors, entries, beta0, k) {
    Error(e) -> Error(e)
    Ok(next_x_delta) ->
      case vector.add(initial, next_x_delta) {
        Error(e) -> Error(e)
        Ok(next_x) ->
          case error_analysis.residual_norm2(a, next_x, b) {
            Error(e) -> Error(e)
            Ok(r_norm) if r_norm <=. tolerance ->
              Ok(GmresResult(
                solution: next_x,
                iterations: completed_steps,
                residual_norm: r_norm,
                converged: True,
                happy_breakdown: beta_next <=. tolerance,
              ))
            Ok(r_norm) ->
              case
                completed_steps >= max_iterations
                || completed_steps >= matrix.rows(a)
                || beta_next <=. tolerance
              {
                True ->
                  Ok(GmresResult(
                    solution: next_x,
                    iterations: completed_steps,
                    residual_norm: r_norm,
                    converged: False,
                    happy_breakdown: beta_next <=. tolerance,
                  ))
                False -> {
                  let next_q = vector.scale(w, 1.0 /. beta_next)
                  minres_loop(
                    a,
                    b,
                    initial,
                    beta0,
                    qk,
                    beta_next,
                    list.append(vectors, [next_q]),
                    entries,
                    k + 1,
                    max_iterations,
                    tolerance,
                  )
                }
              }
          }
      }
  }
}

fn minres_entries(
  entries: List(#(Int, Int, Float)),
  k: Int,
  alpha: Float,
  beta_prev: Float,
  beta_next: Float,
) -> List(#(Int, Int, Float)) {
  let entries = [#(k, k, alpha), #(k + 1, k, beta_next), ..entries]
  case k > 0 {
    True -> [#(k - 1, k, beta_prev), ..entries]
    False -> entries
  }
}

fn solve_minres_subproblem(
  rows: Int,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
  beta0: Float,
  k: Int,
) -> Result(Vector, NlaError) {
  let completed_steps = k + 1
  let assert Ok(t_bar) =
    matrix.from_fn(
      rows: completed_steps + 1,
      cols: completed_steps,
      with: fn(i, j) { entry_value(entries, i, j) },
    )
  let rhs = gmres_rhs(completed_steps + 1, beta0)
  case least_squares.householder_qr(t_bar, rhs) {
    Error(e) -> Error(e)
    Ok(ls) -> {
      let basis = basis_from_vectors(rows, vectors, completed_steps)
      matrix.mul_vec(basis, ls.solution)
    }
  }
}

fn basis_from_vectors(rows: Int, vectors: List(Vector), cols: Int) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: rows, cols: cols, with: fn(i, j) {
      unsafe_vector_get(unsafe_vector_at(vectors, j), i)
    })
  result
}

fn arnoldi_loop(
  a: Matrix,
  requested_steps: Int,
  tolerance: Float,
  k: Int,
  vectors: List(Vector),
  entries: List(#(Int, Int, Float)),
) -> Result(ArnoldiResult, NlaError) {
  case k >= requested_steps || k >= matrix.rows(a) {
    True -> build_result(matrix.rows(a), vectors, entries, k, False)
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
                    True ->
                      build_result(
                        matrix.rows(a),
                        vectors,
                        entries,
                        k + 1,
                        True,
                      )
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
  let rhs = gmres_rhs(matrix.rows(arnoldi_result.h), beta)
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
    matrix.from_fn(rows: matrix.rows(q), cols: cols, with: fn(i, j) {
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

fn finish_solver(
  a: Matrix,
  b: Vector,
  x: Vector,
  iterations: Int,
  tolerance: Float,
  happy_breakdown: Bool,
) -> Result(GmresResult, NlaError) {
  case error_analysis.residual_norm2(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) ->
      Ok(GmresResult(
        solution: x,
        iterations: iterations,
        residual_norm: r_norm,
        converged: r_norm <=. tolerance,
        happy_breakdown: happy_breakdown,
      ))
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

fn validate_symmetric_system(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(Nil, NlaError) {
  case validate_system(a, b, initial, max_iterations) {
    Error(e) -> Error(e)
    Ok(_) ->
      case is_symmetric(a, tolerance) {
        True -> Ok(Nil)
        False -> Error(InvalidInput("MINRES matrix must be symmetric"))
      }
  }
}

fn is_symmetric(a: Matrix, tolerance: Float) -> Bool {
  list.all(matrix.indices(matrix.rows(a)), satisfying: fn(i) {
    list.all(matrix.indices(matrix.cols(a)), satisfying: fn(j) {
      float.absolute_value(
        matrix.unsafe_get(a, i, j) -. matrix.unsafe_get(a, j, i),
      )
      <=. tolerance
    })
  })
}

fn validate(a: Matrix, initial: Vector, steps: Int) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.rows(a) == vector.dimension(initial) && steps > 0 {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "square matrix dimension "
              <> int.to_string(matrix.rows(a))
              <> " and positive steps",
            actual: "vector dimension "
              <> int.to_string(vector.dimension(initial))
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
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case
        matrix.rows(a) == vector.dimension(b)
        && vector.dimension(b) == vector.dimension(initial)
        && max_iterations > 0
      {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "square matrix dimension "
              <> int.to_string(matrix.rows(a))
              <> ", matching vectors and positive iterations",
            actual: "b="
              <> int.to_string(vector.dimension(b))
              <> ", initial="
              <> int.to_string(vector.dimension(initial))
              <> ", iterations="
              <> int.to_string(max_iterations),
          ))
      }
  }
}

fn validate_bicg_system(
  a: Matrix,
  b: Vector,
  initial: Vector,
  shadow_residual: Vector,
  max_iterations: Int,
) -> Result(Nil, NlaError) {
  case validate_system(a, b, initial, max_iterations) {
    Error(e) -> Error(e)
    Ok(_) ->
      case vector.dimension(shadow_residual) == vector.dimension(b) {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "shadow residual dimension "
              <> int.to_string(vector.dimension(b)),
            actual: int.to_string(vector.dimension(shadow_residual)),
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

fn unsafe_vector_get(values: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
}
