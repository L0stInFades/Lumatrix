import gleam/float
import gleam/int
import gleam/list
import gleam/result
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, DimensionMismatch, InternalInvariant,
  InvalidInput, NonFiniteInput, NotSquare,
}
import lumatrix/error_analysis
import lumatrix/least_squares
import lumatrix/matrix.{type Matrix}
import lumatrix/numerics
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

type ScaledSystem {
  ScaledSystem(
    a: Matrix,
    b: Vector,
    initial: Vector,
    scale: Float,
    tolerance: Float,
  )
}

type MinresRotationScalars {
  MinresRotationScalars(
    delta: Float,
    gbar: Float,
    next_epsln: Float,
    next_dbar: Float,
  )
}

pub fn arnoldi(
  a: Matrix,
  initial: Vector,
  steps: Int,
  tolerance: Float,
) -> Result(ArnoldiResult, NlaError) {
  case validate(a, initial, steps, tolerance) {
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
  case validate_symmetric(a, initial, steps, tolerance, 1.0e-10) {
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
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      finish_scaled_result(
        gmres_cycle(
          system.a,
          system.b,
          system.initial,
          max_iterations,
          system.tolerance,
        ),
        system.scale,
      )
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
      case prepare_system(a, b, initial, max_iterations, tolerance) {
        Error(e) -> Error(e)
        Ok(system) ->
          finish_scaled_result(
            restarted_gmres_loop(
              system.a,
              system.b,
              system.initial,
              0,
              restart,
              max_iterations,
              system.tolerance,
              False,
            ),
            system.scale,
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
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      case error_analysis.residual(system.a, system.initial, system.b) {
        Error(e) -> Error(e)
        Ok(r0) ->
          finish_scaled_result(
            bicg_start(
              system.a,
              system.b,
              system.initial,
              r0,
              r0,
              max_iterations,
              system.tolerance,
            ),
            system.scale,
          )
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
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      case
        validate_shadow_residual(shadow_residual, vector.dimension(system.b))
      {
        Error(e) -> Error(e)
        Ok(_) ->
          case vector.divide(shadow_residual, system.scale) {
            Error(e) -> Error(e)
            Ok(scaled_shadow) ->
              case error_analysis.residual(system.a, system.initial, system.b) {
                Error(e) -> Error(e)
                Ok(r0) ->
                  finish_scaled_result(
                    bicg_start(
                      system.a,
                      system.b,
                      system.initial,
                      r0,
                      scaled_shadow,
                      max_iterations,
                      system.tolerance,
                    ),
                    system.scale,
                  )
              }
          }
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
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      case error_analysis.residual(system.a, system.initial, system.b) {
        Error(e) -> Error(e)
        Ok(r0) ->
          case vector.zeros(vector.dimension(system.b)) {
            Error(e) -> Error(e)
            Ok(zero) ->
              finish_scaled_result(
                bicgstab_loop(
                  system.a,
                  system.b,
                  system.initial,
                  r0,
                  r0,
                  zero,
                  zero,
                  1.0,
                  1.0,
                  1.0,
                  0,
                  max_iterations,
                  system.tolerance,
                ),
                system.scale,
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
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      case is_symmetric(system.a, 1.0e-10) {
        False -> Error(InvalidInput("MINRES matrix must be symmetric"))
        True ->
          case error_analysis.residual(system.a, system.initial, system.b) {
            Error(e) -> Error(e)
            Ok(r0) ->
              case vector.norm2(r0) {
                Error(e) -> Error(e)
                Ok(beta0) if beta0 <=. system.tolerance ->
                  finish_scaled_result(
                    Ok(GmresResult(
                      solution: system.initial,
                      iterations: 0,
                      residual_norm: beta0,
                      converged: True,
                      happy_breakdown: False,
                    )),
                    system.scale,
                  )
                Ok(beta0) -> {
                  let assert Ok(zero) = vector.zeros(matrix.rows(system.a))
                  finish_scaled_result(
                    minres_short_loop(
                      system.a,
                      system.b,
                      system.initial,
                      r0,
                      r0,
                      beta0,
                      0.0,
                      zero,
                      zero,
                      -1.0,
                      0.0,
                      0.0,
                      0.0,
                      beta0,
                      0,
                      max_iterations,
                      system.tolerance,
                      False,
                    ),
                    system.scale,
                  )
                }
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
                      case vector_breakdown(beta, aq, tolerance) {
                        Error(e) -> Error(e)
                        Ok(happy_breakdown) ->
                          case
                            happy_breakdown
                            || completed_steps >= requested_steps
                            || completed_steps >= matrix.rows(a)
                          {
                            True ->
                              build_lanczos_result(
                                matrix.rows(a),
                                vectors,
                                entries,
                                completed_steps,
                                happy_breakdown,
                              )
                            False -> {
                              let assert Ok(next_q) = vector.normalize(w)
                              lanczos_loop(
                                a,
                                requested_steps,
                                tolerance,
                                k + 1,
                                qk,
                                beta,
                                list.append(vectors, [next_q]),
                                [
                                  #(k, k + 1, beta),
                                  #(k + 1, k, beta),
                                  ..entries
                                ],
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
          case dot_breakdown(rho, shadow_residual, r0) {
            Error(e) -> Error(e)
            Ok(True) ->
              Error(InvalidInput(
                "BiCG shadow residual is orthogonal to the residual",
              ))
            Ok(False) ->
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
              case dot_breakdown(denominator, shadow_p, ap) {
                Error(e) -> Error(e)
                Ok(True) ->
                  Error(InvalidInput(
                    "BiCG breakdown: search directions are nearly A-orthogonal",
                  ))
                Ok(False) -> {
                  use alpha <- result.try(checked_scalar_divide(
                    rho,
                    denominator,
                    "BiCG alpha",
                  ))
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
          case dot_breakdown(next_rho, next_shadow_r, next_r) {
            Error(e) -> Error(e)
            Ok(True) ->
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
            Ok(False) -> {
              use beta <- result.try(checked_scalar_divide(
                next_rho,
                rho,
                "BiCG beta",
              ))
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
      case dot_breakdown(rho, shadow_r0, r) {
        Error(e) -> Error(e)
        Ok(True) ->
          Error(InvalidInput("BiCGSTAB breakdown: rho is nearly zero"))
        Ok(False) ->
          case float.absolute_value(omega) <=. 0.0 {
            True ->
              Error(InvalidInput("BiCGSTAB breakdown: omega is nearly zero"))
            False -> {
              use rho_ratio <- result.try(checked_scalar_divide(
                rho,
                rho_old,
                "BiCGSTAB beta rho ratio",
              ))
              use alpha_ratio <- result.try(checked_scalar_divide(
                alpha,
                omega,
                "BiCGSTAB beta alpha ratio",
              ))
              use beta <- result.try(checked_scalar_multiply(
                rho_ratio,
                alpha_ratio,
                "BiCGSTAB beta",
              ))
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
      case dot_breakdown(denominator, shadow_r0, v) {
        Error(e) -> Error(e)
        Ok(True) ->
          Error(InvalidInput(
            "BiCGSTAB breakdown: alpha denominator is nearly zero",
          ))
        Ok(False) -> {
          use next_alpha <- result.try(checked_scalar_divide(
            rho,
            denominator,
            "BiCGSTAB alpha",
          ))
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
      case dot_breakdown(tt, t, t) {
        Error(e) -> Error(e)
        Ok(True) ->
          Error(InvalidInput(
            "BiCGSTAB breakdown: stabilizing direction vanished",
          ))
        Ok(False) ->
          case vector.dot(t, s) {
            Error(e) -> Error(e)
            Ok(ts) -> {
              use omega <- result.try(checked_scalar_divide(
                ts,
                tt,
                "BiCGSTAB omega",
              ))
              case float.absolute_value(omega) <=. 0.0 {
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

fn minres_short_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  r_prev: Vector,
  r_curr: Vector,
  beta: Float,
  old_beta: Float,
  w_older: Vector,
  w_old: Vector,
  cs: Float,
  sn: Float,
  dbar: Float,
  epsln: Float,
  phibar: Float,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  happy_breakdown: Bool,
) -> Result(GmresResult, NlaError) {
  case iteration >= max_iterations || happy_breakdown {
    True -> finish_solver(a, b, x, iteration, tolerance, happy_breakdown)
    False ->
      case beta <=. 0.0 {
        True -> finish_solver(a, b, x, iteration, tolerance, True)
        False -> {
          let assert Ok(v) = vector.normalize(r_curr)
          case minres_lanczos_step(a, v, r_prev, beta, old_beta, iteration) {
            Error(e) -> Error(e)
            Ok(y0) ->
              case vector.dot(v, y0) {
                Error(e) -> Error(e)
                Ok(alpha) -> {
                  use coefficient <- result.try(checked_scalar_divide(
                    alpha,
                    beta,
                    "MINRES Lanczos coefficient",
                  ))
                  case vector.axpy(0.0 -. coefficient, r_curr, y0) {
                    Error(e) -> Error(e)
                    Ok(next_r) ->
                      case vector.norm2(next_r) {
                        Error(e) -> Error(e)
                        Ok(next_beta) ->
                          minres_rotate_and_update(
                            a,
                            b,
                            x,
                            r_curr,
                            next_r,
                            v,
                            beta,
                            next_beta,
                            w_older,
                            w_old,
                            cs,
                            sn,
                            dbar,
                            epsln,
                            phibar,
                            alpha,
                            iteration,
                            max_iterations,
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

fn minres_lanczos_step(
  a: Matrix,
  v: Vector,
  r_prev: Vector,
  beta: Float,
  old_beta: Float,
  iteration: Int,
) -> Result(Vector, NlaError) {
  case matrix.mul_vec(a, v) {
    Error(e) -> Error(e)
    Ok(av) ->
      case iteration >= 1 {
        True ->
          case old_beta <=. 0.0 {
            True -> Error(InternalInvariant("MINRES previous beta"))
            False -> {
              use coefficient <- result.try(checked_scalar_divide(
                beta,
                old_beta,
                "MINRES previous-vector coefficient",
              ))
              vector.axpy(0.0 -. coefficient, r_prev, av)
            }
          }
        False -> Ok(av)
      }
  }
}

fn minres_rotate_and_update(
  a: Matrix,
  b: Vector,
  x: Vector,
  r_curr: Vector,
  next_r: Vector,
  v: Vector,
  beta: Float,
  next_beta: Float,
  w_older: Vector,
  w_old: Vector,
  cs: Float,
  sn: Float,
  dbar: Float,
  epsln: Float,
  phibar: Float,
  alpha: Float,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  let oldeps = epsln
  use scalars <- result.try(minres_rotation_scalars(
    cs,
    sn,
    dbar,
    alpha,
    next_beta,
  ))
  case numerics.hypot(scalars.gbar, next_beta) {
    Error(_) -> Error(ArithmeticOverflow("MINRES rotation norm"))
    Ok(gamma) if gamma <=. 0.0 ->
      finish_solver(a, b, x, iteration, tolerance, True)
    Ok(gamma) -> {
      use next_cs <- result.try(checked_scalar_divide(
        scalars.gbar,
        gamma,
        "MINRES cosine",
      ))
      use next_sn <- result.try(checked_scalar_divide(
        next_beta,
        gamma,
        "MINRES sine",
      ))
      use phi <- result.try(checked_scalar_multiply(
        next_cs,
        phibar,
        "MINRES phi",
      ))
      use next_phibar <- result.try(checked_scalar_multiply(
        next_sn,
        phibar,
        "MINRES residual recurrence",
      ))
      case minres_direction(v, oldeps, w_older, scalars.delta, w_old, gamma) {
        Error(e) -> Error(e)
        Ok(w) ->
          case vector.axpy(phi, w, x) {
            Error(e) -> Error(e)
            Ok(next_x) ->
              minres_finish_short_iteration(
                a,
                b,
                next_x,
                r_curr,
                next_r,
                next_beta,
                beta,
                w_old,
                w,
                next_cs,
                next_sn,
                scalars.next_dbar,
                scalars.next_epsln,
                next_phibar,
                iteration + 1,
                max_iterations,
                tolerance,
              )
          }
      }
    }
  }
}

fn minres_direction(
  v: Vector,
  oldeps: Float,
  w_older: Vector,
  delta: Float,
  w_old: Vector,
  gamma: Float,
) -> Result(Vector, NlaError) {
  case vector.axpy(0.0 -. oldeps, w_older, v) {
    Error(e) -> Error(e)
    Ok(without_older) ->
      case vector.axpy(0.0 -. delta, w_old, without_older) {
        Error(e) -> Error(e)
        Ok(direction) -> vector.divide(direction, gamma)
      }
  }
}

fn minres_finish_short_iteration(
  a: Matrix,
  b: Vector,
  x: Vector,
  r_curr: Vector,
  next_r: Vector,
  beta: Float,
  old_beta: Float,
  w_older: Vector,
  w_old: Vector,
  cs: Float,
  sn: Float,
  dbar: Float,
  epsln: Float,
  phibar: Float,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(GmresResult, NlaError) {
  let happy_breakdown = beta <=. tolerance
  case error_analysis.residual_norm2(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(GmresResult(
        solution: x,
        iterations: iteration,
        residual_norm: r_norm,
        converged: True,
        happy_breakdown: happy_breakdown,
      ))
    Ok(r_norm) ->
      case iteration >= max_iterations || happy_breakdown {
        True ->
          Ok(GmresResult(
            solution: x,
            iterations: iteration,
            residual_norm: r_norm,
            converged: False,
            happy_breakdown: happy_breakdown,
          ))
        False ->
          minres_short_loop(
            a,
            b,
            x,
            r_curr,
            next_r,
            beta,
            old_beta,
            w_older,
            w_old,
            cs,
            sn,
            dbar,
            epsln,
            phibar,
            iteration,
            max_iterations,
            tolerance,
            happy_breakdown,
          )
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
                      let assert Ok(next_q) = vector.normalize(w)
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

fn dot_breakdown(
  value: Float,
  left: Vector,
  right: Vector,
) -> Result(Bool, NlaError) {
  case vector.norm2(left) {
    Error(e) -> Error(e)
    Ok(left_norm) ->
      case vector.norm2(right) {
        Error(e) -> Error(e)
        Ok(right_norm) if left_norm <=. 0.0 || right_norm <=. 0.0 -> Ok(True)
        Ok(right_norm) -> {
          let normalized_pairs =
            list.zip(vector.to_list(left), with: vector.to_list(right))
            |> list.map(fn(pair) {
              #(pair.0 /. left_norm, pair.1 /. right_norm)
            })
          case numerics.checked_dot_pairs(normalized_pairs) {
            Error(_) ->
              Error(ArithmeticOverflow("normalized Krylov dot product"))
            Ok(cosine) ->
              case float.absolute_value(cosine) <=. breakdown_tolerance {
                True -> Ok(True)
                False ->
                  case value == 0.0 {
                    True ->
                      Error(ArithmeticOverflow("Krylov dot product underflow"))
                    False -> Ok(False)
                  }
              }
          }
        }
      }
  }
}

fn vector_breakdown(
  value: Float,
  reference: Vector,
  tolerance: Float,
) -> Result(Bool, NlaError) {
  case vector.norm2(reference) {
    Error(e) -> Error(e)
    Ok(reference_norm) ->
      Ok(numerics.relative_near_zero(value, reference_norm, tolerance))
  }
}

fn prepare_system(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(ScaledSystem, NlaError) {
  case validate_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let raw_scale = float.max(matrix.norm_inf(a), vector.norm_inf(b))
      let scale = case raw_scale >. 0.0 {
        True -> raw_scale
        False -> 1.0
      }
      case matrix.divide(a, scale) {
        Error(e) -> Error(e)
        Ok(scaled_a) ->
          case vector.divide(b, scale) {
            Error(e) -> Error(e)
            Ok(scaled_b) ->
              Ok(ScaledSystem(
                a: scaled_a,
                b: scaled_b,
                initial: initial,
                scale: scale,
                tolerance: normalized_tolerance(tolerance, scale),
              ))
          }
      }
    }
  }
}

fn normalized_tolerance(tolerance: Float, scale: Float) -> Float {
  case tolerance <=. 0.0 {
    True -> 0.0
    False ->
      case numerics.checked_divide(tolerance, scale) {
        Ok(value) -> value
        Error(_) -> numerics.largest_finite()
      }
  }
}

fn finish_scaled_result(
  result: Result(GmresResult, NlaError),
  scale: Float,
) -> Result(GmresResult, NlaError) {
  case result {
    Error(e) -> Error(e)
    Ok(value) ->
      case vector.is_finite(value.solution) {
        False -> Error(ArithmeticOverflow("Krylov solution"))
        True ->
          case numerics.checked_multiply(value.residual_norm, scale) {
            Error(_) -> Error(ArithmeticOverflow("Krylov residual rescaling"))
            Ok(residual_norm) ->
              Ok(GmresResult(
                solution: value.solution,
                iterations: value.iterations,
                residual_norm: residual_norm,
                converged: value.converged,
                happy_breakdown: value.happy_breakdown,
              ))
          }
      }
  }
}

fn validate_shadow_residual(
  shadow_residual: Vector,
  expected_size: Int,
) -> Result(Nil, NlaError) {
  case vector.dimension(shadow_residual) == expected_size {
    False ->
      Error(DimensionMismatch(
        expected: "shadow residual dimension " <> int.to_string(expected_size),
        actual: int.to_string(vector.dimension(shadow_residual)),
      ))
    True ->
      case vector.is_finite(shadow_residual) {
        True -> Ok(Nil)
        False -> Error(NonFiniteInput("BiCG shadow residual"))
      }
  }
}

fn validate_symmetric(
  a: Matrix,
  initial: Vector,
  steps: Int,
  algorithm_tolerance: Float,
  symmetry_tolerance: Float,
) -> Result(Nil, NlaError) {
  case validate(a, initial, steps, algorithm_tolerance) {
    Error(e) -> Error(e)
    Ok(_) ->
      case is_symmetric(a, symmetry_tolerance) {
        True -> Ok(Nil)
        False -> Error(InvalidInput("Lanczos matrix must be symmetric"))
      }
  }
}

fn is_symmetric(a: Matrix, tolerance: Float) -> Bool {
  let scale = matrix.norm_inf(a)
  list.all(matrix.indices(matrix.rows(a)), satisfying: fn(i) {
    list.all(matrix.indices(matrix.cols(a)), satisfying: fn(j) {
      numerics.relative_close_at_scale(
        matrix.unsafe_get(a, i, j),
        matrix.unsafe_get(a, j, i),
        scale,
        tolerance,
      )
    })
  })
}

fn validate(
  a: Matrix,
  initial: Vector,
  steps: Int,
  tolerance: Float,
) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.rows(a) == vector.dimension(initial) {
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
        True ->
          case matrix.is_finite(a) && vector.is_finite(initial) {
            False -> Error(NonFiniteInput("Krylov basis inputs"))
            True -> validate_positive_options(steps, tolerance, "steps")
          }
      }
  }
}

fn validate_system(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case
        matrix.rows(a) == vector.dimension(b)
        && vector.dimension(b) == vector.dimension(initial)
      {
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
        True ->
          case
            matrix.is_finite(a)
            && vector.is_finite(b)
            && vector.is_finite(initial)
          {
            False -> Error(NonFiniteInput("Krylov linear system"))
            True ->
              validate_positive_options(
                max_iterations,
                tolerance,
                "max_iterations",
              )
          }
      }
  }
}

fn validate_positive_options(
  count: Int,
  tolerance: Float,
  count_name: String,
) -> Result(Nil, NlaError) {
  case count <= 0 {
    True -> Error(InvalidInput(count_name <> " must be positive"))
    False ->
      case numerics.is_finite(tolerance) {
        False -> Error(NonFiniteInput("Krylov tolerance"))
        True if tolerance <. 0.0 ->
          Error(InvalidInput("tolerance must be non-negative"))
        True -> Ok(Nil)
      }
  }
}

fn minres_rotation_scalars(
  cs: Float,
  sn: Float,
  dbar: Float,
  alpha: Float,
  next_beta: Float,
) -> Result(MinresRotationScalars, NlaError) {
  use cs_dbar <- result.try(checked_scalar_multiply(cs, dbar, "MINRES delta"))
  use sn_alpha <- result.try(checked_scalar_multiply(sn, alpha, "MINRES delta"))
  use delta <- result.try(checked_scalar_add(cs_dbar, sn_alpha, "MINRES delta"))
  use sn_dbar <- result.try(checked_scalar_multiply(sn, dbar, "MINRES gbar"))
  use cs_alpha <- result.try(checked_scalar_multiply(cs, alpha, "MINRES gbar"))
  use gbar <- result.try(checked_scalar_subtract(
    sn_dbar,
    cs_alpha,
    "MINRES gbar",
  ))
  use next_epsln <- result.try(checked_scalar_multiply(
    sn,
    next_beta,
    "MINRES epsilon recurrence",
  ))
  use next_dbar <- result.try(checked_scalar_multiply(
    0.0 -. cs,
    next_beta,
    "MINRES diagonal recurrence",
  ))
  Ok(MinresRotationScalars(
    delta: delta,
    gbar: gbar,
    next_epsln: next_epsln,
    next_dbar: next_dbar,
  ))
}

fn checked_scalar_add(
  left: Float,
  right: Float,
  operation: String,
) -> Result(Float, NlaError) {
  case numerics.checked_add(left, right) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
  }
}

fn checked_scalar_subtract(
  left: Float,
  right: Float,
  operation: String,
) -> Result(Float, NlaError) {
  case numerics.checked_subtract(left, right) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
  }
}

fn checked_scalar_multiply(
  left: Float,
  right: Float,
  operation: String,
) -> Result(Float, NlaError) {
  case numerics.checked_multiply(left, right) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
  }
}

fn checked_scalar_divide(
  numerator: Float,
  denominator: Float,
  operation: String,
) -> Result(Float, NlaError) {
  case numerics.checked_divide(numerator, denominator) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
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
