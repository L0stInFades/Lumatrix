import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, DimensionMismatch, InternalInvariant,
  InvalidInput, NonFiniteInput, NotSquare, SingularMatrix,
}
import lumatrix/error_analysis
import lumatrix/matrix.{type Matrix}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

const diagonal_tolerance = 1.0e-12

pub type IterationResult {
  IterationResult(
    solution: Vector,
    iterations: Int,
    residual_norm: Float,
    converged: Bool,
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

pub type StationaryMethod {
  JacobiIteration
  GaussSeidelIteration
  SorIteration(omega: Float)
}

pub type StationaryConvergenceDiagnostics {
  StationaryConvergenceDiagnostics(
    iteration_matrix: Matrix,
    infinity_norm_bound: Float,
    sufficient_convergence: Bool,
  )
}

pub fn jacobi(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      finish_scaled(
        stationary_loop(
          system.a,
          system.b,
          system.initial,
          max_iterations,
          system.tolerance,
          jacobi_step,
        ),
        system.scale,
      )
  }
}

pub fn gauss_seidel(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      finish_scaled(
        stationary_loop(
          system.a,
          system.b,
          system.initial,
          max_iterations,
          system.tolerance,
          gauss_seidel_step,
        ),
        system.scale,
      )
  }
}

pub fn sor(
  a: Matrix,
  b: Vector,
  initial: Vector,
  omega: Float,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case validate_omega(omega) {
    Error(e) -> Error(e)
    Ok(_) ->
      case prepare_system(a, b, initial, max_iterations, tolerance) {
        Error(e) -> Error(e)
        Ok(system) ->
          finish_scaled(
            stationary_loop(
              system.a,
              system.b,
              system.initial,
              max_iterations,
              system.tolerance,
              fn(a, b, x) { sor_step(a, b, x, omega) },
            ),
            system.scale,
          )
      }
  }
}

pub fn jacobi_convergence_diagnostics(
  a: Matrix,
) -> Result(StationaryConvergenceDiagnostics, NlaError) {
  stationary_convergence_diagnostics(a, JacobiIteration)
}

pub fn gauss_seidel_convergence_diagnostics(
  a: Matrix,
) -> Result(StationaryConvergenceDiagnostics, NlaError) {
  stationary_convergence_diagnostics(a, GaussSeidelIteration)
}

pub fn sor_convergence_diagnostics(
  a: Matrix,
  omega: Float,
) -> Result(StationaryConvergenceDiagnostics, NlaError) {
  stationary_convergence_diagnostics(a, SorIteration(omega))
}

pub fn stationary_convergence_diagnostics(
  a: Matrix,
  method: StationaryMethod,
) -> Result(StationaryConvergenceDiagnostics, NlaError) {
  case validate_stationary_matrix(a) {
    Error(e) -> Error(e)
    Ok(_) ->
      case validate_stationary_method(method) {
        Error(e) -> Error(e)
        Ok(_) ->
          case stationary_iteration_matrix(a, method) {
            Error(e) -> Error(e)
            Ok(iteration_matrix) -> {
              let bound = matrix.norm_inf(iteration_matrix)
              Ok(StationaryConvergenceDiagnostics(
                iteration_matrix: iteration_matrix,
                infinity_norm_bound: bound,
                sufficient_convergence: bound <. 1.0,
              ))
            }
          }
      }
  }
}

pub fn steepest_descent(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      finish_scaled(
        steepest_loop(
          system.a,
          system.b,
          system.initial,
          0,
          max_iterations,
          system.tolerance,
        ),
        system.scale,
      )
  }
}

pub fn conjugate_gradient(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      case error_analysis.residual(system.a, system.initial, system.b) {
        Error(e) -> Error(e)
        Ok(r) ->
          finish_scaled(
            conjugate_gradient_loop(
              system.a,
              system.initial,
              r,
              r,
              0,
              max_iterations,
              system.tolerance,
            ),
            system.scale,
          )
      }
  }
}

pub fn practical_conjugate_gradient(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
  recompute_every: Int,
) -> Result(IterationResult, NlaError) {
  case recompute_every <= 0 {
    True -> Error(InvalidInput("recompute_every must be positive"))
    False ->
      case prepare_system(a, b, initial, max_iterations, tolerance) {
        Error(e) -> Error(e)
        Ok(system) ->
          case error_analysis.residual(system.a, system.initial, system.b) {
            Error(e) -> Error(e)
            Ok(r) ->
              finish_scaled(
                practical_cg_loop(
                  system.a,
                  system.b,
                  system.initial,
                  r,
                  r,
                  0,
                  max_iterations,
                  system.tolerance,
                  recompute_every,
                ),
                system.scale,
              )
          }
      }
  }
}

pub fn preconditioned_conjugate_gradient(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) ->
      finish_scaled(
        run_pcg(system, max_iterations, fn(r) {
          jacobi_precondition(system.a, r)
        }),
        system.scale,
      )
  }
}

pub fn preconditioned_conjugate_gradient_with(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
  preconditioner: fn(Vector) -> Result(Vector, NlaError),
) -> Result(IterationResult, NlaError) {
  case prepare_system(a, b, initial, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(system) -> {
      let scaled_preconditioner = fn(residual) {
        case vector.checked_scale(residual, system.scale) {
          Error(e) -> Error(e)
          Ok(original_residual) -> preconditioner(original_residual)
        }
      }
      finish_scaled(
        run_pcg(system, max_iterations, scaled_preconditioner),
        system.scale,
      )
    }
  }
}

fn run_pcg(
  system: ScaledSystem,
  max_iterations: Int,
  preconditioner: fn(Vector) -> Result(Vector, NlaError),
) -> Result(IterationResult, NlaError) {
  let checked_preconditioner = fn(residual) {
    apply_preconditioner(
      preconditioner,
      residual,
      vector.dimension(system.initial),
    )
  }
  case error_analysis.residual(system.a, system.initial, system.b) {
    Error(e) -> Error(e)
    Ok(r) ->
      case checked_preconditioner(r) {
        Error(e) -> Error(e)
        Ok(z) ->
          pcg_loop(
            system.a,
            system.initial,
            r,
            z,
            z,
            0,
            max_iterations,
            system.tolerance,
            checked_preconditioner,
          )
      }
  }
}

fn apply_preconditioner(
  preconditioner: fn(Vector) -> Result(Vector, NlaError),
  residual: Vector,
  expected_size: Int,
) -> Result(Vector, NlaError) {
  case preconditioner(residual) {
    Error(e) -> Error(e)
    Ok(result) ->
      case vector.dimension(result) == expected_size {
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(expected_size),
            actual: int.to_string(vector.dimension(result)),
          ))
        True ->
          case vector.is_finite(result) {
            True -> Ok(result)
            False -> Error(NonFiniteInput("preconditioner output"))
          }
      }
  }
}

fn stationary_iteration_matrix(
  a: Matrix,
  method: StationaryMethod,
) -> Result(Matrix, NlaError) {
  case vector.zeros(matrix.rows(a)) {
    Error(_) -> Error(InternalInvariant("stationary matrix dimensions"))
    Ok(zero_b) -> build_stationary_columns(a, zero_b, method, 0, [])
  }
}

fn build_stationary_columns(
  a: Matrix,
  zero_b: Vector,
  method: StationaryMethod,
  j: Int,
  columns: List(Vector),
) -> Result(Matrix, NlaError) {
  case j >= matrix.cols(a) {
    True -> columns_to_matrix(matrix.rows(a), matrix.cols(a), columns)
    False ->
      case vector.basis(matrix.cols(a), j) {
        Error(e) -> Error(e)
        Ok(basis) ->
          case stationary_method_step(method, a, zero_b, basis) {
            Error(e) -> Error(e)
            Ok(column) ->
              build_stationary_columns(
                a,
                zero_b,
                method,
                j + 1,
                list.append(columns, [column]),
              )
          }
      }
  }
}

fn stationary_method_step(
  method: StationaryMethod,
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(Vector, NlaError) {
  case method {
    JacobiIteration -> jacobi_step(a, b, x)
    GaussSeidelIteration -> gauss_seidel_step(a, b, x)
    SorIteration(omega) -> sor_step(a, b, x, omega)
  }
}

fn columns_to_matrix(
  rows: Int,
  cols: Int,
  columns: List(Vector),
) -> Result(Matrix, NlaError) {
  case
    list.length(columns) == cols
    && list.all(columns, satisfying: fn(column) {
      vector.dimension(column) == rows
    })
  {
    True -> matrix.from_columns(columns)
    False -> Error(InternalInvariant("stationary iteration matrix columns"))
  }
}

fn stationary_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  iteration: Int,
  tolerance: Float,
  step: fn(Matrix, Vector, Vector) -> Result(Vector, NlaError),
) -> Result(IterationResult, NlaError) {
  case residual_norm(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(IterationResult(
        solution: x,
        iterations: 0,
        residual_norm: r_norm,
        converged: True,
      ))
    Ok(_) -> stationary_loop_step(a, b, x, 0, iteration, tolerance, step)
  }
}

fn stationary_loop_step(
  a: Matrix,
  b: Vector,
  x: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  step: fn(Matrix, Vector, Vector) -> Result(Vector, NlaError),
) -> Result(IterationResult, NlaError) {
  case iteration >= max_iterations {
    True ->
      case residual_norm(a, x, b) {
        Ok(r_norm) ->
          Ok(IterationResult(
            solution: x,
            iterations: iteration,
            residual_norm: r_norm,
            converged: False,
          ))
        Error(e) -> Error(e)
      }
    False ->
      case step(a, b, x) {
        Error(e) -> Error(e)
        Ok(next) ->
          case residual_norm(a, next, b) {
            Error(e) -> Error(e)
            Ok(r_norm) if r_norm <=. tolerance ->
              Ok(IterationResult(
                solution: next,
                iterations: iteration + 1,
                residual_norm: r_norm,
                converged: True,
              ))
            Ok(_) ->
              stationary_loop_step(
                a,
                b,
                next,
                iteration + 1,
                max_iterations,
                tolerance,
                step,
              )
          }
      }
  }
}

fn jacobi_step(a: Matrix, b: Vector, x: Vector) -> Result(Vector, NlaError) {
  build_stationary_step(a, b, x, [], 0, False, fn(_, _old, candidate) {
    Ok(candidate)
  })
}

fn gauss_seidel_step(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(Vector, NlaError) {
  build_stationary_step(a, b, x, [], 0, True, fn(_, _, candidate) {
    Ok(candidate)
  })
}

fn sor_step(
  a: Matrix,
  b: Vector,
  x: Vector,
  omega: Float,
) -> Result(Vector, NlaError) {
  build_stationary_step(a, b, x, [], 0, True, fn(i, old, candidate) {
    case numerics.checked_multiply(1.0 -. omega, unsafe_vector_get(old, i)) {
      Error(_) -> Error(ArithmeticOverflow("SOR update"))
      Ok(old_part) ->
        case numerics.checked_multiply(omega, candidate) {
          Error(_) -> Error(ArithmeticOverflow("SOR update"))
          Ok(candidate_part) ->
            case numerics.checked_add(old_part, candidate_part) {
              Ok(value) -> Ok(value)
              Error(_) -> Error(ArithmeticOverflow("SOR update"))
            }
        }
    }
  })
}

fn build_stationary_step(
  a: Matrix,
  b: Vector,
  old: Vector,
  new_values: List(Float),
  i: Int,
  use_new_values: Bool,
  transform: fn(Int, Vector, Float) -> Result(Float, NlaError),
) -> Result(Vector, NlaError) {
  case i >= matrix.rows(a) {
    True -> Ok(vector.from_list(new_values))
    False -> {
      let diagonal = matrix.unsafe_get(a, i, i)
      case
        singular_magnitude(float.absolute_value(diagonal), matrix.norm_inf(a))
      {
        True -> Error(SingularMatrix(i))
        False -> {
          let pairs =
            list.map(matrix.indices(matrix.cols(a)), fn(j) {
              case j == i {
                True -> #(0.0, 0.0)
                False -> {
                  let xj = case use_new_values && j < i {
                    True -> unsafe_at(new_values, j)
                    False -> unsafe_vector_get(old, j)
                  }
                  #(matrix.unsafe_get(a, i, j), xj)
                }
              }
            })
          case numerics.checked_dot_pairs(pairs) {
            Error(_) -> Error(ArithmeticOverflow("stationary row sum"))
            Ok(sum) ->
              case numerics.checked_subtract(unsafe_vector_get(b, i), sum) {
                Error(_) -> Error(ArithmeticOverflow("stationary residual"))
                Ok(numerator) ->
                  case numerics.checked_divide(numerator, diagonal) {
                    Error(_) -> Error(ArithmeticOverflow("stationary division"))
                    Ok(candidate) ->
                      case transform(i, old, candidate) {
                        Error(e) -> Error(e)
                        Ok(updated) ->
                          build_stationary_step(
                            a,
                            b,
                            old,
                            list.append(new_values, [updated]),
                            i + 1,
                            use_new_values,
                            transform,
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

fn steepest_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case error_analysis.residual(a, x, b) {
    Error(e) -> Error(e)
    Ok(r) ->
      case vector.norm2(r) {
        Error(e) -> Error(e)
        Ok(r_norm) if r_norm <=. tolerance ->
          Ok(IterationResult(
            solution: x,
            iterations: iteration,
            residual_norm: r_norm,
            converged: True,
          ))
        Ok(r_norm) ->
          case iteration >= max_iterations {
            True ->
              Ok(IterationResult(
                solution: x,
                iterations: iteration,
                residual_norm: r_norm,
                converged: False,
              ))
            False ->
              case matrix.mul_vec(a, r) {
                Error(e) -> Error(e)
                Ok(ar) ->
                  case vector.dot(r, r) {
                    Error(e) -> Error(e)
                    Ok(rr) ->
                      case vector.dot(r, ar) {
                        Error(e) -> Error(e)
                        Ok(rar) -> {
                          case dot_singular(rar, r, ar) {
                            Error(e) -> Error(e)
                            Ok(True) -> Error(SingularMatrix(iteration))
                            Ok(False) ->
                              case
                                checked_ratio(rr, rar, "steepest descent step")
                              {
                                Error(e) -> Error(e)
                                Ok(alpha) ->
                                  case vector.axpy(alpha, r, x) {
                                    Error(e) -> Error(e)
                                    Ok(next) ->
                                      steepest_loop(
                                        a,
                                        b,
                                        next,
                                        iteration + 1,
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
  }
}

fn conjugate_gradient_loop(
  a: Matrix,
  x: Vector,
  r: Vector,
  p: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case vector.norm2(r) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(IterationResult(
        solution: x,
        iterations: iteration,
        residual_norm: r_norm,
        converged: True,
      ))
    Ok(r_norm) ->
      case iteration >= max_iterations {
        True ->
          Ok(IterationResult(
            solution: x,
            iterations: iteration,
            residual_norm: r_norm,
            converged: False,
          ))
        False ->
          case cg_next(a, x, r, p) {
            Error(e) -> Error(e)
            Ok(#(next_x, next_r, next_p)) ->
              conjugate_gradient_loop(
                a,
                next_x,
                next_r,
                next_p,
                iteration + 1,
                max_iterations,
                tolerance,
              )
          }
      }
  }
}

fn practical_cg_loop(
  a: Matrix,
  b: Vector,
  x: Vector,
  r: Vector,
  p: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  recompute_every: Int,
) -> Result(IterationResult, NlaError) {
  case vector.norm2(r) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(IterationResult(
        solution: x,
        iterations: iteration,
        residual_norm: r_norm,
        converged: True,
      ))
    Ok(r_norm) ->
      case iteration >= max_iterations {
        True ->
          Ok(IterationResult(
            solution: x,
            iterations: iteration,
            residual_norm: r_norm,
            converged: False,
          ))
        False ->
          case cg_next(a, x, r, p) {
            Error(e) -> Error(e)
            Ok(#(next_x, next_r, next_p)) -> {
              let next_iteration = iteration + 1
              case next_iteration % recompute_every == 0 {
                True ->
                  case error_analysis.residual(a, next_x, b) {
                    Error(e) -> Error(e)
                    Ok(recomputed_r) ->
                      practical_cg_loop(
                        a,
                        b,
                        next_x,
                        recomputed_r,
                        recomputed_r,
                        next_iteration,
                        max_iterations,
                        tolerance,
                        recompute_every,
                      )
                  }
                False ->
                  practical_cg_loop(
                    a,
                    b,
                    next_x,
                    next_r,
                    next_p,
                    next_iteration,
                    max_iterations,
                    tolerance,
                    recompute_every,
                  )
              }
            }
          }
      }
  }
}

fn cg_next(
  a: Matrix,
  x: Vector,
  r: Vector,
  p: Vector,
) -> Result(#(Vector, Vector, Vector), NlaError) {
  case matrix.mul_vec(a, p) {
    Error(e) -> Error(e)
    Ok(ap) ->
      case vector.dot(r, r) {
        Error(e) -> Error(e)
        Ok(rr) ->
          case vector.dot(p, ap) {
            Error(e) -> Error(e)
            Ok(pap) -> {
              case dot_singular(pap, p, ap) {
                Error(e) -> Error(e)
                Ok(True) -> Error(SingularMatrix(0))
                Ok(False) ->
                  case checked_ratio(rr, pap, "conjugate gradient alpha") {
                    Error(e) -> Error(e)
                    Ok(alpha) ->
                      case vector.axpy(alpha, p, x) {
                        Error(e) -> Error(e)
                        Ok(next_x) ->
                          case vector.axpy(0.0 -. alpha, ap, r) {
                            Error(e) -> Error(e)
                            Ok(next_r) ->
                              case vector.dot(next_r, next_r) {
                                Error(e) -> Error(e)
                                Ok(next_rr) ->
                                  case
                                    checked_ratio(
                                      next_rr,
                                      rr,
                                      "conjugate gradient beta",
                                    )
                                  {
                                    Error(e) -> Error(e)
                                    Ok(beta) ->
                                      case vector.axpy(beta, p, next_r) {
                                        Error(e) -> Error(e)
                                        Ok(next_p) ->
                                          Ok(#(next_x, next_r, next_p))
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
}

fn pcg_loop(
  a: Matrix,
  x: Vector,
  r: Vector,
  z: Vector,
  p: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  preconditioner: fn(Vector) -> Result(Vector, NlaError),
) -> Result(IterationResult, NlaError) {
  case vector.norm2(r) {
    Error(e) -> Error(e)
    Ok(r_norm) if r_norm <=. tolerance ->
      Ok(IterationResult(
        solution: x,
        iterations: iteration,
        residual_norm: r_norm,
        converged: True,
      ))
    Ok(r_norm) ->
      case iteration >= max_iterations {
        True ->
          Ok(IterationResult(
            solution: x,
            iterations: iteration,
            residual_norm: r_norm,
            converged: False,
          ))
        False ->
          case pcg_next(a, x, r, z, p, preconditioner) {
            Error(e) -> Error(e)
            Ok(#(next_x, next_r, next_z, next_p)) ->
              pcg_loop(
                a,
                next_x,
                next_r,
                next_z,
                next_p,
                iteration + 1,
                max_iterations,
                tolerance,
                preconditioner,
              )
          }
      }
  }
}

fn pcg_next(
  a: Matrix,
  x: Vector,
  r: Vector,
  z: Vector,
  p: Vector,
  preconditioner: fn(Vector) -> Result(Vector, NlaError),
) -> Result(#(Vector, Vector, Vector, Vector), NlaError) {
  case matrix.mul_vec(a, p) {
    Error(e) -> Error(e)
    Ok(ap) ->
      case vector.dot(r, z) {
        Error(e) -> Error(e)
        Ok(rz) ->
          case vector.dot(p, ap) {
            Error(e) -> Error(e)
            Ok(pap) -> {
              case dot_singular(pap, p, ap) {
                Error(e) -> Error(e)
                Ok(True) -> Error(SingularMatrix(0))
                Ok(False) ->
                  case checked_ratio(rz, pap, "preconditioned CG alpha") {
                    Error(e) -> Error(e)
                    Ok(alpha) ->
                      case vector.axpy(alpha, p, x) {
                        Error(e) -> Error(e)
                        Ok(next_x) ->
                          case vector.axpy(0.0 -. alpha, ap, r) {
                            Error(e) -> Error(e)
                            Ok(next_r) ->
                              case preconditioner(next_r) {
                                Error(e) -> Error(e)
                                Ok(next_z) ->
                                  case vector.dot(next_r, next_z) {
                                    Error(e) -> Error(e)
                                    Ok(next_rz) ->
                                      case
                                        checked_ratio(
                                          next_rz,
                                          rz,
                                          "preconditioned CG beta",
                                        )
                                      {
                                        Error(e) -> Error(e)
                                        Ok(beta) ->
                                          case vector.axpy(beta, p, next_z) {
                                            Error(e) -> Error(e)
                                            Ok(next_p) ->
                                              Ok(#(
                                                next_x,
                                                next_r,
                                                next_z,
                                                next_p,
                                              ))
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
  }
}

fn jacobi_precondition(a: Matrix, r: Vector) -> Result(Vector, NlaError) {
  build_preconditioned(a, r, 0, [])
}

fn build_preconditioned(
  a: Matrix,
  r: Vector,
  i: Int,
  values: List(Float),
) -> Result(Vector, NlaError) {
  case i >= vector.dimension(r) {
    True -> Ok(vector.from_list(values))
    False -> {
      let diagonal = matrix.unsafe_get(a, i, i)
      case
        singular_magnitude(float.absolute_value(diagonal), matrix.norm_inf(a))
      {
        True -> Error(SingularMatrix(i))
        False ->
          case numerics.checked_divide(unsafe_vector_get(r, i), diagonal) {
            Error(_) -> Error(ArithmeticOverflow("Jacobi preconditioner"))
            Ok(value) ->
              build_preconditioned(a, r, i + 1, list.append(values, [value]))
          }
      }
    }
  }
}

fn validate_stationary_matrix(a: Matrix) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.is_finite(a) {
        True -> Ok(Nil)
        False -> Error(NonFiniteInput("stationary iteration matrix"))
      }
  }
}

fn validate_stationary_method(
  method: StationaryMethod,
) -> Result(Nil, NlaError) {
  case method {
    SorIteration(omega) -> validate_omega(omega)
    _ -> Ok(Nil)
  }
}

fn validate_omega(omega: Float) -> Result(Nil, NlaError) {
  case numerics.is_finite(omega) {
    False -> Error(NonFiniteInput("SOR omega"))
    True if omega >. 0.0 && omega <. 2.0 -> Ok(Nil)
    True -> Error(InvalidInput("SOR omega should be in (0, 2)"))
  }
}

fn validate_iterative_options(
  max_iterations: Int,
  tolerance: Float,
) -> Result(Nil, NlaError) {
  case max_iterations < 0 {
    True -> Error(InvalidInput("max_iterations must be non-negative"))
    False ->
      case numerics.is_finite(tolerance) {
        False -> Error(NonFiniteInput("iteration tolerance"))
        True if tolerance <. 0.0 ->
          Error(InvalidInput("tolerance must be non-negative"))
        True -> Ok(Nil)
      }
  }
}

fn prepare_system(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(ScaledSystem, NlaError) {
  case validate_square_system(a, b, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      case validate_iterative_options(max_iterations, tolerance) {
        Error(e) -> Error(e)
        Ok(_) ->
          case
            matrix.is_finite(a)
            && vector.is_finite(b)
            && vector.is_finite(initial)
          {
            False -> Error(NonFiniteInput("iterative system"))
            True -> {
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

fn checked_ratio(
  numerator: Float,
  denominator: Float,
  operation: String,
) -> Result(Float, NlaError) {
  case numerics.checked_divide(numerator, denominator) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
  }
}

fn finish_scaled(
  result: Result(IterationResult, NlaError),
  scale: Float,
) -> Result(IterationResult, NlaError) {
  case result {
    Error(e) -> Error(e)
    Ok(value) ->
      case vector.is_finite(value.solution) {
        False -> Error(ArithmeticOverflow("iterative solution"))
        True ->
          case numerics.checked_multiply(value.residual_norm, scale) {
            Error(_) -> Error(ArithmeticOverflow("residual norm rescaling"))
            Ok(residual_norm) ->
              Ok(IterationResult(
                solution: value.solution,
                iterations: value.iterations,
                residual_norm: residual_norm,
                converged: value.converged,
              ))
          }
      }
  }
}

fn validate_square_system(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case
        matrix.rows(a) == vector.dimension(b)
        && vector.dimension(b) == vector.dimension(x)
      {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(matrix.rows(a)),
            actual: int.to_string(vector.dimension(b))
              <> " and "
              <> int.to_string(vector.dimension(x)),
          ))
      }
  }
}

fn residual_norm(a: Matrix, x: Vector, b: Vector) -> Result(Float, NlaError) {
  error_analysis.residual_norm2(a, x, b)
}

fn singular_magnitude(magnitude: Float, scale: Float) -> Bool {
  numerics.relative_near_zero(magnitude, scale, diagonal_tolerance)
}

fn dot_singular(
  _value: Float,
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
            Error(_) -> Error(ArithmeticOverflow("normalized dot product"))
            Ok(cosine) ->
              Ok(float.absolute_value(cosine) <=. diagonal_tolerance)
          }
        }
      }
  }
}

fn unsafe_vector_get(values: Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
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
