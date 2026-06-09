import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, NotSquare, SingularMatrix,
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
  case validate_square_system(a, b, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      stationary_loop(a, b, initial, max_iterations, tolerance, jacobi_step)
  }
}

pub fn gauss_seidel(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case validate_square_system(a, b, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      stationary_loop(
        a,
        b,
        initial,
        max_iterations,
        tolerance,
        gauss_seidel_step,
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
  case omega >. 0.0 && omega <. 2.0 {
    False -> Error(InvalidInput("SOR omega should be in (0, 2)"))
    True ->
      case validate_square_system(a, b, initial) {
        Error(e) -> Error(e)
        Ok(_) ->
          stationary_loop(a, b, initial, max_iterations, tolerance, fn(a, b, x) {
            sor_step(a, b, x, omega)
          })
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
  case validate_square_system(a, b, initial) {
    Error(e) -> Error(e)
    Ok(_) -> steepest_loop(a, b, initial, 0, max_iterations, tolerance)
  }
}

pub fn conjugate_gradient(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterationResult, NlaError) {
  case validate_square_system(a, b, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      case error_analysis.residual(a, initial, b) {
        Error(e) -> Error(e)
        Ok(r) ->
          conjugate_gradient_loop(
            a,
            initial,
            r,
            r,
            0,
            max_iterations,
            tolerance,
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
      case validate_square_system(a, b, initial) {
        Error(e) -> Error(e)
        Ok(_) ->
          case error_analysis.residual(a, initial, b) {
            Error(e) -> Error(e)
            Ok(r) ->
              practical_cg_loop(
                a,
                b,
                initial,
                r,
                r,
                0,
                max_iterations,
                tolerance,
                recompute_every,
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
  preconditioned_conjugate_gradient_with(
    a,
    b,
    initial,
    max_iterations,
    tolerance,
    fn(r) { jacobi_precondition(a, r) },
  )
}

pub fn preconditioned_conjugate_gradient_with(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
  preconditioner: fn(Vector) -> Result(Vector, NlaError),
) -> Result(IterationResult, NlaError) {
  case validate_square_system(a, b, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      case error_analysis.residual(a, initial, b) {
        Error(e) -> Error(e)
        Ok(r) ->
          case preconditioner(r) {
            Error(e) -> Error(e)
            Ok(z) ->
              pcg_loop(
                a,
                initial,
                r,
                z,
                z,
                0,
                max_iterations,
                tolerance,
                preconditioner,
              )
          }
      }
  }
}

fn stationary_iteration_matrix(
  a: Matrix,
  method: StationaryMethod,
) -> Result(Matrix, NlaError) {
  let assert Ok(zero_b) = vector.zeros(matrix.rows(a))
  build_stationary_columns(a, zero_b, method, 0, [])
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
  matrix.from_fn(rows: rows, cols: cols, with: fn(i, j) {
    unsafe_vector_get(unsafe_vector_at(columns, j), i)
  })
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
    candidate
  })
}

fn gauss_seidel_step(
  a: Matrix,
  b: Vector,
  x: Vector,
) -> Result(Vector, NlaError) {
  build_stationary_step(a, b, x, [], 0, True, fn(_, _, candidate) { candidate })
}

fn sor_step(
  a: Matrix,
  b: Vector,
  x: Vector,
  omega: Float,
) -> Result(Vector, NlaError) {
  build_stationary_step(a, b, x, [], 0, True, fn(i, old, candidate) {
    { 1.0 -. omega } *. unsafe_vector_get(old, i) +. omega *. candidate
  })
}

fn build_stationary_step(
  a: Matrix,
  b: Vector,
  old: Vector,
  new_values: List(Float),
  i: Int,
  use_new_values: Bool,
  transform: fn(Int, Vector, Float) -> Float,
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
          let sum =
            numerics.compensated_sum_map(matrix.indices(matrix.cols(a)), fn(j) {
              case j == i {
                True -> 0.0
                False -> {
                  let xj = case use_new_values && j < i {
                    True -> unsafe_at(new_values, j)
                    False -> unsafe_vector_get(old, j)
                  }
                  matrix.unsafe_get(a, i, j) *. xj
                }
              }
            })
          let candidate = { unsafe_vector_get(b, i) -. sum } /. diagonal
          let updated = transform(i, old, candidate)
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
                              case vector.axpy(rr /. rar, r, x) {
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
                Ok(False) -> {
                  let alpha = rr /. pap
                  case vector.axpy(alpha, p, x) {
                    Error(e) -> Error(e)
                    Ok(next_x) ->
                      case vector.axpy(0.0 -. alpha, ap, r) {
                        Error(e) -> Error(e)
                        Ok(next_r) ->
                          case vector.dot(next_r, next_r) {
                            Error(e) -> Error(e)
                            Ok(next_rr) -> {
                              let beta = next_rr /. rr
                              case vector.axpy(beta, p, next_r) {
                                Error(e) -> Error(e)
                                Ok(next_p) -> Ok(#(next_x, next_r, next_p))
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
                Ok(False) -> {
                  let alpha = rz /. pap
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
                                Ok(next_rz) -> {
                                  let beta = next_rz /. rz
                                  case vector.axpy(beta, p, next_z) {
                                    Error(e) -> Error(e)
                                    Ok(next_p) ->
                                      Ok(#(next_x, next_r, next_z, next_p))
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
          build_preconditioned(
            a,
            r,
            i + 1,
            list.append(values, [unsafe_vector_get(r, i) /. diagonal]),
          )
      }
    }
  }
}

fn validate_stationary_matrix(a: Matrix) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> Ok(Nil)
  }
}

fn validate_stationary_method(
  method: StationaryMethod,
) -> Result(Nil, NlaError) {
  case method {
    SorIteration(omega) ->
      case omega >. 0.0 && omega <. 2.0 {
        True -> Ok(Nil)
        False -> Error(InvalidInput("SOR omega should be in (0, 2)"))
      }
    _ -> Ok(Nil)
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

fn residual_norm(a: Matrix, x: Vector, b: Vector) -> Result(Float, NlaError) {
  error_analysis.residual_norm2(a, x, b)
}

fn singular_magnitude(magnitude: Float, scale: Float) -> Bool {
  numerics.relative_near_zero(magnitude, scale, diagonal_tolerance)
}

fn dot_singular(
  value: Float,
  left: Vector,
  right: Vector,
) -> Result(Bool, NlaError) {
  case vector.norm2(left) {
    Error(e) -> Error(e)
    Ok(left_norm) ->
      case vector.norm2(right) {
        Error(e) -> Error(e)
        Ok(right_norm) ->
          Ok(numerics.relative_near_zero(
            value,
            left_norm *. right_norm,
            diagonal_tolerance,
          ))
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
