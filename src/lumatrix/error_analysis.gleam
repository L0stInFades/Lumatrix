import gleam/list
import lumatrix/direct
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, InvalidInput, NonFiniteInput,
}
import lumatrix/matrix.{type Matrix}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

pub type IterativeRefinementResult {
  IterativeRefinementResult(
    solution: Vector,
    iterations: Int,
    residual_norm: Float,
    converged: Bool,
    residual_history: List(Float),
  )
}

pub fn residual(a: Matrix, x: Vector, b: Vector) -> Result(Vector, NlaError) {
  case matrix.mul_vec(a, x) {
    Error(e) -> Error(e)
    Ok(ax) -> vector.sub(b, ax)
  }
}

pub fn residual_norm2(
  a: Matrix,
  x: Vector,
  b: Vector,
) -> Result(Float, NlaError) {
  case residual(a, x, b) {
    Error(e) -> Error(e)
    Ok(r) -> vector.norm2(r)
  }
}

pub fn normwise_relative_residual(
  a: Matrix,
  x: Vector,
  b: Vector,
) -> Result(Float, NlaError) {
  case residual_norm2(a, x, b) {
    Error(e) -> Error(e)
    Ok(r_norm) -> {
      case vector.norm2(b) {
        Error(e) -> Error(e)
        Ok(b_norm) if b_norm >. 0.0 ->
          checked_division(r_norm, b_norm, "relative residual")
        Ok(_) -> Error(InvalidInput("relative residual needs non-zero b"))
      }
    }
  }
}

pub fn normwise_relative_residual_inf(
  a: Matrix,
  x: Vector,
  b: Vector,
) -> Result(Float, NlaError) {
  case residual(a, x, b) {
    Error(e) -> Error(e)
    Ok(r) -> {
      let b_norm = vector.norm_inf(b)
      case b_norm >. 0.0 {
        True ->
          checked_division(vector.norm_inf(r), b_norm, "relative residual")
        False -> Error(InvalidInput("relative residual needs non-zero b"))
      }
    }
  }
}

pub fn backward_error_inf(
  a: Matrix,
  x: Vector,
  b: Vector,
) -> Result(Float, NlaError) {
  case residual(a, x, b) {
    Error(e) -> Error(e)
    Ok(r) -> {
      case numerics.checked_multiply(matrix.norm_inf(a), vector.norm_inf(x)) {
        Error(_) -> Error(ArithmeticOverflow("backward error denominator"))
        Ok(matrix_term) ->
          case numerics.checked_add(matrix_term, vector.norm_inf(b)) {
            Error(_) -> Error(ArithmeticOverflow("backward error denominator"))
            Ok(denominator) if denominator >. 0.0 ->
              checked_division(
                vector.norm_inf(r),
                denominator,
                "backward error",
              )
            Ok(_) -> Error(InvalidInput("backward error denominator is zero"))
          }
      }
    }
  }
}

pub fn forward_error_inf(
  exact: Vector,
  computed: Vector,
) -> Result(Float, NlaError) {
  case vector.sub(exact, computed) {
    Error(e) -> Error(e)
    Ok(delta) -> {
      let denominator = vector.norm_inf(exact)
      case denominator >. 0.0 {
        True ->
          checked_division(vector.norm_inf(delta), denominator, "forward error")
        False ->
          Error(InvalidInput("forward error needs non-zero exact solution"))
      }
    }
  }
}

pub fn condition_number_inf(a: Matrix) -> Result(Float, NlaError) {
  case direct.inverse(a) {
    Error(e) -> Error(e)
    Ok(inv) ->
      case numerics.checked_multiply(matrix.norm_inf(a), matrix.norm_inf(inv)) {
        Ok(value) -> Ok(value)
        Error(_) -> Error(ArithmeticOverflow("condition number"))
      }
  }
}

pub fn residual_forward_bound_inf(
  a: Matrix,
  x: Vector,
  b: Vector,
) -> Result(Float, NlaError) {
  case condition_number_inf(a) {
    Error(e) -> Error(e)
    Ok(kappa) ->
      case normwise_relative_residual_inf(a, x, b) {
        Error(e) -> Error(e)
        Ok(relative_residual) ->
          case numerics.checked_multiply(kappa, relative_residual) {
            Ok(value) -> Ok(value)
            Error(_) -> Error(ArithmeticOverflow("residual forward bound"))
          }
      }
  }
}

pub fn iterative_refinement(
  a: Matrix,
  b: Vector,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(IterativeRefinementResult, NlaError) {
  case max_iterations < 0 {
    True -> Error(InvalidInput("max_iterations must be non-negative"))
    False ->
      case numerics.is_finite(tolerance) {
        False -> Error(NonFiniteInput("iterative refinement tolerance"))
        True if tolerance <. 0.0 ->
          Error(InvalidInput("tolerance must be non-negative"))
        True ->
          case direct.lu_factor(a) {
            Error(e) -> Error(e)
            Ok(factors) ->
              refinement_loop(
                a,
                factors,
                b,
                initial,
                0,
                max_iterations,
                tolerance,
                [],
              )
          }
      }
  }
}

pub fn perturbation_bound(
  relative_matrix_error: Float,
  relative_rhs_error: Float,
  condition_number: Float,
) -> Result(Float, NlaError) {
  case
    numerics.is_finite(relative_matrix_error)
    && numerics.is_finite(relative_rhs_error)
    && numerics.is_finite(condition_number)
  {
    False -> Error(NonFiniteInput("perturbation bound inputs"))
    True
      if relative_matrix_error <. 0.0
      || relative_rhs_error <. 0.0
      || condition_number <. 0.0
    -> Error(InvalidInput("perturbation bound inputs must be non-negative"))
    True ->
      case numerics.checked_multiply(condition_number, relative_matrix_error) {
        Error(_) -> Error(ArithmeticOverflow("perturbation denominator"))
        Ok(matrix_term) ->
          case numerics.checked_subtract(1.0, matrix_term) {
            Error(_) -> Error(ArithmeticOverflow("perturbation denominator"))
            Ok(denominator) if denominator <=. 0.0 ->
              Error(InvalidInput("kappa * relative_matrix_error must be < 1"))
            Ok(denominator) ->
              case
                numerics.checked_add(relative_matrix_error, relative_rhs_error)
              {
                Error(_) -> Error(ArithmeticOverflow("perturbation numerator"))
                Ok(total_error) ->
                  case
                    numerics.checked_multiply(condition_number, total_error)
                  {
                    Error(_) ->
                      Error(ArithmeticOverflow("perturbation numerator"))
                    Ok(numerator) ->
                      checked_division(
                        numerator,
                        denominator,
                        "perturbation bound",
                      )
                  }
              }
          }
      }
  }
}

fn checked_division(
  numerator: Float,
  denominator: Float,
  operation: String,
) -> Result(Float, NlaError) {
  case numerics.checked_divide(numerator, denominator) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(ArithmeticOverflow(operation))
  }
}

fn refinement_loop(
  a: Matrix,
  factors: direct.LU,
  b: Vector,
  x: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  history: List(Float),
) -> Result(IterativeRefinementResult, NlaError) {
  case residual(a, x, b) {
    Error(e) -> Error(e)
    Ok(r) ->
      case vector.norm2(r) {
        Error(e) -> Error(e)
        Ok(r_norm) -> {
          let next_history = list.append(history, [r_norm])
          case r_norm <=. tolerance {
            True ->
              Ok(IterativeRefinementResult(
                solution: x,
                iterations: iteration,
                residual_norm: r_norm,
                converged: True,
                residual_history: next_history,
              ))
            False ->
              case iteration >= max_iterations {
                True ->
                  Ok(IterativeRefinementResult(
                    solution: x,
                    iterations: iteration,
                    residual_norm: r_norm,
                    converged: False,
                    residual_history: next_history,
                  ))
                False ->
                  case direct.lu_solve(factors, r) {
                    Error(e) -> Error(e)
                    Ok(correction) ->
                      case vector.add(x, correction) {
                        Error(e) -> Error(e)
                        Ok(next_x) ->
                          refinement_loop(
                            a,
                            factors,
                            b,
                            next_x,
                            iteration + 1,
                            max_iterations,
                            tolerance,
                            next_history,
                          )
                      }
                  }
              }
          }
        }
      }
  }
}
