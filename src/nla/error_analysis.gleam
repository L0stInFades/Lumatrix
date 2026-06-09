import gleam/list
import nla/direct
import nla/error.{type NlaError, DimensionMismatch, InvalidInput}
import nla/matrix.{type Matrix}
import nla/vector.{type Vector}

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
        Ok(b_norm) if b_norm >. 0.0 -> Ok(r_norm /. b_norm)
        Ok(_) -> Error(InvalidInput("relative residual needs non-zero b"))
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
      let denominator =
        matrix.norm_inf(a) *. vector.norm_inf(x) +. vector.norm_inf(b)
      case denominator >. 0.0 {
        True -> Ok(vector.norm_inf(r) /. denominator)
        False -> Error(InvalidInput("backward error denominator is zero"))
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
        True -> Ok(vector.norm_inf(delta) /. denominator)
        False ->
          Error(InvalidInput("forward error needs non-zero exact solution"))
      }
    }
  }
}

pub fn condition_number_inf(a: Matrix) -> Result(Float, NlaError) {
  case direct.inverse(a) {
    Error(e) -> Error(e)
    Ok(inv) -> Ok(matrix.norm_inf(a) *. matrix.norm_inf(inv))
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
      case normwise_relative_residual(a, x, b) {
        Error(e) -> Error(e)
        Ok(relative_residual) -> Ok(kappa *. relative_residual)
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

pub fn perturbation_bound(
  relative_matrix_error: Float,
  relative_rhs_error: Float,
  condition_number: Float,
) -> Result(Float, NlaError) {
  let denominator = 1.0 -. condition_number *. relative_matrix_error
  case denominator >. 0.0 {
    True ->
      Ok(
        condition_number
        *. { relative_matrix_error +. relative_rhs_error }
        /. denominator,
      )
    False ->
      Error(DimensionMismatch(
        expected: "kappa * relative_matrix_error < 1",
        actual: "ill-conditioned perturbation bound",
      ))
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
