import gleam/float
import gleam/int
import gleam/list
import lumatrix/complex
import lumatrix/direct
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, NoConvergence, NotSquare,
  ZeroNorm,
}
import lumatrix/matrix.{type Matrix}
import lumatrix/orthogonal
import lumatrix/vector.{type Vector}

const small = 1.0e-12

pub type Eigenpair {
  Eigenpair(
    value: Float,
    vector: Vector,
    residual_norm: Float,
    iterations: Int,
    converged: Bool,
  )
}

pub type ComplexEigenpair {
  ComplexEigenpair(
    value: complex.Complex,
    vector: complex.ComplexVector,
    residual_norm: Float,
    iterations: Int,
    converged: Bool,
  )
}

pub type Eigenvalue {
  RealEigenvalue(value: Float)
  ComplexEigenvalue(real: Float, imaginary: Float)
}

pub type SchurResult {
  SchurResult(q: Matrix, t: Matrix, iterations: Int, converged: Bool)
}

pub type HessenbergResult {
  HessenbergResult(q: Matrix, h: Matrix, iterations: Int)
}

pub type TridiagonalResult {
  TridiagonalResult(q: Matrix, t: Matrix, iterations: Int)
}

pub type QrConvergenceStep {
  QrConvergenceStep(iteration: Int, shift: Float, off_diagonal_norm: Float)
}

pub type QrConvergenceHistory {
  QrConvergenceHistory(steps: List(QrConvergenceStep), result: SchurResult)
}

pub type SchurBlock {
  RealBlock(index: Int, value: Float)
  ComplexConjugateBlock(
    start: Int,
    real: Float,
    imaginary: Float,
    trace: Float,
    determinant: Float,
  )
}

pub type SymmetricEigenResult {
  SymmetricEigenResult(
    eigenvectors: Matrix,
    diagonal: Vector,
    iterations: Int,
    converged: Bool,
  )
}

pub fn power_method(
  a: Matrix,
  initial: Vector,
  max_iterations: Int,
  tolerance: Float,
) -> Result(Eigenpair, NlaError) {
  case validate_square_vector(a, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      case vector.normalize(initial) {
        Error(e) -> Error(e)
        Ok(x0) -> power_loop(a, x0, 0, max_iterations, tolerance)
      }
  }
}

pub fn inverse_power_method(
  a: Matrix,
  initial: Vector,
  shift: Float,
  max_iterations: Int,
  tolerance: Float,
) -> Result(Eigenpair, NlaError) {
  case validate_square_vector(a, initial) {
    Error(e) -> Error(e)
    Ok(_) ->
      case vector.normalize(initial) {
        Error(e) -> Error(e)
        Ok(x0) ->
          inverse_power_loop(
            a,
            shifted(a, shift),
            x0,
            0,
            max_iterations,
            tolerance,
          )
      }
  }
}

pub fn qr_iteration(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_loop(a, q0, 0, max_iterations, tolerance, fn(_t) { 0.0 })
    }
  }
}

pub fn shifted_qr_iteration(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_loop(a, q0, 0, max_iterations, tolerance, trailing_rayleigh_shift)
    }
  }
}

pub fn qr_convergence_history(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(QrConvergenceHistory, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_history_loop(a, q0, 0, max_iterations, tolerance, fn(_t) { 0.0 }, [])
    }
  }
}

pub fn shifted_qr_convergence_history(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(QrConvergenceHistory, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_history_loop(
        a,
        q0,
        0,
        max_iterations,
        tolerance,
        trailing_rayleigh_shift,
        [],
      )
    }
  }
}

pub fn symmetric_qr_convergence_history(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(QrConvergenceHistory, NlaError) {
  case validate_symmetric(a, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_history_loop(a, q0, 0, max_iterations, tolerance, wilkinson_shift, [])
    }
  }
}

pub fn implicit_qr_iteration(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case hessenberg_reduction(a) {
    Error(e) -> Error(e)
    Ok(reduced) ->
      implicit_qr_loop(
        reduced.h,
        reduced.q,
        0,
        max_iterations,
        tolerance,
        wilkinson_shift,
      )
  }
}

pub fn double_shift_qr_iteration(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case hessenberg_reduction(a) {
    Error(e) -> Error(e)
    Ok(reduced) ->
      double_shift_loop(reduced.h, reduced.q, 0, max_iterations, tolerance)
  }
}

pub fn symmetric_qr(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case validate_symmetric(a, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      qr_loop(a, q0, 0, max_iterations, tolerance, wilkinson_shift)
    }
  }
}

pub fn symmetric_qr_eigen(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SymmetricEigenResult, NlaError) {
  case validate_symmetric(a, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) ->
      case symmetric_tridiagonal_reduction(a) {
        Error(e) -> Error(e)
        Ok(reduced) ->
          case
            qr_loop(
              reduced.t,
              reduced.q,
              0,
              max_iterations,
              tolerance,
              wilkinson_shift,
            )
          {
            Error(e) -> Error(e)
            Ok(schur) ->
              Ok(SymmetricEigenResult(
                eigenvectors: schur.q,
                diagonal: diagonal_vector(schur.t),
                iterations: schur.iterations,
                converged: schur.converged,
              ))
          }
      }
  }
}

pub fn wilkinson_shift(a: Matrix) -> Float {
  case matrix.rows(a) < 2 || matrix.cols(a) < 2 {
    True -> trailing_rayleigh_shift(a)
    False -> {
      let n = matrix.rows(a)
      let a00 = matrix.unsafe_get(a, n - 2, n - 2)
      let a01 = matrix.unsafe_get(a, n - 2, n - 1)
      let a11 = matrix.unsafe_get(a, n - 1, n - 1)
      let delta = { a00 -. a11 } /. 2.0
      let scale = float.absolute_value(delta)
      case float.square_root(delta *. delta +. a01 *. a01) {
        Error(_) -> a11
        Ok(denominator_root) -> {
          let denominator = scale +. denominator_root
          case denominator <=. small {
            True -> a11
            False -> a11 -. sign(delta) *. a01 *. a01 /. denominator
          }
        }
      }
    }
  }
}

pub fn hessenberg_reduction(a: Matrix) -> Result(HessenbergResult, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True -> {
      let assert Ok(q0) = matrix.identity(matrix.rows(a))
      hessenberg_loop(a, q0, 0)
    }
  }
}

pub fn symmetric_tridiagonal_reduction(
  a: Matrix,
) -> Result(TridiagonalResult, NlaError) {
  case validate_symmetric(a, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) ->
      case hessenberg_reduction(a) {
        Error(e) -> Error(e)
        Ok(reduced) ->
          Ok(TridiagonalResult(
            q: reduced.q,
            t: reduced.h,
            iterations: reduced.iterations,
          ))
      }
  }
}

pub fn jacobi_eigen(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SymmetricEigenResult, NlaError) {
  case validate_symmetric(a, 1.0e-10) {
    Error(e) -> Error(e)
    Ok(_) -> {
      let assert Ok(v0) = matrix.identity(matrix.rows(a))
      jacobi_loop(a, v0, 0, max_iterations, tolerance)
    }
  }
}

pub fn real_schur_basic(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case hessenberg_reduction(a) {
    Error(e) -> Error(e)
    Ok(reduced) ->
      real_schur_loop(reduced.h, reduced.q, 0, max_iterations, tolerance)
  }
}

pub fn real_schur_blocks(
  t: Matrix,
  tolerance: Float,
) -> Result(List(SchurBlock), NlaError) {
  case matrix.is_square(t) {
    False -> Error(NotSquare(matrix.rows(t), matrix.cols(t)))
    True ->
      case quasi_lower_off_diagonal_norm(t, tolerance) <=. tolerance {
        True -> Ok(scan_schur_blocks(t, 0, tolerance, []))
        False -> Error(InvalidInput("matrix is not in real Schur form"))
      }
  }
}

pub fn real_schur_eigenvalues(
  t: Matrix,
  tolerance: Float,
) -> Result(List(Eigenvalue), NlaError) {
  case real_schur_blocks(t, tolerance) {
    Error(e) -> Error(e)
    Ok(blocks) -> Ok(list.flat_map(blocks, schur_block_eigenvalues))
  }
}

pub fn real_schur_eigenvalues_of(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(List(Eigenvalue), NlaError) {
  case real_schur_basic(a, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(schur) ->
      case schur.converged {
        True -> real_schur_eigenvalues(schur.t, tolerance)
        False -> Error(InvalidInput("real Schur iteration did not converge"))
      }
  }
}

pub fn generalized_standard_matrix(
  a: Matrix,
  b: Matrix,
) -> Result(Matrix, NlaError) {
  case validate_square_pair(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case direct.inverse_complete_pivoting(b) {
        Error(e) -> Error(e)
        Ok(b_inverse) -> matrix.mul(b_inverse, a)
      }
  }
}

pub fn generalized_real_schur(
  a: Matrix,
  b: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  case generalized_standard_matrix(a, b) {
    Error(e) -> Error(e)
    Ok(standard) -> real_schur_basic(standard, max_iterations, tolerance)
  }
}

pub fn generalized_eigenvalues(
  a: Matrix,
  b: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(List(Eigenvalue), NlaError) {
  case generalized_real_schur(a, b, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(schur) ->
      case schur.converged {
        False ->
          Error(NoConvergence(
            iterations: schur.iterations,
            residual: quasi_lower_off_diagonal_norm(schur.t, tolerance),
          ))
        True -> real_schur_eigenvalues(schur.t, tolerance)
      }
  }
}

pub fn generalized_complex_eigenpairs(
  a: Matrix,
  b: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(List(ComplexEigenpair), NlaError) {
  case generalized_standard_matrix(a, b) {
    Error(e) -> Error(e)
    Ok(standard) ->
      case complex_eigenpairs_of(standard, max_iterations, tolerance) {
        Error(e) -> Error(e)
        Ok(pairs) ->
          list.try_map(over: pairs, with: fn(pair) {
            case
              generalized_complex_eigen_residual(a, b, pair.vector, pair.value)
            {
              Error(e) -> Error(e)
              Ok(residual_norm) ->
                Ok(ComplexEigenpair(
                  value: pair.value,
                  vector: pair.vector,
                  residual_norm: residual_norm,
                  iterations: pair.iterations,
                  converged: pair.converged && residual_norm <=. tolerance,
                ))
            }
          })
      }
  }
}

pub fn real_schur_complex_eigenpairs(
  q: Matrix,
  t: Matrix,
  tolerance: Float,
) -> Result(List(ComplexEigenpair), NlaError) {
  case validate_schur_eigenpair_inputs(q, t) {
    Error(e) -> Error(e)
    Ok(_) ->
      case real_schur_blocks(t, tolerance) {
        Error(e) -> Error(e)
        Ok(blocks) -> schur_blocks_complex_eigenpairs(q, t, blocks, blocks, 0)
      }
  }
}

pub fn complex_eigenpairs_of(
  a: Matrix,
  max_iterations: Int,
  tolerance: Float,
) -> Result(List(ComplexEigenpair), NlaError) {
  case real_schur_basic(a, max_iterations, tolerance) {
    Error(e) -> Error(e)
    Ok(schur) ->
      case schur.converged {
        False ->
          Error(NoConvergence(
            iterations: schur.iterations,
            residual: quasi_lower_off_diagonal_norm(schur.t, tolerance),
          ))
        True ->
          case real_schur_complex_eigenpairs(schur.q, schur.t, tolerance) {
            Error(e) -> Error(e)
            Ok(pairs) ->
              list.try_map(over: pairs, with: fn(pair) {
                case complex_eigen_residual(a, pair.vector, pair.value) {
                  Error(e) -> Error(e)
                  Ok(residual_norm) ->
                    Ok(ComplexEigenpair(
                      value: pair.value,
                      vector: pair.vector,
                      residual_norm: residual_norm,
                      iterations: schur.iterations,
                      converged: True,
                    ))
                }
              })
          }
      }
  }
}

fn jacobi_loop(
  a: Matrix,
  eigenvectors: Matrix,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SymmetricEigenResult, NlaError) {
  let #(p, q, magnitude) = largest_off_diagonal(a)
  case magnitude <=. tolerance {
    True ->
      Ok(SymmetricEigenResult(
        eigenvectors: eigenvectors,
        diagonal: diagonal_vector(a),
        iterations: iteration,
        converged: True,
      ))
    False ->
      case iteration >= max_iterations {
        True ->
          Ok(SymmetricEigenResult(
            eigenvectors: eigenvectors,
            diagonal: diagonal_vector(a),
            iterations: iteration,
            converged: False,
          ))
        False -> {
          let apq = matrix.unsafe_get(a, p, q)
          let app = matrix.unsafe_get(a, p, p)
          let aqq = matrix.unsafe_get(a, q, q)
          let tau = { aqq -. app } /. { 2.0 *. apq }
          let t = jacobi_t(tau)
          let c = reciprocal_square_root(1.0 +. t *. t)
          let s = t *. c
          let next_a = apply_jacobi_similarity(a, p, q, c, s, t)
          let next_v = apply_jacobi_to_eigenvectors(eigenvectors, p, q, c, s)
          jacobi_loop(next_a, next_v, iteration + 1, max_iterations, tolerance)
        }
      }
  }
}

fn power_loop(
  a: Matrix,
  x: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(Eigenpair, NlaError) {
  case eigen_residual(a, x) {
    Error(e) -> Error(e)
    Ok(#(lambda, residual)) if residual <=. tolerance ->
      Ok(Eigenpair(
        value: lambda,
        vector: x,
        residual_norm: residual,
        iterations: iteration,
        converged: True,
      ))
    Ok(#(lambda, residual)) ->
      case iteration >= max_iterations {
        True ->
          Ok(Eigenpair(
            value: lambda,
            vector: x,
            residual_norm: residual,
            iterations: iteration,
            converged: False,
          ))
        False ->
          case matrix.mul_vec(a, x) {
            Error(e) -> Error(e)
            Ok(y) ->
              case vector.normalize(y) {
                Error(ZeroNorm) -> Error(ZeroNorm)
                Error(e) -> Error(e)
                Ok(next) ->
                  power_loop(a, next, iteration + 1, max_iterations, tolerance)
              }
          }
      }
  }
}

fn inverse_power_loop(
  a: Matrix,
  shifted_a: Matrix,
  x: Vector,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(Eigenpair, NlaError) {
  case eigen_residual(a, x) {
    Error(e) -> Error(e)
    Ok(#(lambda, residual)) if residual <=. tolerance ->
      Ok(Eigenpair(
        value: lambda,
        vector: x,
        residual_norm: residual,
        iterations: iteration,
        converged: True,
      ))
    Ok(#(lambda, residual)) ->
      case iteration >= max_iterations {
        True ->
          Ok(Eigenpair(
            value: lambda,
            vector: x,
            residual_norm: residual,
            iterations: iteration,
            converged: False,
          ))
        False ->
          case direct.solve(shifted_a, x) {
            Error(e) -> Error(e)
            Ok(y) ->
              case vector.normalize(y) {
                Error(e) -> Error(e)
                Ok(next) ->
                  inverse_power_loop(
                    a,
                    shifted_a,
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

fn qr_loop(
  t: Matrix,
  q_acc: Matrix,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  shift: fn(Matrix) -> Float,
) -> Result(SchurResult, NlaError) {
  let off = lower_off_diagonal_norm(t)
  case off <=. tolerance {
    True ->
      Ok(SchurResult(q: q_acc, t: t, iterations: iteration, converged: True))
    False ->
      case iteration >= max_iterations {
        True ->
          Ok(SchurResult(
            q: q_acc,
            t: t,
            iterations: iteration,
            converged: False,
          ))
        False -> {
          let mu = shift(t)
          case qr_step(t, mu) {
            Error(e) -> Error(e)
            Ok(#(next_t, step_q)) ->
              case matrix.mul(q_acc, step_q) {
                Error(e) -> Error(e)
                Ok(next_q) ->
                  qr_loop(
                    next_t,
                    next_q,
                    iteration + 1,
                    max_iterations,
                    tolerance,
                    shift,
                  )
              }
          }
        }
      }
  }
}

fn qr_history_loop(
  t: Matrix,
  q_acc: Matrix,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  shift: fn(Matrix) -> Float,
  steps: List(QrConvergenceStep),
) -> Result(QrConvergenceHistory, NlaError) {
  let off = lower_off_diagonal_norm(t)
  let mu = shift(t)
  let next_steps = [
    QrConvergenceStep(iteration: iteration, shift: mu, off_diagonal_norm: off),
    ..steps
  ]
  case off <=. tolerance {
    True ->
      Ok(QrConvergenceHistory(
        steps: list.reverse(next_steps),
        result: SchurResult(
          q: q_acc,
          t: t,
          iterations: iteration,
          converged: True,
        ),
      ))
    False ->
      case iteration >= max_iterations {
        True ->
          Ok(QrConvergenceHistory(
            steps: list.reverse(next_steps),
            result: SchurResult(
              q: q_acc,
              t: t,
              iterations: iteration,
              converged: False,
            ),
          ))
        False ->
          case qr_step(t, mu) {
            Error(e) -> Error(e)
            Ok(#(next_t, step_q)) ->
              case matrix.mul(q_acc, step_q) {
                Error(e) -> Error(e)
                Ok(next_q) ->
                  qr_history_loop(
                    next_t,
                    next_q,
                    iteration + 1,
                    max_iterations,
                    tolerance,
                    shift,
                    next_steps,
                  )
              }
          }
      }
  }
}

fn qr_step(t: Matrix, mu: Float) -> Result(#(Matrix, Matrix), NlaError) {
  let shifted_t = shifted(t, mu)
  case orthogonal.householder_qr(shifted_t) {
    Error(e) -> Error(e)
    Ok(qr) ->
      case matrix.mul(qr.r, qr.q) {
        Error(e) -> Error(e)
        Ok(rq) ->
          case matrix.add(rq, diagonal_shift(matrix.rows(t), mu)) {
            Error(e) -> Error(e)
            Ok(next_t) -> Ok(#(next_t, qr.q))
          }
      }
  }
}

fn implicit_qr_loop(
  t: Matrix,
  q_acc: Matrix,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
  shift: fn(Matrix) -> Float,
) -> Result(SchurResult, NlaError) {
  let off = lower_off_diagonal_norm(t)
  case off <=. tolerance {
    True ->
      Ok(SchurResult(q: q_acc, t: t, iterations: iteration, converged: True))
    False ->
      case iteration >= max_iterations {
        True ->
          Ok(SchurResult(
            q: q_acc,
            t: t,
            iterations: iteration,
            converged: False,
          ))
        False -> {
          let mu = shift(t)
          case single_shift_givens_step(t, mu) {
            Error(e) -> Error(e)
            Ok(#(next_t, step_q)) ->
              case matrix.mul(q_acc, step_q) {
                Error(e) -> Error(e)
                Ok(next_q) ->
                  implicit_qr_loop(
                    next_t,
                    next_q,
                    iteration + 1,
                    max_iterations,
                    tolerance,
                    shift,
                  )
              }
          }
        }
      }
  }
}

fn double_shift_loop(
  t: Matrix,
  q_acc: Matrix,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  let off = lower_off_diagonal_norm(t)
  case off <=. tolerance {
    True ->
      Ok(SchurResult(q: q_acc, t: t, iterations: iteration, converged: True))
    False ->
      case iteration >= max_iterations {
        True ->
          Ok(SchurResult(
            q: q_acc,
            t: t,
            iterations: iteration,
            converged: False,
          ))
        False ->
          case double_shift_step(t) {
            Error(e) -> Error(e)
            Ok(#(next_t, step_q)) ->
              case matrix.mul(q_acc, step_q) {
                Error(e) -> Error(e)
                Ok(next_q) ->
                  double_shift_loop(
                    next_t,
                    next_q,
                    iteration + 1,
                    max_iterations,
                    tolerance,
                  )
              }
          }
      }
  }
}

fn real_schur_loop(
  t: Matrix,
  q_acc: Matrix,
  iteration: Int,
  max_iterations: Int,
  tolerance: Float,
) -> Result(SchurResult, NlaError) {
  let off = quasi_lower_off_diagonal_norm(t, tolerance)
  case off <=. tolerance {
    True ->
      Ok(SchurResult(q: q_acc, t: t, iterations: iteration, converged: True))
    False ->
      case iteration >= max_iterations {
        True ->
          Ok(SchurResult(
            q: q_acc,
            t: t,
            iterations: iteration,
            converged: False,
          ))
        False ->
          case double_shift_step(t) {
            Error(e) -> Error(e)
            Ok(#(next_t, step_q)) ->
              case matrix.mul(q_acc, step_q) {
                Error(e) -> Error(e)
                Ok(next_q) ->
                  real_schur_loop(
                    next_t,
                    next_q,
                    iteration + 1,
                    max_iterations,
                    tolerance,
                  )
              }
          }
      }
  }
}

fn hessenberg_loop(
  t: Matrix,
  q_acc: Matrix,
  k: Int,
) -> Result(HessenbergResult, NlaError) {
  case k >= matrix.rows(t) - 2 {
    True -> Ok(HessenbergResult(q: q_acc, h: t, iterations: k))
    False -> {
      let x =
        vector.from_list(
          list.map(
            list.drop(matrix.indices(matrix.rows(t)), up_to: k + 1),
            fn(i) { matrix.unsafe_get(t, i, k) },
          ),
        )
      case orthogonal.householder_matrix(x) {
        Error(ZeroNorm) -> hessenberg_loop(t, q_acc, k + 1)
        Error(e) -> Error(e)
        Ok(#(small_h, _)) -> {
          let h = embed(k + 1, matrix.rows(t), small_h)
          case matrix.mul(h, t) {
            Error(e) -> Error(e)
            Ok(ht) ->
              case matrix.mul(ht, h) {
                Error(e) -> Error(e)
                Ok(next_t) ->
                  case matrix.mul(q_acc, h) {
                    Error(e) -> Error(e)
                    Ok(next_q) -> hessenberg_loop(next_t, next_q, k + 1)
                  }
              }
          }
        }
      }
    }
  }
}

fn apply_jacobi_similarity(
  a: Matrix,
  p: Int,
  q: Int,
  c: Float,
  s: Float,
  t: Float,
) -> Matrix {
  let app = matrix.unsafe_get(a, p, p)
  let aqq = matrix.unsafe_get(a, q, q)
  let apq = matrix.unsafe_get(a, p, q)
  let next_app = app -. t *. apq
  let next_aqq = aqq +. t *. apq

  let assert Ok(result) =
    matrix.from_fn(rows: matrix.rows(a), cols: matrix.cols(a), with: fn(i, j) {
      case i == p && j == p {
        True -> next_app
        False ->
          case i == q && j == q {
            True -> next_aqq
            False ->
              case { i == p && j == q } || { i == q && j == p } {
                True -> 0.0
                False ->
                  case i == p {
                    True -> {
                      let akp = matrix.unsafe_get(a, j, p)
                      let akq = matrix.unsafe_get(a, j, q)
                      c *. akp -. s *. akq
                    }
                    False ->
                      case j == p {
                        True -> {
                          let aip = matrix.unsafe_get(a, i, p)
                          let aiq = matrix.unsafe_get(a, i, q)
                          c *. aip -. s *. aiq
                        }
                        False ->
                          case i == q {
                            True -> {
                              let akp = matrix.unsafe_get(a, j, p)
                              let akq = matrix.unsafe_get(a, j, q)
                              s *. akp +. c *. akq
                            }
                            False ->
                              case j == q {
                                True -> {
                                  let aip = matrix.unsafe_get(a, i, p)
                                  let aiq = matrix.unsafe_get(a, i, q)
                                  s *. aip +. c *. aiq
                                }
                                False -> matrix.unsafe_get(a, i, j)
                              }
                          }
                      }
                  }
              }
          }
      }
    })
  result
}

fn apply_jacobi_to_eigenvectors(
  v: Matrix,
  p: Int,
  q: Int,
  c: Float,
  s: Float,
) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: matrix.rows(v), cols: matrix.cols(v), with: fn(i, j) {
      case j == p {
        True ->
          c *. matrix.unsafe_get(v, i, p) -. s *. matrix.unsafe_get(v, i, q)
        False ->
          case j == q {
            True ->
              s *. matrix.unsafe_get(v, i, p) +. c *. matrix.unsafe_get(v, i, q)
            False -> matrix.unsafe_get(v, i, j)
          }
      }
    })
  result
}

fn single_shift_givens_step(
  t: Matrix,
  mu: Float,
) -> Result(#(Matrix, Matrix), NlaError) {
  let shifted_t = shifted(t, mu)
  case orthogonal.givens_qr(shifted_t) {
    Error(e) -> Error(e)
    Ok(qr) ->
      case matrix.mul(qr.r, qr.q) {
        Error(e) -> Error(e)
        Ok(rq) ->
          case matrix.add(rq, diagonal_shift(matrix.rows(t), mu)) {
            Error(e) -> Error(e)
            Ok(next_t) -> Ok(#(next_t, qr.q))
          }
      }
  }
}

fn double_shift_step(t: Matrix) -> Result(#(Matrix, Matrix), NlaError) {
  let n = matrix.rows(t)
  case n < 2 {
    True -> single_shift_givens_step(t, trailing_rayleigh_shift(t))
    False -> {
      let a = matrix.unsafe_get(t, n - 2, n - 2)
      let b = matrix.unsafe_get(t, n - 2, n - 1)
      let c = matrix.unsafe_get(t, n - 1, n - 2)
      let d = matrix.unsafe_get(t, n - 1, n - 1)
      let s = a +. d
      let p = a *. d -. b *. c
      case matrix.mul(t, t) {
        Error(e) -> Error(e)
        Ok(t2) ->
          case matrix.sub(t2, matrix.scale(t, s)) {
            Error(e) -> Error(e)
            Ok(partial) ->
              case matrix.add(partial, diagonal_shift(n, p)) {
                Error(e) -> Error(e)
                Ok(poly) ->
                  case matrix.frobenius_norm(poly) {
                    Error(e) -> Error(e)
                    Ok(norm) if norm <=. small ->
                      single_shift_givens_step(t, wilkinson_shift(t))
                    Ok(_) ->
                      case orthogonal.householder_qr(poly) {
                        Error(e) -> Error(e)
                        Ok(qr) -> {
                          let qt = matrix.transpose(qr.q)
                          case matrix.mul(qt, t) {
                            Error(e) -> Error(e)
                            Ok(qtt) ->
                              case matrix.mul(qtt, qr.q) {
                                Error(e) -> Error(e)
                                Ok(next_t) -> Ok(#(next_t, qr.q))
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

fn eigen_residual(a: Matrix, x: Vector) -> Result(#(Float, Float), NlaError) {
  case matrix.mul_vec(a, x) {
    Error(e) -> Error(e)
    Ok(ax) ->
      case rayleigh_quotient(x, ax) {
        Error(e) -> Error(e)
        Ok(lambda) ->
          case vector.axpy(0.0 -. lambda, x, ax) {
            Error(e) -> Error(e)
            Ok(r) ->
              case vector.norm2(r) {
                Error(e) -> Error(e)
                Ok(norm) -> Ok(#(lambda, norm))
              }
          }
      }
  }
}

fn rayleigh_quotient(x: Vector, ax: Vector) -> Result(Float, NlaError) {
  case vector.dot(x, x) {
    Error(e) -> Error(e)
    Ok(denominator) if denominator <=. small -> Error(ZeroNorm)
    Ok(denominator) ->
      case vector.dot(x, ax) {
        Error(e) -> Error(e)
        Ok(numerator) -> Ok(numerator /. denominator)
      }
  }
}

fn validate_symmetric(a: Matrix, tolerance: Float) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case is_symmetric(a, tolerance) {
        True -> Ok(Nil)
        False -> Error(InvalidInput("matrix must be symmetric"))
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

fn largest_off_diagonal(a: Matrix) -> #(Int, Int, Float) {
  list.fold(matrix.indices(matrix.rows(a)), #(0, 0, 0.0), fn(best, i) {
    list.fold(matrix.indices(matrix.cols(a)), best, fn(inner_best, j) {
      case i < j {
        True -> {
          let value = float.absolute_value(matrix.unsafe_get(a, i, j))
          case value >. inner_best.2 {
            True -> #(i, j, value)
            False -> inner_best
          }
        }
        False -> inner_best
      }
    })
  })
}

fn diagonal_vector(a: Matrix) -> Vector {
  vector.from_list(
    list.map(matrix.indices(matrix.rows(a)), fn(i) {
      matrix.unsafe_get(a, i, i)
    }),
  )
}

fn jacobi_t(tau: Float) -> Float {
  let denominator =
    float.absolute_value(tau) +. square_root_or_one(1.0 +. tau *. tau)
  case denominator <=. small {
    True -> 1.0
    False -> sign(tau) /. denominator
  }
}

fn reciprocal_square_root(value: Float) -> Float {
  case float.square_root(value) {
    Ok(root) if root >. small -> 1.0 /. root
    _ -> 0.0
  }
}

fn square_root_or_one(value: Float) -> Float {
  case float.square_root(value) {
    Ok(root) -> root
    Error(_) -> 1.0
  }
}

fn shifted(a: Matrix, shift: Float) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: matrix.rows(a), cols: matrix.cols(a), with: fn(i, j) {
      matrix.unsafe_get(a, i, j)
      -. case i == j {
        True -> shift
        False -> 0.0
      }
    })
  result
}

fn diagonal_shift(size: Int, shift: Float) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: size, cols: size, with: fn(i, j) {
      case i == j {
        True -> shift
        False -> 0.0
      }
    })
  result
}

fn trailing_rayleigh_shift(a: Matrix) -> Float {
  matrix.unsafe_get(a, matrix.rows(a) - 1, matrix.cols(a) - 1)
}

fn sign(value: Float) -> Float {
  case value <. 0.0 {
    True -> -1.0
    False -> 1.0
  }
}

fn lower_off_diagonal_norm(a: Matrix) -> Float {
  list.fold(matrix.indices(matrix.rows(a)), 0.0, fn(acc, i) {
    list.fold(matrix.indices(matrix.cols(a)), acc, fn(inner, j) {
      case i > j {
        True -> inner +. float.absolute_value(matrix.unsafe_get(a, i, j))
        False -> inner
      }
    })
  })
}

fn quasi_lower_off_diagonal_norm(a: Matrix, tolerance: Float) -> Float {
  lower_beyond_first_subdiagonal_norm(a)
  +. invalid_subdiagonal_norm(a, 0, tolerance)
}

fn lower_beyond_first_subdiagonal_norm(a: Matrix) -> Float {
  list.fold(matrix.indices(matrix.rows(a)), 0.0, fn(acc, i) {
    list.fold(matrix.indices(matrix.cols(a)), acc, fn(inner, j) {
      case i > j + 1 {
        True -> inner +. float.absolute_value(matrix.unsafe_get(a, i, j))
        False -> inner
      }
    })
  })
}

fn invalid_subdiagonal_norm(a: Matrix, index: Int, tolerance: Float) -> Float {
  case index >= matrix.rows(a) - 1 {
    True -> 0.0
    False -> {
      let value = float.absolute_value(matrix.unsafe_get(a, index + 1, index))
      case value <=. tolerance {
        True -> invalid_subdiagonal_norm(a, index + 1, tolerance)
        False ->
          case is_complex_schur_pair(a, index, tolerance) {
            True -> invalid_subdiagonal_norm(a, index + 2, tolerance)
            False -> value +. invalid_subdiagonal_norm(a, index + 1, tolerance)
          }
      }
    }
  }
}

fn schur_blocks_complex_eigenpairs(
  q: Matrix,
  t: Matrix,
  all_blocks: List(SchurBlock),
  remaining: List(SchurBlock),
  iterations: Int,
) -> Result(List(ComplexEigenpair), NlaError) {
  case remaining {
    [] -> Ok([])
    [block, ..rest] ->
      case schur_block_complex_eigenpairs(q, t, all_blocks, block, iterations) {
        Error(e) -> Error(e)
        Ok(block_pairs) ->
          case
            schur_blocks_complex_eigenpairs(q, t, all_blocks, rest, iterations)
          {
            Error(e) -> Error(e)
            Ok(rest_pairs) -> Ok(list.append(block_pairs, rest_pairs))
          }
      }
  }
}

fn schur_block_complex_eigenpairs(
  q: Matrix,
  t: Matrix,
  all_blocks: List(SchurBlock),
  block: SchurBlock,
  iterations: Int,
) -> Result(List(ComplexEigenpair), NlaError) {
  case block {
    RealBlock(index: _, value: value) -> {
      let lambda = complex.from_real(value)
      case schur_eigenvector(t, all_blocks, block, lambda) {
        Error(e) -> Error(e)
        Ok(local_vector) ->
          case transform_schur_vector(q, local_vector) {
            Error(e) -> Error(e)
            Ok(vector) ->
              case complex_eigen_residual(t, local_vector, lambda) {
                Error(e) -> Error(e)
                Ok(residual_norm) ->
                  Ok([
                    ComplexEigenpair(
                      value: lambda,
                      vector: vector,
                      residual_norm: residual_norm,
                      iterations: iterations,
                      converged: True,
                    ),
                  ])
              }
          }
      }
    }
    ComplexConjugateBlock(
      start: _,
      real: real_part,
      imaginary: imaginary,
      trace: _,
      determinant: _,
    ) -> {
      let positive = complex.new(real: real_part, imaginary: imaginary)
      let negative = complex.conjugate(positive)
      case
        schur_eigenpair_for_value(q, t, all_blocks, block, positive, iterations)
      {
        Error(e) -> Error(e)
        Ok(positive_pair) ->
          case
            schur_eigenpair_for_value(
              q,
              t,
              all_blocks,
              block,
              negative,
              iterations,
            )
          {
            Error(e) -> Error(e)
            Ok(negative_pair) -> Ok([positive_pair, negative_pair])
          }
      }
    }
  }
}

fn schur_eigenpair_for_value(
  q: Matrix,
  t: Matrix,
  all_blocks: List(SchurBlock),
  block: SchurBlock,
  lambda: complex.Complex,
  iterations: Int,
) -> Result(ComplexEigenpair, NlaError) {
  case schur_eigenvector(t, all_blocks, block, lambda) {
    Error(e) -> Error(e)
    Ok(local_vector) ->
      case transform_schur_vector(q, local_vector) {
        Error(e) -> Error(e)
        Ok(vector) ->
          case complex_eigen_residual(t, local_vector, lambda) {
            Error(e) -> Error(e)
            Ok(residual_norm) ->
              Ok(ComplexEigenpair(
                value: lambda,
                vector: vector,
                residual_norm: residual_norm,
                iterations: iterations,
                converged: True,
              ))
          }
      }
  }
}

fn schur_eigenvector(
  t: Matrix,
  blocks: List(SchurBlock),
  target: SchurBlock,
  lambda: complex.Complex,
) -> Result(complex.ComplexVector, NlaError) {
  case seed_schur_vector(t, target, lambda) {
    Error(e) -> Error(e)
    Ok(seed) ->
      case
        schur_back_substitute(
          t,
          blocks_before(blocks, block_start(target), []),
          lambda,
          seed,
        )
      {
        Error(e) -> Error(e)
        Ok(vector) -> complex.vector_normalize(vector)
      }
  }
}

fn schur_back_substitute(
  t: Matrix,
  reversed_blocks: List(SchurBlock),
  lambda: complex.Complex,
  y: complex.ComplexVector,
) -> Result(complex.ComplexVector, NlaError) {
  case reversed_blocks {
    [] -> Ok(y)
    [block, ..rest] ->
      case solve_previous_schur_block(t, block, lambda, y) {
        Error(e) -> Error(e)
        Ok(next_y) -> schur_back_substitute(t, rest, lambda, next_y)
      }
  }
}

fn solve_previous_schur_block(
  t: Matrix,
  block: SchurBlock,
  lambda: complex.Complex,
  y: complex.ComplexVector,
) -> Result(complex.ComplexVector, NlaError) {
  case block {
    RealBlock(index: index, value: value) -> {
      let rhs = complex.negate(schur_known_rhs(t, index, index + 1, y))
      let denominator = complex.sub(complex.from_real(value), lambda)
      case safe_complex_div(rhs, denominator) {
        Error(e) -> Error(e)
        Ok(entry) -> Ok(set_complex_vector(y, index, entry))
      }
    }
    ComplexConjugateBlock(
      start: start,
      real: _,
      imaginary: _,
      trace: _,
      determinant: _,
    ) -> {
      let end = start + 2
      let rhs0 = complex.negate(schur_known_rhs(t, start, end, y))
      let rhs1 = complex.negate(schur_known_rhs(t, start + 1, end, y))
      let a =
        complex.sub(
          complex.from_real(matrix.unsafe_get(t, start, start)),
          lambda,
        )
      let b = complex.from_real(matrix.unsafe_get(t, start, start + 1))
      let c = complex.from_real(matrix.unsafe_get(t, start + 1, start))
      let d =
        complex.sub(
          complex.from_real(matrix.unsafe_get(t, start + 1, start + 1)),
          lambda,
        )
      case solve_complex_2x2(a, b, c, d, rhs0, rhs1) {
        Error(e) -> Error(e)
        Ok(#(first, second)) ->
          Ok(set_complex_vector(
            set_complex_vector(y, start, first),
            start + 1,
            second,
          ))
      }
    }
  }
}

fn solve_complex_2x2(
  a: complex.Complex,
  b: complex.Complex,
  c: complex.Complex,
  d: complex.Complex,
  rhs0: complex.Complex,
  rhs1: complex.Complex,
) -> Result(#(complex.Complex, complex.Complex), NlaError) {
  let determinant = complex.sub(complex.mul(a, d), complex.mul(b, c))
  case
    safe_complex_div(
      complex.sub(complex.mul(rhs0, d), complex.mul(b, rhs1)),
      determinant,
    )
  {
    Error(e) -> Error(e)
    Ok(first) ->
      case
        safe_complex_div(
          complex.sub(complex.mul(a, rhs1), complex.mul(rhs0, c)),
          determinant,
        )
      {
        Error(e) -> Error(e)
        Ok(second) -> Ok(#(first, second))
      }
  }
}

fn schur_known_rhs(
  t: Matrix,
  row: Int,
  from_col: Int,
  y: complex.ComplexVector,
) -> complex.Complex {
  real_complex_dot(
    list.drop(matrix_row_values(t, row), up_to: from_col),
    list.drop(complex.vector_to_list(y), up_to: from_col),
  )
}

fn seed_schur_vector(
  t: Matrix,
  block: SchurBlock,
  lambda: complex.Complex,
) -> Result(complex.ComplexVector, NlaError) {
  case complex.vector_zeros(matrix.rows(t)) {
    Error(e) -> Error(e)
    Ok(zero_vector) ->
      case block {
        RealBlock(index: index, value: _) ->
          Ok(set_complex_vector(zero_vector, index, complex.one()))
        ComplexConjugateBlock(
          start: start,
          real: _,
          imaginary: _,
          trace: _,
          determinant: _,
        ) -> {
          case complex_block_seed(t, start, lambda) {
            Error(e) -> Error(e)
            Ok(#(first, second)) ->
              Ok(set_complex_vector(
                set_complex_vector(zero_vector, start, first),
                start + 1,
                second,
              ))
          }
        }
      }
  }
}

fn complex_block_seed(
  t: Matrix,
  start: Int,
  lambda: complex.Complex,
) -> Result(#(complex.Complex, complex.Complex), NlaError) {
  let a00 = matrix.unsafe_get(t, start, start)
  let a01 = matrix.unsafe_get(t, start, start + 1)
  let a10 = matrix.unsafe_get(t, start + 1, start)
  let a11 = matrix.unsafe_get(t, start + 1, start + 1)
  case float.absolute_value(a01) >=. float.absolute_value(a10) {
    True ->
      case float.absolute_value(a01) <=. small {
        True -> Error(InvalidInput("complex Schur block is singular"))
        False ->
          Ok(#(
            complex.from_real(a01),
            complex.sub(lambda, complex.from_real(a00)),
          ))
      }
    False ->
      case float.absolute_value(a10) <=. small {
        True -> Error(InvalidInput("complex Schur block is singular"))
        False ->
          Ok(#(
            complex.sub(lambda, complex.from_real(a11)),
            complex.from_real(a10),
          ))
      }
  }
}

fn transform_schur_vector(
  q: Matrix,
  y: complex.ComplexVector,
) -> Result(complex.ComplexVector, NlaError) {
  case matrix.cols(q) == complex.vector_dimension(y) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.cols(q)),
        actual: int.to_string(complex.vector_dimension(y)),
      ))
    True -> {
      let y_values = complex.vector_to_list(y)
      Ok(
        complex.vector_from_list(
          list.map(matrix.to_rows(q), fn(row) {
            real_complex_dot(row, y_values)
          }),
        ),
      )
    }
  }
}

fn complex_eigen_residual(
  a: Matrix,
  x: complex.ComplexVector,
  lambda: complex.Complex,
) -> Result(Float, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.cols(a) == complex.vector_dimension(x) {
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(matrix.cols(a)),
            actual: int.to_string(complex.vector_dimension(x)),
          ))
        True ->
          case real_matrix_mul_complex_vec(a, x) {
            Error(e) -> Error(e)
            Ok(ax) ->
              case complex.vector_axpy(complex.negate(lambda), x, ax) {
                Error(e) -> Error(e)
                Ok(residual) -> complex.vector_norm2(residual)
              }
          }
      }
  }
}

fn generalized_complex_eigen_residual(
  a: Matrix,
  b: Matrix,
  x: complex.ComplexVector,
  lambda: complex.Complex,
) -> Result(Float, NlaError) {
  case validate_square_pair(a, b) {
    Error(e) -> Error(e)
    Ok(_) ->
      case real_matrix_mul_complex_vec(a, x) {
        Error(e) -> Error(e)
        Ok(ax) ->
          case real_matrix_mul_complex_vec(b, x) {
            Error(e) -> Error(e)
            Ok(bx) ->
              case complex.vector_axpy(complex.negate(lambda), bx, ax) {
                Error(e) -> Error(e)
                Ok(residual) -> complex.vector_norm2(residual)
              }
          }
      }
  }
}

fn real_matrix_mul_complex_vec(
  a: Matrix,
  x: complex.ComplexVector,
) -> Result(complex.ComplexVector, NlaError) {
  case matrix.cols(a) == complex.vector_dimension(x) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.cols(a)),
        actual: int.to_string(complex.vector_dimension(x)),
      ))
    True -> {
      let x_values = complex.vector_to_list(x)
      Ok(
        complex.vector_from_list(
          list.map(matrix.to_rows(a), fn(row) {
            real_complex_dot(row, x_values)
          }),
        ),
      )
    }
  }
}

fn real_complex_dot(
  row: List(Float),
  values: List(complex.Complex),
) -> complex.Complex {
  list.fold(list.zip(row, with: values), complex.zero(), fn(acc, pair) {
    let #(matrix_value, vector_value) = pair
    complex.add(acc, complex.scale(vector_value, matrix_value))
  })
}

fn matrix_row_values(a: Matrix, row: Int) -> List(Float) {
  list.map(matrix.indices(matrix.cols(a)), fn(col) {
    matrix.unsafe_get(a, row, col)
  })
}

fn validate_square_pair(a: Matrix, b: Matrix) -> Result(Nil, NlaError) {
  case matrix.is_square(a), matrix.is_square(b) {
    False, _ -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    _, False -> Error(NotSquare(matrix.rows(b), matrix.cols(b)))
    True, True ->
      case matrix.rows(a) == matrix.rows(b) {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "matching square dimension "
              <> int.to_string(matrix.rows(a)),
            actual: int.to_string(matrix.rows(b)),
          ))
      }
  }
}

fn safe_complex_div(
  numerator: complex.Complex,
  denominator: complex.Complex,
) -> Result(complex.Complex, NlaError) {
  case complex.abs_squared(denominator) <=. small *. small {
    True -> Error(InvalidInput("ill-conditioned complex Schur solve"))
    False -> complex.div(numerator, denominator)
  }
}

fn blocks_before(
  blocks: List(SchurBlock),
  target_start: Int,
  acc: List(SchurBlock),
) -> List(SchurBlock) {
  case blocks {
    [] -> acc
    [block, ..rest] ->
      case block_start(block) < target_start {
        True -> blocks_before(rest, target_start, [block, ..acc])
        False -> acc
      }
  }
}

fn block_start(block: SchurBlock) -> Int {
  case block {
    RealBlock(index: index, value: _) -> index
    ComplexConjugateBlock(
      start: start,
      real: _,
      imaginary: _,
      trace: _,
      determinant: _,
    ) -> start
  }
}

fn set_complex_vector(
  vector: complex.ComplexVector,
  index: Int,
  value: complex.Complex,
) -> complex.ComplexVector {
  complex.vector_from_list(set_complex_at(
    complex.vector_to_list(vector),
    index,
    value,
  ))
}

fn set_complex_at(
  data: List(complex.Complex),
  index: Int,
  value: complex.Complex,
) -> List(complex.Complex) {
  case data {
    [] -> []
    [first, ..rest] ->
      case index == 0 {
        True -> [value, ..rest]
        False -> [first, ..set_complex_at(rest, index - 1, value)]
      }
  }
}

fn validate_schur_eigenpair_inputs(
  q: Matrix,
  t: Matrix,
) -> Result(Nil, NlaError) {
  case matrix.is_square(q), matrix.is_square(t) {
    False, _ -> Error(NotSquare(matrix.rows(q), matrix.cols(q)))
    _, False -> Error(NotSquare(matrix.rows(t), matrix.cols(t)))
    True, True ->
      case
        matrix.rows(q) == matrix.rows(t) && matrix.cols(q) == matrix.cols(t)
      {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: int.to_string(matrix.rows(t))
              <> "x"
              <> int.to_string(matrix.cols(t)),
            actual: int.to_string(matrix.rows(q))
              <> "x"
              <> int.to_string(matrix.cols(q)),
          ))
      }
  }
}

fn scan_schur_blocks(
  t: Matrix,
  index: Int,
  tolerance: Float,
  blocks: List(SchurBlock),
) -> List(SchurBlock) {
  case index >= matrix.rows(t) {
    True -> list.reverse(blocks)
    False ->
      case is_complex_schur_pair(t, index, tolerance) {
        True -> {
          let #(trace, determinant, discriminant) = schur_block_values(t, index)
          let imaginary = square_root_or_zero(0.0 -. discriminant) /. 2.0
          let block =
            ComplexConjugateBlock(
              start: index,
              real: trace /. 2.0,
              imaginary: imaginary,
              trace: trace,
              determinant: determinant,
            )
          scan_schur_blocks(t, index + 2, tolerance, [block, ..blocks])
        }
        False -> {
          let block =
            RealBlock(index: index, value: matrix.unsafe_get(t, index, index))
          scan_schur_blocks(t, index + 1, tolerance, [block, ..blocks])
        }
      }
  }
}

fn schur_block_eigenvalues(block: SchurBlock) -> List(Eigenvalue) {
  case block {
    RealBlock(index: _, value: value) -> [RealEigenvalue(value)]
    ComplexConjugateBlock(
      start: _,
      real: real_part,
      imaginary: imaginary,
      trace: _,
      determinant: _,
    ) -> [
      ComplexEigenvalue(real: real_part, imaginary: imaginary),
      ComplexEigenvalue(real: real_part, imaginary: 0.0 -. imaginary),
    ]
  }
}

fn is_complex_schur_pair(a: Matrix, start: Int, tolerance: Float) -> Bool {
  case start + 1 < matrix.rows(a) {
    False -> False
    True -> {
      let subdiagonal =
        float.absolute_value(matrix.unsafe_get(a, start + 1, start))
      let #(_, _, discriminant) = schur_block_values(a, start)
      subdiagonal >. tolerance && discriminant <. 0.0
    }
  }
}

fn schur_block_values(a: Matrix, start: Int) -> #(Float, Float, Float) {
  let a00 = matrix.unsafe_get(a, start, start)
  let a01 = matrix.unsafe_get(a, start, start + 1)
  let a10 = matrix.unsafe_get(a, start + 1, start)
  let a11 = matrix.unsafe_get(a, start + 1, start + 1)
  let trace = a00 +. a11
  let determinant = a00 *. a11 -. a01 *. a10
  #(trace, determinant, trace *. trace -. 4.0 *. determinant)
}

fn square_root_or_zero(value: Float) -> Float {
  case float.square_root(value) {
    Ok(root) -> root
    Error(_) -> 0.0
  }
}

fn embed(offset: Int, size: Int, small_h: Matrix) -> Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: size, cols: size, with: fn(i, j) {
      case i < offset || j < offset {
        True ->
          case i == j {
            True -> 1.0
            False -> 0.0
          }
        False -> matrix.unsafe_get(small_h, i - offset, j - offset)
      }
    })
  result
}

fn validate_square_vector(a: Matrix, x: Vector) -> Result(Nil, NlaError) {
  case matrix.is_square(a) {
    False -> Error(NotSquare(matrix.rows(a), matrix.cols(a)))
    True ->
      case matrix.rows(a) == vector.dimension(x) {
        True -> Ok(Nil)
        False ->
          Error(DimensionMismatch(
            expected: "matrix dimension " <> int_to_string(matrix.rows(a)),
            actual: "vector dimension " <> int_to_string(vector.dimension(x)),
          ))
      }
  }
}

fn int_to_string(value: Int) -> String {
  int.to_string(value)
}
