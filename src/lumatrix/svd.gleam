import gleam/float
import gleam/int
import gleam/list
import gleam/order
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, NoConvergence,
}
import lumatrix/matrix.{type Matrix}
import lumatrix/vector.{type Vector}

const default_max_sweeps = 80

const default_tolerance = 1.0e-10

const completion_tolerance = 1.0e-10

/// Thin singular value decomposition.
///
/// For an m-by-n matrix, `u` is m-by-k, `singular_values` has length k, and
/// `vt` is k-by-n, where k = min(m, n). The reconstruction is
/// `u * diag(singular_values) * vt`. `off_diagonal_norm` is the largest scaled
/// column correlation left by the one-sided Jacobi iteration.
pub type SVD {
  SVD(
    u: Matrix,
    singular_values: Vector,
    vt: Matrix,
    iterations: Int,
    converged: Bool,
    off_diagonal_norm: Float,
  )
}

type JacobiState {
  JacobiState(
    work: Matrix,
    v: Matrix,
    iterations: Int,
    converged: Bool,
    off_diagonal_norm: Float,
  )
}

type PairStats {
  PairStats(alpha: Float, beta: Float, gamma: Float)
}

type SingularComponent {
  SingularComponent(value: Float, work_column: Vector, right: Vector)
}

pub fn decompose(a: Matrix) -> Result(SVD, NlaError) {
  decompose_with(a, default_max_sweeps, default_tolerance)
}

pub fn decompose_with(
  a: Matrix,
  max_sweeps: Int,
  tolerance: Float,
) -> Result(SVD, NlaError) {
  case validate_options(max_sweeps, tolerance) {
    Error(e) -> Error(e)
    Ok(_) ->
      case matrix.rows(a) >= matrix.cols(a) {
        True -> decompose_tall(a, max_sweeps, tolerance)
        False ->
          case decompose_tall(matrix.transpose(a), max_sweeps, tolerance) {
            Error(e) -> Error(e)
            Ok(transposed) ->
              Ok(SVD(
                u: matrix.transpose(transposed.vt),
                singular_values: transposed.singular_values,
                vt: matrix.transpose(transposed.u),
                iterations: transposed.iterations,
                converged: transposed.converged,
                off_diagonal_norm: transposed.off_diagonal_norm,
              ))
          }
      }
  }
}

pub fn singular_values(a: Matrix) -> Result(Vector, NlaError) {
  case decompose(a) {
    Error(e) -> Error(e)
    Ok(result) ->
      case require_converged(result) {
        Error(e) -> Error(e)
        Ok(_) -> Ok(result.singular_values)
      }
  }
}

pub fn norm2(a: Matrix) -> Result(Float, NlaError) {
  case decompose(a) {
    Error(e) -> Error(e)
    Ok(result) ->
      case require_converged(result) {
        Error(e) -> Error(e)
        Ok(_) -> Ok(norm2_from(result))
      }
  }
}

pub fn norm2_from(result: SVD) -> Float {
  max_singular_value(result.singular_values)
}

pub fn rank(result: SVD, tolerance: Float) -> Result(Int, NlaError) {
  case require_converged(result) {
    Error(e) -> Error(e)
    Ok(_) ->
      case validate_tolerance(tolerance) {
        Error(e) -> Error(e)
        Ok(_) -> {
          let cutoff = cutoff_for(result, tolerance)
          Ok(
            list.count(vector.to_list(result.singular_values), fn(value) {
              value >. cutoff
            }),
          )
        }
      }
  }
}

pub fn numerical_rank(a: Matrix, tolerance: Float) -> Result(Int, NlaError) {
  case decompose_with(a, default_max_sweeps, tolerance) {
    Error(e) -> Error(e)
    Ok(result) -> rank(result, tolerance)
  }
}

pub fn condition_number(
  result: SVD,
  tolerance: Float,
) -> Result(Float, NlaError) {
  case rank(result, tolerance) {
    Error(e) -> Error(e)
    Ok(r) -> {
      let values = vector.to_list(result.singular_values)
      let k = list.length(values)
      case r == k {
        False -> Error(InvalidInput("matrix is numerically rank deficient"))
        True -> {
          let largest = max_singular_value(result.singular_values)
          let smallest = min_singular_value(values, largest)
          case smallest >. 0.0 {
            True -> Ok(largest /. smallest)
            False -> Error(InvalidInput("matrix is numerically rank deficient"))
          }
        }
      }
    }
  }
}

pub fn condition_number_2(
  a: Matrix,
  tolerance: Float,
) -> Result(Float, NlaError) {
  case decompose_with(a, default_max_sweeps, tolerance) {
    Error(e) -> Error(e)
    Ok(result) -> condition_number(result, tolerance)
  }
}

pub fn pseudoinverse(a: Matrix) -> Result(Matrix, NlaError) {
  pseudoinverse_with(a, default_tolerance)
}

pub fn pseudoinverse_with(
  a: Matrix,
  tolerance: Float,
) -> Result(Matrix, NlaError) {
  case decompose_with(a, default_max_sweeps, tolerance) {
    Error(e) -> Error(e)
    Ok(result) -> pseudoinverse_from(result, tolerance)
  }
}

pub fn pseudoinverse_from(
  result: SVD,
  tolerance: Float,
) -> Result(Matrix, NlaError) {
  case require_converged(result) {
    Error(e) -> Error(e)
    Ok(_) ->
      case validate_tolerance(tolerance) {
        Error(e) -> Error(e)
        Ok(_) -> {
          let cutoff = cutoff_for(result, tolerance)
          let sigma_plus =
            vector.to_list(result.singular_values)
            |> list.map(fn(value) {
              case value >. cutoff {
                True -> 1.0 /. value
                False -> 0.0
              }
            })
          let assert Ok(sigma_plus_matrix) = matrix.diagonal(sigma_plus)
          let v = matrix.transpose(result.vt)
          case matrix.mul(v, sigma_plus_matrix) {
            Error(e) -> Error(e)
            Ok(v_sigma_plus) ->
              matrix.mul(v_sigma_plus, matrix.transpose(result.u))
          }
        }
      }
  }
}

pub fn solve(a: Matrix, b: Vector) -> Result(Vector, NlaError) {
  solve_with(a, b, default_tolerance)
}

pub fn solve_with(
  a: Matrix,
  b: Vector,
  tolerance: Float,
) -> Result(Vector, NlaError) {
  case matrix.rows(a) == vector.dimension(b) {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(matrix.rows(a)),
        actual: int.to_string(vector.dimension(b)),
      ))
    True ->
      case pseudoinverse_with(a, tolerance) {
        Error(e) -> Error(e)
        Ok(a_plus) -> matrix.mul_vec(a_plus, b)
      }
  }
}

fn decompose_tall(
  a: Matrix,
  max_sweeps: Int,
  tolerance: Float,
) -> Result(SVD, NlaError) {
  let assert Ok(v0) = matrix.identity(matrix.cols(a))
  case jacobi_loop(a, v0, 0, max_sweeps, tolerance) {
    Error(e) -> Error(e)
    Ok(state) ->
      build_tall_svd(
        state.work,
        state.v,
        state.iterations,
        state.converged,
        state.off_diagonal_norm,
        tolerance,
      )
  }
}

fn jacobi_loop(
  work: Matrix,
  v: Matrix,
  sweep: Int,
  max_sweeps: Int,
  tolerance: Float,
) -> Result(JacobiState, NlaError) {
  let off_diagonal_norm = off_diagonal_measure(work)
  case off_diagonal_norm <=. tolerance {
    True ->
      Ok(JacobiState(
        work: work,
        v: v,
        iterations: sweep,
        converged: True,
        off_diagonal_norm: off_diagonal_norm,
      ))
    False ->
      case sweep >= max_sweeps {
        True ->
          Ok(JacobiState(
            work: work,
            v: v,
            iterations: sweep,
            converged: False,
            off_diagonal_norm: off_diagonal_norm,
          ))
        False ->
          case jacobi_sweep(0, matrix.cols(work), work, v, tolerance) {
            Error(e) -> Error(e)
            Ok(#(next_work, next_v)) ->
              jacobi_loop(next_work, next_v, sweep + 1, max_sweeps, tolerance)
          }
      }
  }
}

fn jacobi_sweep(
  p: Int,
  n: Int,
  work: Matrix,
  v: Matrix,
  tolerance: Float,
) -> Result(#(Matrix, Matrix), NlaError) {
  case p >= n - 1 {
    True -> Ok(#(work, v))
    False ->
      case jacobi_sweep_q(p, p + 1, n, work, v, tolerance) {
        Error(e) -> Error(e)
        Ok(#(next_work, next_v)) ->
          jacobi_sweep(p + 1, n, next_work, next_v, tolerance)
      }
  }
}

fn jacobi_sweep_q(
  p: Int,
  q: Int,
  n: Int,
  work: Matrix,
  v: Matrix,
  tolerance: Float,
) -> Result(#(Matrix, Matrix), NlaError) {
  case q >= n {
    True -> Ok(#(work, v))
    False ->
      case rotate_pair_if_needed(work, v, p, q, tolerance) {
        Error(e) -> Error(e)
        Ok(#(next_work, next_v)) ->
          jacobi_sweep_q(p, q + 1, n, next_work, next_v, tolerance)
      }
  }
}

fn rotate_pair_if_needed(
  work: Matrix,
  v: Matrix,
  p: Int,
  q: Int,
  tolerance: Float,
) -> Result(#(Matrix, Matrix), NlaError) {
  let stats = pair_stats(work, p, q)
  let product = stats.alpha *. stats.beta
  case product <=. 0.0 {
    True -> Ok(#(work, v))
    False ->
      case float.square_root(product) {
        Error(_) -> Error(InvalidInput("cannot compute Jacobi pair norm"))
        Ok(denominator) ->
          case float.absolute_value(stats.gamma) <=. tolerance *. denominator {
            True -> Ok(#(work, v))
            False -> {
              let #(c, s) =
                jacobi_rotation(stats.alpha, stats.beta, stats.gamma)
              case rotate_columns(work, p, q, c, s) {
                Error(e) -> Error(e)
                Ok(next_work) ->
                  case rotate_columns(v, p, q, c, s) {
                    Error(e) -> Error(e)
                    Ok(next_v) -> Ok(#(next_work, next_v))
                  }
              }
            }
          }
      }
  }
}

fn jacobi_rotation(alpha: Float, beta: Float, gamma: Float) -> #(Float, Float) {
  let tau = { beta -. alpha } /. { 2.0 *. gamma }
  let t = case float.absolute_value(tau) >. 1.0e150 {
    True -> 0.5 /. tau
    False -> {
      let assert Ok(root) = float.square_root(1.0 +. tau *. tau)
      sign(tau) /. { float.absolute_value(tau) +. root }
    }
  }
  let assert Ok(c) = float.square_root(1.0 /. { 1.0 +. t *. t })
  #(c, c *. t)
}

fn rotate_columns(
  a: Matrix,
  p: Int,
  q: Int,
  c: Float,
  s: Float,
) -> Result(Matrix, NlaError) {
  matrix.from_fn(rows: matrix.rows(a), cols: matrix.cols(a), with: fn(i, j) {
    case j == p {
      True -> c *. matrix.unsafe_get(a, i, p) -. s *. matrix.unsafe_get(a, i, q)
      False ->
        case j == q {
          True ->
            s *. matrix.unsafe_get(a, i, p) +. c *. matrix.unsafe_get(a, i, q)
          False -> matrix.unsafe_get(a, i, j)
        }
    }
  })
}

fn build_tall_svd(
  work: Matrix,
  v: Matrix,
  iterations: Int,
  converged: Bool,
  off_diagonal_norm: Float,
  tolerance: Float,
) -> Result(SVD, NlaError) {
  let components =
    matrix.indices(matrix.cols(work))
    |> list.map(fn(j) {
      let assert Ok(work_column) = matrix.column(work, j)
      let assert Ok(right) = matrix.column(v, j)
      SingularComponent(
        value: stable_norm(work_column),
        work_column: work_column,
        right: right,
      )
    })
    |> list.sort(by: compare_components)
  let max_value =
    list.fold(components, 0.0, fn(best, component) {
      float.max(best, component.value)
    })
  let cutoff =
    tolerance
    *. int.to_float(max_int(matrix.rows(work), matrix.cols(work)))
    *. max_value
  case build_left_columns(components, [], [], matrix.rows(work), cutoff) {
    Error(e) -> Error(e)
    Ok(left_columns) -> {
      let right_columns =
        list.map(components, fn(component) { component.right })
      let singular_values =
        vector.from_list(
          list.map(components, fn(component) { component.value }),
        )
      case matrix.from_columns(left_columns) {
        Error(e) -> Error(e)
        Ok(u) ->
          case matrix.from_columns(right_columns) {
            Error(e) -> Error(e)
            Ok(v_matrix) ->
              Ok(SVD(
                u: u,
                singular_values: singular_values,
                vt: matrix.transpose(v_matrix),
                iterations: iterations,
                converged: converged,
                off_diagonal_norm: off_diagonal_norm,
              ))
          }
      }
    }
  }
}

fn build_left_columns(
  components: List(SingularComponent),
  used_reversed: List(Vector),
  output_reversed: List(Vector),
  size: Int,
  cutoff: Float,
) -> Result(List(Vector), NlaError) {
  case components {
    [] -> Ok(list.reverse(output_reversed))
    [component, ..rest] -> {
      let candidate = case component.value >. cutoff {
        True -> normalize_against(component.work_column, used_reversed)
        False -> complete_left_column(used_reversed, size, 0)
      }
      case candidate {
        Error(e) -> Error(e)
        Ok(left) ->
          build_left_columns(
            rest,
            [left, ..used_reversed],
            [left, ..output_reversed],
            size,
            cutoff,
          )
      }
    }
  }
}

fn normalize_against(
  column: Vector,
  used_reversed: List(Vector),
) -> Result(Vector, NlaError) {
  case orthogonalize(column, used_reversed) {
    Error(e) -> Error(e)
    Ok(cleaned) ->
      case stable_norm(cleaned) {
        norm if norm >. completion_tolerance ->
          Ok(vector.scale(cleaned, 1.0 /. norm))
        _ -> Error(InvalidInput("cannot build orthonormal SVD basis"))
      }
  }
}

fn complete_left_column(
  used_reversed: List(Vector),
  size: Int,
  index: Int,
) -> Result(Vector, NlaError) {
  case index >= size {
    True -> Error(InvalidInput("cannot complete SVD left basis"))
    False ->
      case vector.basis(size, index) {
        Error(e) -> Error(e)
        Ok(basis) ->
          case orthogonalize(basis, used_reversed) {
            Error(e) -> Error(e)
            Ok(candidate) -> {
              let norm = stable_norm(candidate)
              case norm >. completion_tolerance {
                True -> Ok(vector.scale(candidate, 1.0 /. norm))
                False -> complete_left_column(used_reversed, size, index + 1)
              }
            }
          }
      }
  }
}

fn orthogonalize(
  column: Vector,
  used_reversed: List(Vector),
) -> Result(Vector, NlaError) {
  list.try_fold(over: used_reversed, from: column, with: fn(acc, q) {
    case vector.dot(acc, q) {
      Error(e) -> Error(e)
      Ok(coeff) -> vector.axpy(0.0 -. coeff, q, acc)
    }
  })
}

fn off_diagonal_measure(work: Matrix) -> Float {
  off_diagonal_measure_p(work, 0, 0.0)
}

fn off_diagonal_measure_p(work: Matrix, p: Int, best: Float) -> Float {
  case p >= matrix.cols(work) - 1 {
    True -> best
    False ->
      off_diagonal_measure_p(
        work,
        p + 1,
        off_diagonal_measure_q(work, p, p + 1, best),
      )
  }
}

fn off_diagonal_measure_q(work: Matrix, p: Int, q: Int, best: Float) -> Float {
  case q >= matrix.cols(work) {
    True -> best
    False -> {
      let stats = pair_stats(work, p, q)
      let product = stats.alpha *. stats.beta
      let measure = case product <=. 0.0 {
        True -> 0.0
        False -> {
          let assert Ok(denominator) = float.square_root(product)
          float.absolute_value(stats.gamma) /. denominator
        }
      }
      off_diagonal_measure_q(work, p, q + 1, float.max(best, measure))
    }
  }
}

fn pair_stats(work: Matrix, p: Int, q: Int) -> PairStats {
  let scale =
    list.fold(matrix.indices(matrix.rows(work)), 0.0, fn(best, i) {
      float.max(
        best,
        float.max(
          float.absolute_value(matrix.unsafe_get(work, i, p)),
          float.absolute_value(matrix.unsafe_get(work, i, q)),
        ),
      )
    })
  case scale <=. 0.0 {
    True -> PairStats(alpha: 0.0, beta: 0.0, gamma: 0.0)
    False ->
      list.fold(
        matrix.indices(matrix.rows(work)),
        PairStats(alpha: 0.0, beta: 0.0, gamma: 0.0),
        fn(stats, i) {
          let x = matrix.unsafe_get(work, i, p) /. scale
          let y = matrix.unsafe_get(work, i, q) /. scale
          PairStats(
            alpha: stats.alpha +. x *. x,
            beta: stats.beta +. y *. y,
            gamma: stats.gamma +. x *. y,
          )
        },
      )
  }
}

fn stable_norm(column: Vector) -> Float {
  let values = vector.to_list(column)
  let scale =
    list.fold(values, 0.0, fn(best, value) {
      float.max(best, float.absolute_value(value))
    })
  case scale <=. 0.0 {
    True -> 0.0
    False -> {
      let sum =
        list.fold(values, 0.0, fn(acc, value) {
          let scaled = value /. scale
          acc +. scaled *. scaled
        })
      let assert Ok(root) = float.square_root(sum)
      scale *. root
    }
  }
}

fn cutoff_for(result: SVD, tolerance: Float) -> Float {
  tolerance
  *. int.to_float(max_int(matrix.rows(result.u), matrix.cols(result.vt)))
  *. max_singular_value(result.singular_values)
}

fn max_singular_value(values: Vector) -> Float {
  list.fold(vector.to_list(values), 0.0, float.max)
}

fn min_singular_value(values: List(Float), fallback: Float) -> Float {
  list.fold(values, fallback, float.min)
}

fn compare_components(
  a: SingularComponent,
  b: SingularComponent,
) -> order.Order {
  case float.compare(a.value, with: b.value) {
    order.Gt -> order.Lt
    order.Lt -> order.Gt
    order.Eq -> order.Eq
  }
}

fn validate_options(
  max_sweeps: Int,
  tolerance: Float,
) -> Result(Nil, NlaError) {
  case max_sweeps < 0 {
    True -> Error(InvalidInput("max_sweeps must be non-negative"))
    False -> validate_tolerance(tolerance)
  }
}

fn validate_tolerance(tolerance: Float) -> Result(Nil, NlaError) {
  case tolerance >. 0.0 {
    True -> Ok(Nil)
    False -> Error(InvalidInput("tolerance must be positive"))
  }
}

fn require_converged(result: SVD) -> Result(Nil, NlaError) {
  case result.converged {
    True -> Ok(Nil)
    False ->
      Error(NoConvergence(
        iterations: result.iterations,
        residual: result.off_diagonal_norm,
      ))
  }
}

fn sign(value: Float) -> Float {
  case value <. 0.0 {
    True -> -1.0
    False -> 1.0
  }
}

fn max_int(a: Int, b: Int) -> Int {
  case a > b {
    True -> a
    False -> b
  }
}
