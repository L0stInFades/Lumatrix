import gleam/float
import gleam/list

const default_relative_tolerance = 1.0e-14

const largest_finite_float = 1.7976931348623157e308

type CompensatedSum {
  CompensatedSum(sum: Float, correction: Float)
}

type ScaledSquares {
  ScaledSquares(scale: Float, squares: CompensatedSum)
}

pub fn relative_tolerance() -> Float {
  default_relative_tolerance
}

pub fn largest_finite() -> Float {
  largest_finite_float
}

@external(erlang, "erlang", "is_float")
@external(javascript, "./numerics_ffi.mjs", "isFiniteNumber")
pub fn is_finite(value: Float) -> Bool

pub fn checked_add(a: Float, b: Float) -> Result(Float, Nil) {
  case is_finite(a) && is_finite(b) {
    False -> Error(Nil)
    True ->
      case
        { b >. 0.0 && a >. largest_finite_float -. b }
        || { b <. 0.0 && a <. { 0.0 -. largest_finite_float } -. b }
      {
        True -> Error(Nil)
        False -> Ok(a +. b)
      }
  }
}

pub fn checked_subtract(a: Float, b: Float) -> Result(Float, Nil) {
  checked_add(a, 0.0 -. b)
}

pub fn compensated_sum(values: List(Float)) -> Float {
  list.fold(values, zero_sum(), compensated_add)
  |> compensated_total
}

pub fn compensated_sum_map(values: List(a), f: fn(a) -> Float) -> Float {
  list.fold(values, zero_sum(), fn(acc, value) {
    compensated_add(acc, f(value))
  })
  |> compensated_total
}

pub fn checked_sum(values: List(Float)) -> Result(Float, Nil) {
  case list.all(values, satisfying: is_finite) {
    False -> Error(Nil)
    True -> {
      let scale = max_abs(values)
      case scale <=. 0.0 {
        True -> Ok(0.0)
        False ->
          checked_multiply(
            compensated_sum_map(values, fn(value) { value /. scale }),
            scale,
          )
      }
    }
  }
}

pub fn dot_pairs(values: List(#(Float, Float))) -> Float {
  case checked_dot_pairs(values) {
    Ok(value) -> value
    Error(_) -> {
      let sign = scaled_dot(values)
      case sign <. 0.0 {
        True -> 0.0 -. largest_finite_float
        False -> largest_finite_float
      }
    }
  }
}

pub fn checked_dot_pairs(values: List(#(Float, Float))) -> Result(Float, Nil) {
  case
    list.all(values, satisfying: fn(pair) {
      is_finite(pair.0) && is_finite(pair.1)
    })
  {
    False -> Error(Nil)
    True -> {
      let left_scale =
        list.fold(values, 0.0, fn(best, pair) {
          float.max(best, float.absolute_value(pair.0))
        })
      let right_scale =
        list.fold(values, 0.0, fn(best, pair) {
          float.max(best, float.absolute_value(pair.1))
        })
      case left_scale <=. 0.0 || right_scale <=. 0.0 {
        True -> Ok(0.0)
        False ->
          checked_rescale_dot(
            scaled_dot_with(values, left_scale, right_scale),
            left_scale,
            right_scale,
          )
      }
    }
  }
}

pub fn saturating_nonnegative_sum_map(
  values: List(a),
  f: fn(a) -> Float,
) -> Float {
  list.fold(values, zero_sum(), fn(acc, value) {
    let next_value = float.max(0.0, f(value))
    let total = compensated_total(acc)
    case
      total >=. largest_finite_float
      || next_value >. largest_finite_float -. total
    {
      True -> CompensatedSum(sum: largest_finite_float, correction: 0.0)
      False -> compensated_add(acc, next_value)
    }
  })
  |> compensated_total
}

pub fn checked_multiply(a: Float, b: Float) -> Result(Float, Nil) {
  let a_magnitude = float.absolute_value(a)
  let b_magnitude = float.absolute_value(b)
  case !is_finite(a) || !is_finite(b) {
    True -> Error(Nil)
    False if a_magnitude <=. 0.0 || b_magnitude <=. 0.0 -> Ok(0.0)
    False ->
      case a_magnitude <=. 1.0 || b_magnitude <=. 1.0 {
        True -> Ok(a *. b)
        False ->
          case a_magnitude >. largest_finite_float /. b_magnitude {
            True -> Error(Nil)
            False -> Ok(a *. b)
          }
      }
  }
}

pub fn checked_divide(
  numerator: Float,
  denominator: Float,
) -> Result(Float, Nil) {
  let numerator_magnitude = float.absolute_value(numerator)
  let denominator_magnitude = float.absolute_value(denominator)
  case !is_finite(numerator) || !is_finite(denominator) {
    True -> Error(Nil)
    False if denominator_magnitude <=. 0.0 -> Error(Nil)
    False ->
      case numerator_magnitude <=. 0.0 || denominator_magnitude >=. 1.0 {
        True -> Ok(numerator /. denominator)
        False ->
          case
            numerator_magnitude >. largest_finite_float *. denominator_magnitude
          {
            True -> Error(Nil)
            False -> Ok(numerator /. denominator)
          }
      }
  }
}

pub fn norm2(values: List(Float)) -> Result(Float, Nil) {
  case list.all(values, satisfying: is_finite) {
    False -> Error(Nil)
    True -> {
      let state =
        list.fold(
          values,
          ScaledSquares(scale: 0.0, squares: zero_sum()),
          add_square,
        )
      case state.scale <=. 0.0 {
        True -> Ok(0.0)
        False ->
          case float.square_root(compensated_total(state.squares)) {
            Ok(root) -> checked_multiply(state.scale, root)
            Error(_) -> Error(Nil)
          }
      }
    }
  }
}

pub fn hypot(a: Float, b: Float) -> Result(Float, Nil) {
  norm2([a, b])
}

pub fn max_abs(values: List(Float)) -> Float {
  list.fold(values, 0.0, fn(best, value) {
    float.max(best, float.absolute_value(value))
  })
}

pub fn relative_near_zero(
  value: Float,
  scale: Float,
  tolerance: Float,
) -> Bool {
  let magnitude = float.absolute_value(value)
  case magnitude <=. 0.0 {
    True -> True
    False ->
      case scale <=. 0.0 || tolerance <=. 0.0 {
        True -> False
        False ->
          case tolerance <=. 1.0 {
            True -> magnitude <=. tolerance *. scale
            False -> magnitude <=. scale || magnitude /. tolerance <=. scale
          }
      }
  }
}

pub fn relative_close(a: Float, b: Float, tolerance: Float) -> Bool {
  case tolerance <. 0.0 {
    True -> False
    False -> {
      let scale = float.max(float.absolute_value(a), float.absolute_value(b))
      case scale <=. 0.0 {
        True -> a == b
        False -> float.absolute_value(a /. scale -. b /. scale) <=. tolerance
      }
    }
  }
}

/// Compare two values relative to a caller-supplied global scale.
///
/// Dividing before subtracting avoids overflow when finite values have
/// opposite signs. Callers such as matrix symmetry checks should supply a
/// norm that is at least as large as either entry.
pub fn relative_close_at_scale(
  a: Float,
  b: Float,
  scale: Float,
  tolerance: Float,
) -> Bool {
  case
    !is_finite(a)
    || !is_finite(b)
    || !is_finite(scale)
    || !is_finite(tolerance)
    || scale <. 0.0
    || tolerance <. 0.0
  {
    True -> False
    False ->
      case scale <=. 0.0 {
        True -> a == b
        False -> float.absolute_value(a /. scale -. b /. scale) <=. tolerance
      }
  }
}

pub fn absolute_close(a: Float, b: Float, tolerance: Float) -> Bool {
  case tolerance <. 0.0 {
    True -> False
    False -> {
      let scale = float.max(float.absolute_value(a), float.absolute_value(b))
      case scale <=. 1.0 {
        True -> float.absolute_value(a -. b) <=. tolerance
        False ->
          float.absolute_value(a /. scale -. b /. scale) <=. tolerance /. scale
      }
    }
  }
}

fn scaled_dot(values: List(#(Float, Float))) -> Float {
  let left_scale =
    list.fold(values, 0.0, fn(best, pair) {
      float.max(best, float.absolute_value(pair.0))
    })
  let right_scale =
    list.fold(values, 0.0, fn(best, pair) {
      float.max(best, float.absolute_value(pair.1))
    })
  scaled_dot_with(values, left_scale, right_scale)
}

fn scaled_dot_with(
  values: List(#(Float, Float)),
  left_scale: Float,
  right_scale: Float,
) -> Float {
  case left_scale <=. 0.0 || right_scale <=. 0.0 {
    True -> 0.0
    False ->
      compensated_sum_map(values, fn(pair) {
        { pair.0 /. left_scale } *. { pair.1 /. right_scale }
      })
  }
}

fn checked_rescale_dot(
  value: Float,
  left_scale: Float,
  right_scale: Float,
) -> Result(Float, Nil) {
  let #(first_scale, second_scale) = case float.absolute_value(value) <=. 1.0 {
    True ->
      case left_scale >=. right_scale {
        True -> #(left_scale, right_scale)
        False -> #(right_scale, left_scale)
      }
    False ->
      case left_scale <=. right_scale {
        True -> #(left_scale, right_scale)
        False -> #(right_scale, left_scale)
      }
  }
  case checked_multiply(value, first_scale) {
    Error(_) -> Error(Nil)
    Ok(partial) -> checked_multiply(partial, second_scale)
  }
}

fn zero_sum() -> CompensatedSum {
  CompensatedSum(sum: 0.0, correction: 0.0)
}

fn compensated_add(acc: CompensatedSum, value: Float) -> CompensatedSum {
  let next = acc.sum +. value
  let lost = case
    float.absolute_value(acc.sum) >=. float.absolute_value(value)
  {
    True -> { acc.sum -. next } +. value
    False -> { value -. next } +. acc.sum
  }
  let correction = acc.correction +. lost
  CompensatedSum(sum: next, correction: correction)
}

fn compensated_total(acc: CompensatedSum) -> Float {
  acc.sum +. acc.correction
}

fn add_square(state: ScaledSquares, value: Float) -> ScaledSquares {
  let magnitude = float.absolute_value(value)
  case magnitude <=. 0.0 {
    True -> state
    False ->
      case state.scale <=. 0.0 {
        True ->
          ScaledSquares(
            scale: magnitude,
            squares: compensated_add(zero_sum(), 1.0),
          )
        False ->
          case magnitude >. state.scale {
            True -> {
              let ratio = state.scale /. magnitude
              let rescaled = compensated_total(state.squares) *. ratio *. ratio
              ScaledSquares(
                scale: magnitude,
                squares: compensated_add(
                  CompensatedSum(sum: rescaled, correction: 0.0),
                  1.0,
                ),
              )
            }
            False -> {
              let ratio = magnitude /. state.scale
              ScaledSquares(
                scale: state.scale,
                squares: compensated_add(state.squares, ratio *. ratio),
              )
            }
          }
      }
  }
}
