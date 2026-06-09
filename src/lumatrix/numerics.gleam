import gleam/float
import gleam/list

const default_relative_tolerance = 1.0e-14

type CompensatedSum {
  CompensatedSum(sum: Float, correction: Float)
}

type ScaledSquares {
  ScaledSquares(scale: Float, squares: CompensatedSum)
}

pub fn relative_tolerance() -> Float {
  default_relative_tolerance
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

pub fn dot_pairs(values: List(#(Float, Float))) -> Float {
  compensated_sum_map(values, fn(pair) { pair.0 *. pair.1 })
}

pub fn norm2(values: List(Float)) -> Result(Float, Nil) {
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
        Ok(root) -> Ok(state.scale *. root)
        Error(_) -> Error(Nil)
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
  case scale <=. 0.0 {
    True -> magnitude <=. 0.0
    False -> magnitude <=. tolerance *. scale
  }
}

pub fn relative_close(a: Float, b: Float, tolerance: Float) -> Bool {
  let diff = float.absolute_value(a -. b)
  let scale = float.max(float.absolute_value(a), float.absolute_value(b))
  case scale <=. 0.0 {
    True -> diff <=. 0.0
    False -> diff <=. tolerance *. scale
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
