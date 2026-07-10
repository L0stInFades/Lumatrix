import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, DimensionMismatch, InvalidInput,
  NonFiniteInput, OutOfBounds, ZeroNorm,
}
import lumatrix/internal/storage.{type Storage}
import lumatrix/numerics

/// A dense coordinate vector.
///
/// Lumatrix treats vectors as directionless coordinate arrays. In matrix-vector
/// products such as `matrix.mul_vec(a, x)`, the vector is interpreted as the
/// column vector `x` in `A * x`. Use `matrix.row_matrix` or
/// `matrix.column_matrix` when an explicit 1-by-n or n-by-1 matrix is needed.
pub opaque type Vector {
  Vector(size: Int, data: Storage)
}

pub fn from_list(data: List(Float)) -> Vector {
  Vector(size: list.length(data), data: storage.from_list(data))
}

/// Construct a vector while rejecting NaN and infinity.
///
/// `from_list` remains available for API compatibility, but numerical
/// algorithms validate those vectors before performing arithmetic.
pub fn try_from_list(data: List(Float)) -> Result(Vector, NlaError) {
  case list.all(data, satisfying: numerics.is_finite) {
    True -> Ok(from_list(data))
    False -> Error(NonFiniteInput("vector data"))
  }
}

pub fn zeros(size: Int) -> Result(Vector, NlaError) {
  case size >= 0 {
    True ->
      Ok(Vector(
        size: size,
        data: storage.from_list(list.repeat(0.0, times: size)),
      ))
    False -> Error(InvalidInput("vector size must be non-negative"))
  }
}

pub fn basis(size: Int, index: Int) -> Result(Vector, NlaError) {
  case size > 0 && index >= 0 && index < size {
    True ->
      Ok(Vector(
        size: size,
        data: storage.from_list(
          list.map(indices(size), fn(i) {
            case i == index {
              True -> 1.0
              False -> 0.0
            }
          }),
        ),
      ))
    False -> Error(InvalidInput("basis vector index is outside vector size"))
  }
}

pub fn dimension(vector: Vector) -> Int {
  vector.size
}

pub fn to_list(vector: Vector) -> List(Float) {
  storage.to_list(vector.data)
}

pub fn is_finite(vector: Vector) -> Bool {
  list.all(to_list(vector), satisfying: numerics.is_finite)
}

pub fn get(vector: Vector, index: Int) -> Result(Float, NlaError) {
  case index >= 0 && index < vector.size {
    True -> Ok(storage.unsafe_get(vector.data, index))
    False -> Error(OutOfBounds(0, index))
  }
}

pub fn add(a: Vector, b: Vector) -> Result(Vector, NlaError) {
  checked_zip_with(a, b, "vector addition", numerics.checked_add)
}

pub fn sub(a: Vector, b: Vector) -> Result(Vector, NlaError) {
  checked_zip_with(a, b, "vector subtraction", numerics.checked_subtract)
}

pub fn scale(vector: Vector, scalar: Float) -> Vector {
  Vector(
    size: vector.size,
    data: storage.map(vector.data, with: fn(x) { scalar *. x }),
  )
}

pub fn checked_scale(
  vector: Vector,
  scalar: Float,
) -> Result(Vector, NlaError) {
  case numerics.is_finite(scalar) {
    False -> Error(NonFiniteInput("vector scale"))
    True ->
      checked_map(vector, "vector scaling", fn(value) {
        numerics.checked_multiply(value, scalar)
      })
  }
}

pub fn divide(vector: Vector, scalar: Float) -> Result(Vector, NlaError) {
  case numerics.is_finite(scalar) {
    False -> Error(NonFiniteInput("vector divisor"))
    True if scalar == 0.0 ->
      Error(InvalidInput("vector divisor must be non-zero"))
    True ->
      checked_map(vector, "vector division", fn(value) {
        numerics.checked_divide(value, scalar)
      })
  }
}

pub fn axpy(a: Float, x: Vector, y: Vector) -> Result(Vector, NlaError) {
  case numerics.is_finite(a) {
    False -> Error(NonFiniteInput("AXPY scalar"))
    True ->
      checked_zip_with(x, y, "AXPY", fn(xi, yi) {
        case numerics.checked_multiply(a, xi) {
          Error(_) -> Error(Nil)
          Ok(product) -> numerics.checked_add(product, yi)
        }
      })
  }
}

pub fn dot(a: Vector, b: Vector) -> Result(Float, NlaError) {
  case a.size == b.size {
    True ->
      case is_finite(a) && is_finite(b) {
        False -> Error(NonFiniteInput("dot product operands"))
        True ->
          case
            numerics.checked_dot_pairs(list.zip(to_list(a), with: to_list(b)))
          {
            Ok(value) -> Ok(value)
            Error(_) -> Error(ArithmeticOverflow("dot product"))
          }
      }
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
  }
}

pub fn norm2(vector: Vector) -> Result(Float, NlaError) {
  case is_finite(vector) {
    False -> Error(NonFiniteInput("vector norm operand"))
    True ->
      case numerics.norm2(to_list(vector)) {
        Ok(value) -> Ok(value)
        Error(_) -> Error(ArithmeticOverflow("vector norm"))
      }
  }
}

pub fn norm_inf(vector: Vector) -> Float {
  list.fold(to_list(vector), 0.0, fn(acc, x) {
    float.max(acc, float.absolute_value(x))
  })
}

pub fn normalize(vector: Vector) -> Result(Vector, NlaError) {
  case norm2(vector) {
    Ok(n) if n >. 0.0 ->
      Ok(Vector(
        size: vector.size,
        data: storage.map(vector.data, with: fn(value) { value /. n }),
      ))
    Ok(_) -> Error(ZeroNorm)
    Error(e) -> Error(e)
  }
}

pub fn approx_equal(a: Vector, b: Vector, tolerance: Float) -> Bool {
  case a.size == b.size {
    False -> False
    True ->
      list.all(list.zip(to_list(a), with: to_list(b)), satisfying: fn(pair) {
        numerics.absolute_close(pair.0, pair.1, tolerance)
      })
  }
}

pub fn zip_with(
  a: Vector,
  b: Vector,
  f: fn(Float, Float) -> Float,
) -> Result(Vector, NlaError) {
  case a.size == b.size {
    True ->
      Ok(Vector(
        size: a.size,
        data: storage.from_list(
          list.map(list.zip(to_list(a), with: to_list(b)), fn(pair) {
            let #(x, y) = pair
            f(x, y)
          }),
        ),
      ))
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
  }
}

fn checked_map(
  vector: Vector,
  operation: String,
  f: fn(Float) -> Result(Float, Nil),
) -> Result(Vector, NlaError) {
  case is_finite(vector) {
    False -> Error(NonFiniteInput(operation <> " operand"))
    True ->
      case list.try_map(to_list(vector), f) {
        Ok(values) -> Ok(from_list(values))
        Error(_) -> Error(ArithmeticOverflow(operation))
      }
  }
}

fn checked_zip_with(
  a: Vector,
  b: Vector,
  operation: String,
  f: fn(Float, Float) -> Result(Float, Nil),
) -> Result(Vector, NlaError) {
  case a.size == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
    True ->
      case is_finite(a) && is_finite(b) {
        False -> Error(NonFiniteInput(operation <> " operands"))
        True ->
          case
            list.try_map(list.zip(to_list(a), with: to_list(b)), fn(pair) {
              f(pair.0, pair.1)
            })
          {
            Ok(values) -> Ok(from_list(values))
            Error(_) -> Error(ArithmeticOverflow(operation))
          }
      }
  }
}

pub fn indices(size: Int) -> List(Int) {
  int.range(from: 0, to: size, with: [], run: fn(acc, i) { [i, ..acc] })
  |> list.reverse
}
