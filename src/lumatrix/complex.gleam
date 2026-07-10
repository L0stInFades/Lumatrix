import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, ArithmeticOverflow, DimensionMismatch, InvalidInput,
  NonFiniteInput, OutOfBounds, ZeroNorm,
}
import lumatrix/numerics
import lumatrix/vector.{type Vector}

pub type Complex {
  Complex(real: Float, imaginary: Float)
}

pub opaque type ComplexVector {
  ComplexVector(size: Int, data: List(Complex))
}

pub fn new(real real: Float, imaginary imaginary: Float) -> Complex {
  Complex(real: real, imaginary: imaginary)
}

pub fn try_new(
  real real: Float,
  imaginary imaginary: Float,
) -> Result(Complex, NlaError) {
  let value = new(real: real, imaginary: imaginary)
  case is_finite(value) {
    True -> Ok(value)
    False -> Error(NonFiniteInput("complex value"))
  }
}

pub fn is_finite(value: Complex) -> Bool {
  numerics.is_finite(value.real) && numerics.is_finite(value.imaginary)
}

pub fn from_real(value: Float) -> Complex {
  Complex(real: value, imaginary: 0.0)
}

pub fn zero() -> Complex {
  from_real(0.0)
}

pub fn one() -> Complex {
  from_real(1.0)
}

pub fn i() -> Complex {
  Complex(real: 0.0, imaginary: 1.0)
}

pub fn add(a: Complex, b: Complex) -> Complex {
  Complex(real: a.real +. b.real, imaginary: a.imaginary +. b.imaginary)
}

pub fn sub(a: Complex, b: Complex) -> Complex {
  Complex(real: a.real -. b.real, imaginary: a.imaginary -. b.imaginary)
}

pub fn negate(value: Complex) -> Complex {
  Complex(real: 0.0 -. value.real, imaginary: 0.0 -. value.imaginary)
}

pub fn scale(value: Complex, scalar: Float) -> Complex {
  Complex(real: value.real *. scalar, imaginary: value.imaginary *. scalar)
}

pub fn mul(a: Complex, b: Complex) -> Complex {
  Complex(
    real: a.real *. b.real -. a.imaginary *. b.imaginary,
    imaginary: a.real *. b.imaginary +. a.imaginary *. b.real,
  )
}

pub fn div(a: Complex, b: Complex) -> Result(Complex, NlaError) {
  let real_magnitude = float.absolute_value(b.real)
  let imaginary_magnitude = float.absolute_value(b.imaginary)
  case !is_finite(a) || !is_finite(b) {
    True -> Error(NonFiniteInput("complex division operands"))
    False if real_magnitude <=. 0.0 && imaginary_magnitude <=. 0.0 ->
      Error(InvalidInput("complex division by zero"))
    False -> {
      case real_magnitude >=. imaginary_magnitude {
        True -> {
          let ratio = b.imaginary /. b.real
          scaled_smith_division(a, b.real, b.imaginary, ratio, True)
        }
        False -> {
          let ratio = b.real /. b.imaginary
          scaled_smith_division(a, b.imaginary, b.real, ratio, False)
        }
      }
    }
  }
}

pub fn conjugate(value: Complex) -> Complex {
  Complex(real: value.real, imaginary: 0.0 -. value.imaginary)
}

pub fn abs_squared(value: Complex) -> Float {
  value.real *. value.real +. value.imaginary *. value.imaginary
}

pub fn abs(value: Complex) -> Result(Float, NlaError) {
  case is_finite(value) {
    False -> Error(NonFiniteInput("complex norm operand"))
    True ->
      case numerics.hypot(value.real, value.imaginary) {
        Ok(magnitude) -> Ok(magnitude)
        Error(_) -> Error(ArithmeticOverflow("complex norm"))
      }
  }
}

pub fn approx_equal(a: Complex, b: Complex, tolerance: Float) -> Bool {
  numerics.absolute_close(a.real, b.real, tolerance)
  && numerics.absolute_close(a.imaginary, b.imaginary, tolerance)
}

pub fn vector_from_list(data: List(Complex)) -> ComplexVector {
  ComplexVector(size: list.length(data), data: data)
}

pub fn vector_try_from_list(
  data: List(Complex),
) -> Result(ComplexVector, NlaError) {
  case list.all(data, satisfying: is_finite) {
    True -> Ok(vector_from_list(data))
    False -> Error(NonFiniteInput("complex vector data"))
  }
}

pub fn vector_from_real(values: Vector) -> ComplexVector {
  vector.to_list(values)
  |> list.map(from_real)
  |> vector_from_list
}

pub fn vector_zeros(size: Int) -> Result(ComplexVector, NlaError) {
  case size >= 0 {
    True ->
      Ok(ComplexVector(size: size, data: list.repeat(zero(), times: size)))
    False -> Error(InvalidInput("complex vector size must be non-negative"))
  }
}

pub fn vector_dimension(vector: ComplexVector) -> Int {
  vector.size
}

pub fn vector_to_list(vector: ComplexVector) -> List(Complex) {
  vector.data
}

pub fn vector_is_finite(vector: ComplexVector) -> Bool {
  list.all(vector.data, satisfying: is_finite)
}

pub fn vector_get(
  vector: ComplexVector,
  index: Int,
) -> Result(Complex, NlaError) {
  case index >= 0 && index < vector.size {
    True -> at(vector.data, index) |> result_from_nil(OutOfBounds(0, index))
    False -> Error(OutOfBounds(0, index))
  }
}

pub fn vector_add(
  a: ComplexVector,
  b: ComplexVector,
) -> Result(ComplexVector, NlaError) {
  checked_vector_zip_with(a, b, "complex vector addition", checked_add)
}

pub fn vector_sub(
  a: ComplexVector,
  b: ComplexVector,
) -> Result(ComplexVector, NlaError) {
  checked_vector_zip_with(a, b, "complex vector subtraction", checked_subtract)
}

pub fn vector_scale(vector: ComplexVector, scalar: Complex) -> ComplexVector {
  ComplexVector(
    size: vector.size,
    data: list.map(vector.data, fn(value) { mul(scalar, value) }),
  )
}

pub fn vector_scale_real(
  vector: ComplexVector,
  scalar: Float,
) -> ComplexVector {
  ComplexVector(
    size: vector.size,
    data: list.map(vector.data, fn(value) { scale(value, scalar) }),
  )
}

pub fn vector_axpy(
  scalar: Complex,
  x: ComplexVector,
  y: ComplexVector,
) -> Result(ComplexVector, NlaError) {
  case is_finite(scalar) {
    False -> Error(NonFiniteInput("complex AXPY scalar"))
    True ->
      checked_vector_zip_with(x, y, "complex AXPY", fn(xi, yi) {
        case checked_multiply(scalar, xi) {
          Error(_) -> Error(Nil)
          Ok(product) -> checked_add(product, yi)
        }
      })
  }
}

pub fn vector_dot_conjugate(
  a: ComplexVector,
  b: ComplexVector,
) -> Result(Complex, NlaError) {
  case a.size == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
    True ->
      case vector_is_finite(a) && vector_is_finite(b) {
        False -> Error(NonFiniteInput("complex dot product operands"))
        True -> checked_complex_dot_conjugate(list.zip(a.data, with: b.data))
      }
  }
}

pub fn vector_norm2(vector: ComplexVector) -> Result(Float, NlaError) {
  let values =
    list.flat_map(vector.data, fn(value) { [value.real, value.imaginary] })
  case vector_is_finite(vector) {
    False -> Error(NonFiniteInput("complex vector norm operand"))
    True ->
      case numerics.norm2(values) {
        Ok(norm) -> Ok(norm)
        Error(_) -> Error(ArithmeticOverflow("complex vector norm"))
      }
  }
}

pub fn vector_normalize(
  vector: ComplexVector,
) -> Result(ComplexVector, NlaError) {
  case vector_norm2(vector) {
    Error(e) -> Error(e)
    Ok(norm) if norm >. 0.0 ->
      Ok(ComplexVector(
        size: vector.size,
        data: list.map(vector.data, fn(value) {
          Complex(real: value.real /. norm, imaginary: value.imaginary /. norm)
        }),
      ))
    Ok(_) -> Error(ZeroNorm)
  }
}

pub fn vector_approx_equal(
  a: ComplexVector,
  b: ComplexVector,
  tolerance: Float,
) -> Bool {
  case vector_sub(a, b) {
    Error(_) -> False
    Ok(delta) ->
      case vector_norm2(delta) {
        Ok(norm) -> norm <=. tolerance
        Error(_) -> False
      }
  }
}

fn checked_vector_zip_with(
  a: ComplexVector,
  b: ComplexVector,
  operation: String,
  f: fn(Complex, Complex) -> Result(Complex, Nil),
) -> Result(ComplexVector, NlaError) {
  case a.size == b.size {
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
    True ->
      case vector_is_finite(a) && vector_is_finite(b) {
        False -> Error(NonFiniteInput(operation <> " operands"))
        True ->
          case
            list.try_map(list.zip(a.data, with: b.data), fn(pair) {
              f(pair.0, pair.1)
            })
          {
            Ok(values) -> Ok(ComplexVector(size: a.size, data: values))
            Error(_) -> Error(ArithmeticOverflow(operation))
          }
      }
  }
}

fn checked_add(a: Complex, b: Complex) -> Result(Complex, Nil) {
  case numerics.checked_add(a.real, b.real) {
    Error(_) -> Error(Nil)
    Ok(real) ->
      case numerics.checked_add(a.imaginary, b.imaginary) {
        Error(_) -> Error(Nil)
        Ok(imaginary) -> Ok(Complex(real: real, imaginary: imaginary))
      }
  }
}

fn checked_subtract(a: Complex, b: Complex) -> Result(Complex, Nil) {
  case numerics.checked_subtract(a.real, b.real) {
    Error(_) -> Error(Nil)
    Ok(real) ->
      case numerics.checked_subtract(a.imaginary, b.imaginary) {
        Error(_) -> Error(Nil)
        Ok(imaginary) -> Ok(Complex(real: real, imaginary: imaginary))
      }
  }
}

fn checked_multiply(a: Complex, b: Complex) -> Result(Complex, Nil) {
  case
    numerics.checked_multiply(a.real, b.real),
    numerics.checked_multiply(a.imaginary, b.imaginary),
    numerics.checked_multiply(a.real, b.imaginary),
    numerics.checked_multiply(a.imaginary, b.real)
  {
    Ok(real_left), Ok(real_right), Ok(imaginary_left), Ok(imaginary_right) ->
      case
        numerics.checked_subtract(real_left, real_right),
        numerics.checked_add(imaginary_left, imaginary_right)
      {
        Ok(real), Ok(imaginary) -> Ok(Complex(real: real, imaginary: imaginary))
        _, _ -> Error(Nil)
      }
    _, _, _, _ -> Error(Nil)
  }
}

fn checked_complex_dot_conjugate(
  values: List(#(Complex, Complex)),
) -> Result(Complex, NlaError) {
  let real_pairs =
    list.flat_map(values, fn(pair) {
      [#(pair.0.real, pair.1.real), #(pair.0.imaginary, pair.1.imaginary)]
    })
  let imaginary_pairs =
    list.flat_map(values, fn(pair) {
      [
        #(pair.0.real, pair.1.imaginary),
        #(0.0 -. pair.0.imaginary, pair.1.real),
      ]
    })
  case
    numerics.checked_dot_pairs(real_pairs),
    numerics.checked_dot_pairs(imaginary_pairs)
  {
    Ok(real), Ok(imaginary) -> Ok(Complex(real: real, imaginary: imaginary))
    _, _ -> Error(ArithmeticOverflow("complex dot product"))
  }
}

fn scaled_smith_division(
  numerator: Complex,
  dominant: Float,
  secondary: Float,
  ratio: Float,
  real_dominant: Bool,
) -> Result(Complex, NlaError) {
  let half_max = numerics.largest_finite() /. 2.0
  let numerator_factor = case
    float.max(
      float.absolute_value(numerator.real),
      float.absolute_value(numerator.imaginary),
    )
    >. half_max
  {
    True -> 0.5
    False -> 1.0
  }
  let denominator_factor = case float.absolute_value(dominant) >. half_max {
    True -> 0.5
    False -> 1.0
  }
  let scaled_real = numerator.real *. numerator_factor
  let scaled_imaginary = numerator.imaginary *. numerator_factor
  let denominator =
    dominant *. denominator_factor +. secondary *. denominator_factor *. ratio
  let #(real_numerator, imaginary_numerator) = case real_dominant {
    True -> #(
      scaled_real +. scaled_imaginary *. ratio,
      scaled_imaginary -. scaled_real *. ratio,
    )
    False -> #(
      scaled_real *. ratio +. scaled_imaginary,
      scaled_imaginary *. ratio -. scaled_real,
    )
  }
  let result_factor = denominator_factor /. numerator_factor
  case checked_quotient_component(real_numerator, denominator, result_factor) {
    Error(e) -> Error(e)
    Ok(real) ->
      case
        checked_quotient_component(
          imaginary_numerator,
          denominator,
          result_factor,
        )
      {
        Error(e) -> Error(e)
        Ok(imaginary) -> Ok(Complex(real: real, imaginary: imaginary))
      }
  }
}

fn checked_quotient_component(
  numerator: Float,
  denominator: Float,
  result_factor: Float,
) -> Result(Float, NlaError) {
  case numerics.checked_divide(numerator, denominator) {
    Error(_) -> Error(ArithmeticOverflow("complex division"))
    Ok(value) ->
      case numerics.checked_multiply(value, result_factor) {
        Error(_) -> Error(ArithmeticOverflow("complex division"))
        Ok(result) -> Ok(result)
      }
  }
}

fn at(data: List(a), index: Int) -> Result(a, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False -> {
      let #(_, right) = list.split(data, at: index)
      case right {
        [value, ..] -> Ok(value)
        [] -> Error(Nil)
      }
    }
  }
}

fn result_from_nil(
  result: Result(a, Nil),
  error: NlaError,
) -> Result(a, NlaError) {
  case result {
    Ok(value) -> Ok(value)
    Error(_) -> Error(error)
  }
}
