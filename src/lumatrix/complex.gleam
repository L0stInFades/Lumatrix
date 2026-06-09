import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, OutOfBounds,
}
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
  let denominator = abs_squared(b)
  case denominator >. 0.0 {
    False -> Error(InvalidInput("complex division by zero"))
    True ->
      Ok(Complex(
        real: { a.real *. b.real +. a.imaginary *. b.imaginary } /. denominator,
        imaginary: { a.imaginary *. b.real -. a.real *. b.imaginary }
          /. denominator,
      ))
  }
}

pub fn conjugate(value: Complex) -> Complex {
  Complex(real: value.real, imaginary: 0.0 -. value.imaginary)
}

pub fn abs_squared(value: Complex) -> Float {
  value.real *. value.real +. value.imaginary *. value.imaginary
}

pub fn abs(value: Complex) -> Result(Float, NlaError) {
  case float.square_root(abs_squared(value)) {
    Ok(magnitude) -> Ok(magnitude)
    Error(_) -> Error(InvalidInput("cannot take square root of complex norm"))
  }
}

pub fn approx_equal(a: Complex, b: Complex, tolerance: Float) -> Bool {
  float.absolute_value(a.real -. b.real) <=. tolerance
  && float.absolute_value(a.imaginary -. b.imaginary) <=. tolerance
}

pub fn vector_from_list(data: List(Complex)) -> ComplexVector {
  ComplexVector(size: list.length(data), data: data)
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
  vector_zip_with(a, b, add)
}

pub fn vector_sub(
  a: ComplexVector,
  b: ComplexVector,
) -> Result(ComplexVector, NlaError) {
  vector_zip_with(a, b, sub)
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
  vector_zip_with(x, y, fn(xi, yi) { add(mul(scalar, xi), yi) })
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
      Ok(
        list.fold(list.zip(a.data, with: b.data), zero(), fn(acc, pair) {
          let #(x, y) = pair
          add(acc, mul(conjugate(x), y))
        }),
      )
  }
}

pub fn vector_norm2(vector: ComplexVector) -> Result(Float, NlaError) {
  let scale =
    list.fold(vector.data, 0.0, fn(best, value) {
      case abs(value) {
        Ok(magnitude) -> float.max(best, magnitude)
        Error(_) -> best
      }
    })
  case scale <=. 0.0 {
    True -> Ok(0.0)
    False -> {
      let sum =
        list.fold(vector.data, 0.0, fn(acc, value) {
          let scaled_real = value.real /. scale
          let scaled_imaginary = value.imaginary /. scale
          acc
          +. scaled_real
          *. scaled_real
          +. scaled_imaginary
          *. scaled_imaginary
        })
      case float.square_root(sum) {
        Ok(root) -> Ok(scale *. root)
        Error(_) ->
          Error(InvalidInput("cannot take square root of vector norm"))
      }
    }
  }
}

pub fn vector_normalize(
  vector: ComplexVector,
) -> Result(ComplexVector, NlaError) {
  case vector_norm2(vector) {
    Error(e) -> Error(e)
    Ok(norm) if norm >. 0.0 -> Ok(vector_scale_real(vector, 1.0 /. norm))
    Ok(_) -> Error(InvalidInput("cannot normalize zero complex vector"))
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

fn vector_zip_with(
  a: ComplexVector,
  b: ComplexVector,
  f: fn(Complex, Complex) -> Complex,
) -> Result(ComplexVector, NlaError) {
  case a.size == b.size {
    True ->
      Ok(ComplexVector(
        size: a.size,
        data: list.map(list.zip(a.data, with: b.data), fn(pair) {
          let #(x, y) = pair
          f(x, y)
        }),
      ))
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
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
