import gleam/float
import gleam/int
import gleam/list
import lumatrix/error.{
  type NlaError, DimensionMismatch, InvalidInput, OutOfBounds, ZeroNorm,
}

pub type Vector {
  Vector(size: Int, data: List(Float))
}

pub fn from_list(data: List(Float)) -> Vector {
  Vector(size: list.length(data), data: data)
}

pub fn zeros(size: Int) -> Result(Vector, NlaError) {
  case size >= 0 {
    True -> Ok(Vector(size: size, data: list.repeat(0.0, times: size)))
    False -> Error(InvalidInput("vector size must be non-negative"))
  }
}

pub fn basis(size: Int, index: Int) -> Result(Vector, NlaError) {
  case size > 0 && index >= 0 && index < size {
    True ->
      Ok(Vector(
        size: size,
        data: list.map(indices(size), fn(i) {
          case i == index {
            True -> 1.0
            False -> 0.0
          }
        }),
      ))
    False -> Error(InvalidInput("basis vector index is outside vector size"))
  }
}

pub fn dimension(vector: Vector) -> Int {
  vector.size
}

pub fn to_list(vector: Vector) -> List(Float) {
  vector.data
}

pub fn get(vector: Vector, index: Int) -> Result(Float, NlaError) {
  case index >= 0 && index < vector.size {
    True -> at(vector.data, index) |> result_from_nil(OutOfBounds(0, index))
    False -> Error(OutOfBounds(0, index))
  }
}

pub fn add(a: Vector, b: Vector) -> Result(Vector, NlaError) {
  zip_with(a, b, fn(x, y) { x +. y })
}

pub fn sub(a: Vector, b: Vector) -> Result(Vector, NlaError) {
  zip_with(a, b, fn(x, y) { x -. y })
}

pub fn scale(vector: Vector, scalar: Float) -> Vector {
  Vector(size: vector.size, data: list.map(vector.data, fn(x) { scalar *. x }))
}

pub fn axpy(a: Float, x: Vector, y: Vector) -> Result(Vector, NlaError) {
  zip_with(x, y, fn(xi, yi) { a *. xi +. yi })
}

pub fn dot(a: Vector, b: Vector) -> Result(Float, NlaError) {
  case a.size == b.size {
    True ->
      Ok(
        list.fold(list.zip(a.data, with: b.data), 0.0, fn(acc, pair) {
          let #(x, y) = pair
          acc +. x *. y
        }),
      )
    False ->
      Error(DimensionMismatch(
        expected: int.to_string(a.size),
        actual: int.to_string(b.size),
      ))
  }
}

pub fn norm2(vector: Vector) -> Result(Float, NlaError) {
  let squares = list.fold(vector.data, 0.0, fn(acc, x) { acc +. x *. x })
  case float.square_root(squares) {
    Ok(value) -> Ok(value)
    Error(_) -> Error(InvalidInput("cannot take square root of norm"))
  }
}

pub fn norm_inf(vector: Vector) -> Float {
  list.fold(vector.data, 0.0, fn(acc, x) {
    float.max(acc, float.absolute_value(x))
  })
}

pub fn normalize(vector: Vector) -> Result(Vector, NlaError) {
  case norm2(vector) {
    Ok(n) if n >. 0.0 -> Ok(scale(vector, 1.0 /. n))
    Ok(_) -> Error(ZeroNorm)
    Error(e) -> Error(e)
  }
}

pub fn approx_equal(a: Vector, b: Vector, tolerance: Float) -> Bool {
  case sub(a, b) {
    Ok(delta) -> norm_inf(delta) <=. tolerance
    Error(_) -> False
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

pub fn indices(size: Int) -> List(Int) {
  int.range(from: 0, to: size, with: [], run: fn(acc, i) { [i, ..acc] })
  |> list.reverse
}

fn at(data: List(a), index: Int) -> Result(a, Nil) {
  case index < 0 {
    True -> Error(Nil)
    False -> {
      let #(left, right) = list.split(data, at: index)
      case right {
        [value, ..] -> Ok(value)
        [] -> {
          let _ = left
          Error(Nil)
        }
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
