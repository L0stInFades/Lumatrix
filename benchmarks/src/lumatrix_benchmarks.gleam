import benchmark_clock
import gleam/float
import gleam/int
import gleam/io
import gleam/list
import lumatrix/direct
import lumatrix/iterative
import lumatrix/least_squares
import lumatrix/matrix
import lumatrix/svd
import lumatrix/vector

type Benchmark {
  Benchmark(
    operation: String,
    shape: String,
    warmups: Int,
    repetitions: Int,
    run: fn() -> Nil,
  )
}

pub fn main() -> Nil {
  io.println("operation,shape,repetitions,milliseconds_per_operation")
  list.each(benchmarks(), run_benchmark)
}

fn benchmarks() -> List(Benchmark) {
  [
    matvec_benchmark(128, 2, 20),
    matvec_benchmark(256, 1, 10),
    matvec_benchmark(512, 1, 3),
    matmul_benchmark(16, 1, 5),
    matmul_benchmark(32, 1, 3),
    matmul_benchmark(64, 1, 1),
    lu_benchmark(16, 1, 5),
    lu_benchmark(32, 1, 3),
    lu_benchmark(64, 1, 1),
    least_squares_benchmark(32, 16, 1, 3),
    least_squares_benchmark(64, 32, 1, 1),
    svd_benchmark(16, 8, 1, 2),
    svd_benchmark(32, 16, 1, 1),
    cg_benchmark(64, 1, 5),
    cg_benchmark(128, 1, 3),
    cg_benchmark(256, 1, 1),
  ]
}

fn run_benchmark(benchmark: Benchmark) -> Nil {
  repeat(benchmark.warmups, benchmark.run)
  let start = benchmark_clock.monotonic_milliseconds()
  repeat(benchmark.repetitions, benchmark.run)
  let elapsed = benchmark_clock.monotonic_milliseconds() -. start
  let average = elapsed /. int.to_float(benchmark.repetitions)
  io.println(
    benchmark.operation
    <> ","
    <> benchmark.shape
    <> ","
    <> int.to_string(benchmark.repetitions)
    <> ","
    <> float.to_string(average),
  )
}

fn repeat(count: Int, run: fn() -> Nil) -> Nil {
  case count <= 0 {
    True -> Nil
    False -> {
      run()
      repeat(count - 1, run)
    }
  }
}

fn matvec_benchmark(size: Int, warmups: Int, repetitions: Int) -> Benchmark {
  let a = tridiagonal_spd(size)
  let x = ramp_vector(size)
  Benchmark(
    operation: "matvec",
    shape: square_shape(size),
    warmups: warmups,
    repetitions: repetitions,
    run: fn() {
      let assert Ok(result) = matrix.mul_vec(a, x)
      consume_vector(result)
    },
  )
}

fn matmul_benchmark(size: Int, warmups: Int, repetitions: Int) -> Benchmark {
  let a = dense_matrix(size, size, 3)
  let b = dense_matrix(size, size, 11)
  Benchmark(
    operation: "matmul",
    shape: square_shape(size),
    warmups: warmups,
    repetitions: repetitions,
    run: fn() {
      let assert Ok(result) = matrix.mul(a, b)
      consume_matrix(result)
    },
  )
}

fn lu_benchmark(size: Int, warmups: Int, repetitions: Int) -> Benchmark {
  let a = tridiagonal_spd(size)
  let expected = ramp_vector(size)
  let assert Ok(b) = matrix.mul_vec(a, expected)
  Benchmark(
    operation: "lu_solve",
    shape: square_shape(size),
    warmups: warmups,
    repetitions: repetitions,
    run: fn() {
      let assert Ok(result) = direct.solve(a, b)
      consume_vector(result)
    },
  )
}

fn least_squares_benchmark(
  rows: Int,
  cols: Int,
  warmups: Int,
  repetitions: Int,
) -> Benchmark {
  let a = dense_matrix(rows, cols, 19)
  let expected = ramp_vector(cols)
  let assert Ok(b) = matrix.mul_vec(a, expected)
  Benchmark(
    operation: "least_squares_qr",
    shape: rectangular_shape(rows, cols),
    warmups: warmups,
    repetitions: repetitions,
    run: fn() {
      let assert Ok(result) = least_squares.solve(a, b)
      consume_vector(result.solution)
    },
  )
}

fn svd_benchmark(
  rows: Int,
  cols: Int,
  warmups: Int,
  repetitions: Int,
) -> Benchmark {
  let a = dense_matrix(rows, cols, 29)
  Benchmark(
    operation: "thin_svd",
    shape: rectangular_shape(rows, cols),
    warmups: warmups,
    repetitions: repetitions,
    run: fn() {
      let assert Ok(result) = svd.decompose(a)
      assert result.converged
      consume_vector(result.singular_values)
    },
  )
}

fn cg_benchmark(size: Int, warmups: Int, repetitions: Int) -> Benchmark {
  let a = tridiagonal_spd(size)
  let expected = ramp_vector(size)
  let assert Ok(b) = matrix.mul_vec(a, expected)
  let assert Ok(initial) = vector.zeros(size)
  Benchmark(
    operation: "conjugate_gradient",
    shape: square_shape(size),
    warmups: warmups,
    repetitions: repetitions,
    run: fn() {
      let assert Ok(result) =
        iterative.conjugate_gradient(a, b, initial, size * 2, 1.0e-8)
      assert result.converged
      consume_vector(result.solution)
    },
  )
}

fn dense_matrix(rows: Int, cols: Int, salt: Int) -> matrix.Matrix {
  let assert Ok(value) =
    matrix.from_fn(rows: rows, cols: cols, with: fn(i, j) {
      let raw = { i * 37 + j * 61 + salt * 17 + i * j * 3 } % 101
      let centered = int.to_float(raw - 50) /. 25.0
      case i == j {
        True -> centered +. int.to_float(cols) +. 1.0
        False -> centered
      }
    })
  value
}

fn tridiagonal_spd(size: Int) -> matrix.Matrix {
  let assert Ok(value) =
    matrix.from_fn(rows: size, cols: size, with: fn(i, j) {
      case i == j {
        True -> 4.0
        False if i + 1 == j || j + 1 == i -> -1.0
        False -> 0.0
      }
    })
  value
}

fn ramp_vector(size: Int) -> vector.Vector {
  vector.from_list(
    list.map(matrix.indices(size), fn(index) {
      int.to_float(index % 13 - 6) /. 7.0
    }),
  )
}

fn consume_vector(value: vector.Vector) -> Nil {
  let assert Ok(last) = vector.get(value, vector.dimension(value) - 1)
  let _ = last
  Nil
}

fn consume_matrix(value: matrix.Matrix) -> Nil {
  let last =
    matrix.unsafe_get(value, matrix.rows(value) - 1, matrix.cols(value) - 1)
  let _ = last
  Nil
}

fn square_shape(size: Int) -> String {
  rectangular_shape(size, size)
}

fn rectangular_shape(rows: Int, cols: Int) -> String {
  int.to_string(rows) <> "x" <> int.to_string(cols)
}
