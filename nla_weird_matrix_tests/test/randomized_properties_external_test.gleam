import gleam/int
import gleam/list
import lumatrix/direct
import lumatrix/matrix
import lumatrix/vector
import nla_weird_support

type Rng {
  Rng(state: Int)
}

pub fn deterministic_randomized_linear_algebra_properties_test() {
  run_randomized_cases(1, 64)
}

fn run_randomized_cases(seed: Int, remaining: Int) -> Nil {
  case remaining <= 0 {
    True -> Nil
    False -> {
      let size = seed % 5 + 1
      let #(raw, rng) = random_matrix(size, size, Rng(seed * 997 + 17))
      let #(expected_x, _) = random_vector(size, rng)
      let assert Ok(gram) = matrix.mul(matrix.transpose(raw), raw)
      let assert Ok(identity) = matrix.identity(size)
      let assert Ok(ridge) =
        matrix.checked_scale(identity, int.to_float(size) +. 1.0)
      let assert Ok(spd) = matrix.add(gram, ridge)
      let assert Ok(b) = matrix.mul_vec(spd, expected_x)
      let assert Ok(actual_x) = direct.solve(spd, b)
      let assert Ok(cholesky) = direct.cholesky_factor(spd)
      let assert Ok(reconstructed) =
        matrix.mul(cholesky.l, matrix.transpose(cholesky.l))
      let assert Ok(left_matrix) = matrix.mul(spd, raw)
      let assert Ok(left) = matrix.mul_vec(left_matrix, expected_x)
      let assert Ok(raw_x) = matrix.mul_vec(raw, expected_x)
      let assert Ok(right) = matrix.mul_vec(spd, raw_x)

      nla_weird_support.assert_vector_close(actual_x, expected_x, 1.0e-8)
      nla_weird_support.assert_residual_small(spd, actual_x, b, 1.0e-8)
      nla_weird_support.assert_matrix_close(reconstructed, spd, 1.0e-8)
      nla_weird_support.assert_matrix_close(
        matrix.transpose(matrix.transpose(raw)),
        raw,
        0.0,
      )
      nla_weird_support.assert_vector_close(left, right, 1.0e-8)

      run_randomized_cases(seed + 1, remaining - 1)
    }
  }
}

fn random_matrix(rows: Int, cols: Int, rng: Rng) -> #(matrix.Matrix, Rng) {
  let #(data, next_rng) = random_rows(rows, cols, rng, [])
  let assert Ok(value) = matrix.from_rows(list.reverse(data))
  #(value, next_rng)
}

fn random_rows(
  rows: Int,
  cols: Int,
  rng: Rng,
  accumulated: List(List(Float)),
) -> #(List(List(Float)), Rng) {
  case rows <= 0 {
    True -> #(accumulated, rng)
    False -> {
      let #(row, next_rng) = random_values(cols, rng, [])
      random_rows(rows - 1, cols, next_rng, [list.reverse(row), ..accumulated])
    }
  }
}

fn random_vector(size: Int, rng: Rng) -> #(vector.Vector, Rng) {
  let #(values, next_rng) = random_values(size, rng, [])
  #(vector.from_list(list.reverse(values)), next_rng)
}

fn random_values(
  count: Int,
  rng: Rng,
  accumulated: List(Float),
) -> #(List(Float), Rng) {
  case count <= 0 {
    True -> #(accumulated, rng)
    False -> {
      let #(value, next_rng) = random_float(rng)
      random_values(count - 1, next_rng, [value, ..accumulated])
    }
  }
}

fn random_float(rng: Rng) -> #(Float, Rng) {
  let next_state = { rng.state * 75 + 74 } % 65_521
  let centered = next_state % 2001 - 1000
  #(int.to_float(centered) /. 100.0, Rng(next_state))
}
