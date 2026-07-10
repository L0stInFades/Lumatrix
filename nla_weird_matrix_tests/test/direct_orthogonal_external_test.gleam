import gleam/list
import lumatrix/direct
import lumatrix/error
import lumatrix/matrix
import lumatrix/orthogonal
import lumatrix/vector
import nla_weird_matrix_tests/generated_cases
import nla_weird_support

pub fn gauss_transform_eliminates_selected_entry_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 0.0], [6.0, 1.0]])
  let assert Ok(transform) = direct.gauss_transform(matrix: a, pivot: 0, row: 1)
  let assert Ok(eliminated) = matrix.mul(transform, a)

  nla_weird_support.assert_close_to(
    matrix.unsafe_get(eliminated, 1, 0),
    0.0,
    nla_weird_support.tolerance(),
  )
}

pub fn lu_factor_reconstructs_permuted_spd_cases_test() {
  nla_weird_support.each_matrix_case(generated_cases.spd_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let assert Ok(factors) = direct.lu_factor(matrix: a)
    let assert Ok(pa) = matrix.mul(factors.p, a)
    let assert Ok(lu) = matrix.mul(factors.l, factors.u)

    nla_weird_support.assert_matrix_close(
      pa,
      lu,
      nla_weird_support.loose_tolerance(),
    )
  })
}

pub fn solve_gaussian_elimination_inverse_and_determinant_test() {
  nla_weird_support.each_matrix_case(generated_cases.spd_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let b = nla_weird_support.ramp_vector(matrix.rows(a))
    let assert Ok(factors) = direct.lu_factor(matrix: a)
    let assert Ok(x) = direct.solve(a, b)
    let assert Ok(x_from_lu) = direct.lu_solve(factors, b)
    let assert Ok(x_from_gaussian) = direct.gaussian_elimination(a, b)
    let assert Ok(inverse) = direct.inverse(a)
    let assert Ok(ai) = matrix.mul(a, inverse)
    let assert Ok(ia) = matrix.mul(inverse, a)
    let assert Ok(identity) = matrix.identity(matrix.rows(a))
    let assert Ok(det) = direct.determinant(a)

    nla_weird_support.assert_vector_close(
      x,
      x_from_lu,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_vector_close(
      x,
      x_from_gaussian,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_residual_small(
      a,
      x,
      b,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      ai,
      identity,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_matrix_close(
      ia,
      identity,
      nla_weird_support.loose_tolerance(),
    )
    assert det >. 0.0
  })
}

pub fn cholesky_factor_and_solve_spd_cases_test() {
  nla_weird_support.each_matrix_case(generated_cases.spd_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let b = nla_weird_support.alternating_vector(matrix.rows(a))
    let assert Ok(factors) = direct.cholesky_factor(matrix: a)
    let assert Ok(reconstructed) =
      matrix.mul(factors.l, matrix.transpose(factors.l))
    let assert Ok(x) = direct.solve_spd(a, b)
    let assert Ok(x_from_factor) = direct.cholesky_solve(factors, b)

    nla_weird_support.assert_matrix_close(
      reconstructed,
      a,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_vector_close(
      x,
      x_from_factor,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_residual_small(
      a,
      x,
      b,
      nla_weird_support.loose_tolerance(),
    )
  })
}

pub fn triangular_substitution_solves_known_systems_test() {
  let assert Ok(l) =
    matrix.from_rows([[2.0, 0.0, 0.0], [-1.0, 3.0, 0.0], [4.0, 2.0, 1.0]])
  let assert Ok(u) =
    matrix.from_rows([[2.0, -1.0, 4.0], [0.0, 3.0, 2.0], [0.0, 0.0, 5.0]])
  let expected = vector.from_list([1.0, -2.0, 0.5])
  let assert Ok(lb) = matrix.mul_vec(l, expected)
  let assert Ok(ub) = matrix.mul_vec(u, expected)
  let assert Ok(forward) = direct.forward_substitution(l, lb)
  let assert Ok(back) = direct.back_substitution(u, ub)

  nla_weird_support.assert_vector_close(
    forward,
    expected,
    nla_weird_support.tolerance(),
  )
  nla_weird_support.assert_vector_close(
    back,
    expected,
    nla_weird_support.tolerance(),
  )
}

pub fn direct_error_paths_test() {
  let assert Ok(non_square) =
    matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let assert Ok(singular) = matrix.from_rows([[1.0, 2.0], [2.0, 4.0]])
  let assert Ok(nonsymmetric) = matrix.from_rows([[2.0, 1.0], [3.0, 2.0]])
  let assert Ok(lower) = matrix.from_rows([[1.0, 0.0], [2.0, 0.0]])
  let b2 = vector.from_list([1.0, 2.0])
  let b3 = vector.from_list([1.0, 2.0, 3.0])
  let identity = unwrap_matrix(matrix.identity(2))
  let assert Ok(factors) = direct.lu_factor(matrix: identity)

  case direct.lu_factor(matrix: non_square) {
    Error(error.NotSquare(rows: 2, cols: 3)) -> Nil
    _ -> panic as "LU should reject non-square matrices"
  }
  case direct.solve(singular, b2) {
    Error(error.SingularMatrix(_)) -> Nil
    _ -> panic as "solve should reject singular matrices"
  }
  case direct.lu_solve(factors, b3) {
    Error(error.DimensionMismatch(expected: "2", actual: "3")) -> Nil
    _ -> panic as "lu_solve should reject wrong RHS length"
  }
  case direct.forward_substitution(lower, b3) {
    Error(error.DimensionMismatch(expected: "2x2", actual: "3")) -> Nil
    _ -> panic as "forward substitution should reject mismatched RHS"
  }
  case direct.back_substitution(lower, b2) {
    Error(error.SingularMatrix(1)) -> Nil
    _ -> panic as "back substitution should reject zero diagonal"
  }
  case direct.cholesky_factor(matrix: nonsymmetric) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "Cholesky should reject nonsymmetric matrices"
  }
}

pub fn householder_generated_vectors_test() {
  nla_weird_support.each_vector_case(generated_cases.vector_cases(), fn(sample) {
    let x = nla_weird_support.vector_from_case(sample)
    case orthogonal.householder_matrix(x) {
      Ok(#(h_matrix, target_norm)) -> {
        let assert Ok(y) = matrix.mul_vec(h_matrix, x)
        let assert Ok(householder) = orthogonal.householder(vector: x)
        let assert Ok(applied) = orthogonal.apply_householder(householder, x)
        let expected =
          vector.from_list([
            target_norm,
            ..list.repeat(0.0, times: vector.dimension(x) - 1)
          ])

        nla_weird_support.assert_vector_close(
          y,
          expected,
          nla_weird_support.loose_tolerance(),
        )
        nla_weird_support.assert_vector_close(
          applied,
          expected,
          nla_weird_support.loose_tolerance(),
        )
      }
      Error(error.ZeroNorm) -> {
        assert vector.norm_inf(x) <=. 0.0
      }
      Error(_) -> panic as "unexpected Householder error"
    }
  })
}

pub fn givens_matrix_and_left_application_test() {
  let assert Ok(a) = matrix.from_rows([[3.0, 1.0], [4.0, -2.0]])
  let assert Ok(rotation) = orthogonal.givens(3.0, 4.0)
  let assert Ok(g) = orthogonal.givens_matrix(2, 0, 1, rotation)
  let assert Ok(left_by_matrix) = matrix.mul(g, a)
  let assert Ok(left_by_kernel) =
    orthogonal.apply_givens_left(a, 0, 1, rotation)
  let assert Ok(rotated_vector) =
    matrix.mul_vec(g, vector.from_list([3.0, 4.0]))

  nla_weird_support.assert_matrix_close(
    left_by_kernel,
    left_by_matrix,
    nla_weird_support.tolerance(),
  )
  nla_weird_support.assert_vector_close(
    rotated_vector,
    vector.from_list([5.0, 0.0]),
    nla_weird_support.tolerance(),
  )
}

pub fn qr_variants_reconstruct_generated_tall_cases_test() {
  nla_weird_support.each_matrix_case(generated_cases.tall_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let assert Ok(householder) = orthogonal.qr_householder(a)
    let assert Ok(givens) = orthogonal.qr_givens(a)
    let assert Ok(classical) = orthogonal.qr_classical_gram_schmidt(a)
    let assert Ok(modified) = orthogonal.qr_modified_gram_schmidt(a)

    nla_weird_support.assert_qr_reconstructs(
      a,
      householder,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_qr_reconstructs(
      a,
      givens,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_qr_reconstructs(
      a,
      classical,
      nla_weird_support.loose_tolerance(),
    )
    nla_weird_support.assert_qr_reconstructs(
      a,
      modified,
      nla_weird_support.loose_tolerance(),
    )
  })
}

pub fn orthogonal_error_paths_test() {
  let assert Ok(wide) = matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let zero = vector.from_list([0.0, 0.0])
  let assert Ok(rotation) = orthogonal.givens(0.0, 0.0)

  case orthogonal.householder(vector: zero) {
    Error(error.ZeroNorm) -> Nil
    _ -> panic as "Householder should reject a zero vector"
  }
  case orthogonal.givens_matrix(2, 0, 0, rotation) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "Givens matrix should reject repeated axes"
  }
  case orthogonal.apply_givens_left(wide, 0, 5, rotation) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "Givens left application should reject bad axes"
  }
  case orthogonal.qr_householder(wide) {
    Error(error.DimensionMismatch(
      expected: "rows >= columns",
      actual: "rows < columns",
    )) -> Nil
    _ -> panic as "Householder QR should reject wide matrices"
  }
  case orthogonal.qr_givens(wide) {
    Error(error.DimensionMismatch(
      expected: "rows >= columns",
      actual: "rows < columns",
    )) -> Nil
    _ -> panic as "Givens QR should reject wide matrices"
  }
  case orthogonal.qr_classical_gram_schmidt(wide) {
    Error(error.DimensionMismatch(
      expected: "rows >= columns",
      actual: "rows < columns",
    )) -> Nil
    _ -> panic as "classical Gram-Schmidt should reject wide matrices"
  }
  case orthogonal.qr_modified_gram_schmidt(wide) {
    Error(error.DimensionMismatch(
      expected: "rows >= columns",
      actual: "rows < columns",
    )) -> Nil
    _ -> panic as "modified Gram-Schmidt should reject wide matrices"
  }
}

fn unwrap_matrix(
  value: Result(matrix.Matrix, error.NlaError),
) -> matrix.Matrix {
  let assert Ok(unwrapped) = value
  unwrapped
}
