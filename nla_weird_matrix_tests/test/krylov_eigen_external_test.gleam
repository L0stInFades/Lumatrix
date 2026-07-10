import lumatrix/eigen
import lumatrix/error
import lumatrix/krylov
import lumatrix/matrix
import lumatrix/vector
import nla_weird_matrix_tests/generated_cases
import nla_weird_support

pub fn arnoldi_generated_relation_test() {
  let assert Ok(a) =
    matrix.from_rows([[0.0, 2.0, -1.0], [3.0, 5.0, 4.0], [1.0, -2.0, 6.0]])
  let initial = vector.from_list([1.0, -1.0, 0.5])
  let assert Ok(result) = krylov.arnoldi(a, initial, 2, 1.0e-12)

  assert result.steps == 2
  assert matrix.rows(result.q) == 3
  assert matrix.cols(result.h) == result.steps
  assert_arnoldi_columns(a, result, 0)
}

pub fn lanczos_full_step_relation_on_spd_cases_test() {
  nla_weird_support.each_matrix_case(generated_cases.spd_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let initial = nla_weird_support.ramp_vector(matrix.rows(a))
    let assert Ok(result) = krylov.lanczos(a, initial, matrix.rows(a), 1.0e-12)
    let assert Ok(aq) = matrix.mul(a, result.q)
    let assert Ok(qt) = matrix.mul(result.q, result.t)
    let assert Ok(qtq) = matrix.mul(matrix.transpose(result.q), result.q)
    let assert Ok(identity) = matrix.identity(result.steps)

    assert result.steps == matrix.rows(a)
    assert result.happy_breakdown
    nla_weird_support.assert_matrix_close(aq, qt, 1.0e-5)
    nla_weird_support.assert_matrix_close(qtq, identity, 1.0e-5)
    assert_tridiagonal(result.t, 1.0e-7)
  })
}

pub fn gmres_variants_solve_generated_systems_test() {
  nla_weird_support.each_matrix_case(
    generated_cases.stationary_cases(),
    fn(sample) {
      let a = nla_weird_support.matrix_from_case(sample)
      let expected = nla_weird_support.ramp_vector(matrix.rows(a))
      let assert Ok(b) = matrix.mul_vec(a, expected)
      let assert Ok(initial) = vector.zeros(matrix.rows(a))
      let assert Ok(gmres) =
        krylov.gmres(a, b, initial, matrix.rows(a) + 2, 1.0e-9)
      let assert Ok(restarted) =
        krylov.restarted_gmres(a, b, initial, matrix.rows(a), 60, 1.0e-9)

      assert gmres.converged
      assert restarted.converged
      assert gmres.residual_norm <=. 1.0e-7
      assert restarted.residual_norm <=. 1.0e-7
      nla_weird_support.assert_vector_close(gmres.solution, expected, 1.0e-6)
      nla_weird_support.assert_vector_close(
        restarted.solution,
        expected,
        1.0e-6,
      )
    },
  )
}

pub fn krylov_error_paths_test() {
  let assert Ok(non_square) =
    matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let assert Ok(nonsymmetric) = matrix.from_rows([[1.0, 2.0], [0.0, 3.0]])
  let assert Ok(identity) = matrix.identity(2)
  let v2 = vector.from_list([1.0, 1.0])
  let v3 = vector.from_list([1.0, 1.0, 1.0])

  case krylov.arnoldi(non_square, v2, 1, 1.0e-8) {
    Error(error.NotSquare(rows: 2, cols: 3)) -> Nil
    _ -> panic as "Arnoldi should reject non-square matrices"
  }
  case krylov.arnoldi(identity, v3, 1, 1.0e-8) {
    Error(error.DimensionMismatch(
      expected: "square matrix dimension 2 and positive steps",
      actual: "vector dimension 3, steps 1",
    )) -> Nil
    _ -> panic as "Arnoldi should reject wrong initial vector size"
  }
  case krylov.lanczos(nonsymmetric, v2, 1, 1.0e-8) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "Lanczos should reject nonsymmetric matrices"
  }
  case krylov.restarted_gmres(identity, v2, v2, 0, 10, 1.0e-8) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "restarted GMRES should reject a nonpositive restart"
  }
  case krylov.gmres(identity, v3, v2, 2, 1.0e-8) {
    Error(error.DimensionMismatch(
      expected: "square matrix dimension 2, matching vectors and positive iterations",
      actual: "b=3, initial=2, iterations=2",
    )) -> Nil
    _ -> panic as "GMRES should reject mismatched vectors"
  }
}

pub fn power_and_inverse_power_methods_on_diagonal_case_test() {
  let assert Ok(a) =
    matrix.from_rows([[7.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.5]])
  let initial = vector.from_list([1.0, 1.0, 1.0])
  let assert Ok(dominant) = eigen.power_method(a, initial, 100, 1.0e-10)
  let assert Ok(near_half) =
    eigen.inverse_power_method(a, initial, 0.4, 100, 1.0e-10)

  assert dominant.converged
  assert near_half.converged
  assert dominant.residual_norm <=. 1.0e-8
  assert near_half.residual_norm <=. 1.0e-8
  nla_weird_support.assert_close_to(dominant.value, 7.0, 1.0e-7)
  nla_weird_support.assert_close_to(near_half.value, 0.5, 1.0e-7)
}

pub fn qr_iterations_and_histories_on_symmetric_case_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])
  let assert Ok(diagonal) = matrix.from_rows([[3.0, 0.0], [0.0, 1.0]])
  let assert Ok(basic) = eigen.qr_iteration(a, 100, 1.0e-6)
  let assert Ok(shifted) = eigen.shifted_qr_iteration(diagonal, 5, 1.0e-8)
  let assert Ok(symmetric) = eigen.symmetric_qr(a, 100, 1.0e-8)
  let assert Ok(implicit) = eigen.implicit_qr_iteration(a, 100, 1.0e-8)
  let assert Ok(double_shift) = eigen.double_shift_qr_iteration(a, 100, 1.0e-8)
  let assert Ok(history) = eigen.qr_convergence_history(a, 30, 1.0e-6)
  let assert Ok(shifted_history) =
    eigen.shifted_qr_convergence_history(diagonal, 5, 1.0e-8)
  let assert Ok(symmetric_history) =
    eigen.symmetric_qr_convergence_history(a, 30, 1.0e-8)

  assert basic.converged
  assert shifted.converged
  assert symmetric.converged
  assert implicit.converged
  assert double_shift.converged
  assert history.result.converged
  assert shifted_history.result.converged
  assert symmetric_history.result.converged
  assert_history_decreases(history)
  assert_history_decreases(symmetric_history)
  assert_schur_diagonal_is_3_and_1(basic)
  assert_schur_diagonal_is_3_and_1(shifted)
  assert_schur_diagonal_is_3_and_1(symmetric)
  assert_schur_diagonal_is_3_and_1(implicit)
  assert_schur_diagonal_is_3_and_1(double_shift)
}

pub fn hessenberg_and_tridiagonal_reductions_reconstruct_test() {
  let assert Ok(general) =
    matrix.from_rows([
      [3.0, 2.0, 1.0, 0.0],
      [1.0, 4.0, -1.0, 2.0],
      [0.0, -2.0, 1.0, 1.0],
      [0.0, 0.0, 3.0, 2.0],
    ])
  let assert Ok(symmetric) =
    matrix.from_rows([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 2.0]])
  let assert Ok(hessenberg) = eigen.hessenberg_reduction(general)
  let assert Ok(tridiagonal) = eigen.symmetric_tridiagonal_reduction(symmetric)

  assert_hessenberg(hessenberg.h, 1.0e-8)
  assert_tridiagonal(tridiagonal.t, 1.0e-8)
  assert_hessenberg_reconstructs(general, hessenberg, 1.0e-6)
  assert_tridiagonal_reconstructs(symmetric, tridiagonal, 1.0e-6)
}

pub fn symmetric_eigen_methods_decompose_matrix_test() {
  let assert Ok(two_by_two) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])
  let assert Ok(three_by_three) =
    matrix.from_rows([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 2.0]])
  let assert Ok(qr) = eigen.symmetric_qr_eigen(two_by_two, 100, 1.0e-8)
  let assert Ok(jacobi) = eigen.jacobi_eigen(three_by_three, 100, 1.0e-8)

  assert qr.converged
  assert jacobi.converged
  assert_eigendecomposition(two_by_two, qr, 1.0e-5)
  assert_eigendecomposition(three_by_three, jacobi, 1.0e-5)
}

pub fn real_schur_blocks_and_rotation_eigenvalues_test() {
  let assert Ok(rotation) = matrix.from_rows([[0.0, -1.0], [1.0, 0.0]])
  let assert Ok(schur) = eigen.real_schur_basic(rotation, 20, 1.0e-8)
  let assert Ok(blocks) = eigen.real_schur_blocks(schur.t, 1.0e-8)
  let assert Ok(values) = eigen.real_schur_eigenvalues(schur.t, 1.0e-8)
  let assert Ok(values_from_matrix) =
    eigen.real_schur_eigenvalues_of(rotation, 20, 1.0e-8)

  assert schur.converged
  case blocks {
    [
      eigen.ComplexConjugateBlock(
        start: start,
        real: real_part,
        imaginary: imaginary,
        trace: trace,
        determinant: determinant,
      ),
    ] -> {
      assert start == 0
      nla_weird_support.assert_close_to(real_part, 0.0, 1.0e-8)
      nla_weird_support.assert_close_to(imaginary, 1.0, 1.0e-8)
      nla_weird_support.assert_close_to(trace, 0.0, 1.0e-8)
      nla_weird_support.assert_close_to(determinant, 1.0, 1.0e-8)
    }
    _ -> panic as "expected one complex Schur block"
  }
  assert_rotation_eigenvalues(values)
  assert_rotation_eigenvalues(values_from_matrix)
}

pub fn wilkinson_shift_and_eigen_error_paths_test() {
  let assert Ok(one_by_one) = matrix.from_rows([[4.0]])
  let assert Ok(two_by_two) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])
  let assert Ok(non_square) =
    matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let assert Ok(nonsymmetric) = matrix.from_rows([[1.0, 2.0], [0.0, 3.0]])
  let assert Ok(not_schur) = matrix.from_rows([[1.0, 0.0], [1.0, 2.0]])
  let v2 = vector.from_list([1.0, 1.0])
  let v3 = vector.from_list([1.0, 1.0, 1.0])
  let zero = vector.from_list([0.0, 0.0])

  nla_weird_support.assert_close_to(
    eigen.wilkinson_shift(one_by_one),
    4.0,
    1.0e-8,
  )
  nla_weird_support.assert_close_to(
    eigen.wilkinson_shift(two_by_two),
    1.0,
    1.0e-8,
  )
  case eigen.power_method(non_square, v2, 10, 1.0e-8) {
    Error(error.NotSquare(rows: 2, cols: 3)) -> Nil
    _ -> panic as "power method should reject non-square matrices"
  }
  case eigen.power_method(two_by_two, v3, 10, 1.0e-8) {
    Error(error.DimensionMismatch(
      expected: "matrix dimension 2",
      actual: "vector dimension 3",
    )) -> Nil
    _ -> panic as "power method should reject mismatched vector size"
  }
  case eigen.power_method(two_by_two, zero, 10, 1.0e-8) {
    Error(error.ZeroNorm) -> Nil
    _ -> panic as "power method should reject a zero initial vector"
  }
  case eigen.symmetric_tridiagonal_reduction(nonsymmetric) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "symmetric reduction should reject nonsymmetric matrices"
  }
  case eigen.real_schur_blocks(non_square, 1.0e-8) {
    Error(error.NotSquare(rows: 2, cols: 3)) -> Nil
    _ -> panic as "Schur block scan should reject non-square matrices"
  }
  case eigen.real_schur_blocks(not_schur, 1.0e-8) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "Schur block scan should reject non-Schur matrices"
  }
}

fn assert_arnoldi_columns(
  a: matrix.Matrix,
  result: krylov.ArnoldiResult,
  column: Int,
) -> Nil {
  case column >= result.steps {
    True -> Nil
    False -> {
      let assert Ok(q_col) = matrix.col(result.q, column)
      let assert Ok(h_col) = matrix.col(result.h, column)
      let assert Ok(lhs) = matrix.mul_vec(a, q_col)
      let assert Ok(rhs) = matrix.mul_vec(result.q, h_col)
      nla_weird_support.assert_vector_close(lhs, rhs, 1.0e-7)
      assert_arnoldi_columns(a, result, column + 1)
    }
  }
}

fn assert_history_decreases(history: eigen.QrConvergenceHistory) -> Nil {
  let assert [first, ..] = history.steps
  let last = nla_weird_support.last_qr_step(history.steps)
  assert first.iteration == 0
  assert first.off_diagonal_norm >=. last.off_diagonal_norm
}

fn assert_schur_diagonal_is_3_and_1(result: eigen.SchurResult) -> Nil {
  nla_weird_support.assert_close_to(
    matrix.unsafe_get(result.t, 0, 0),
    3.0,
    1.0e-5,
  )
  nla_weird_support.assert_close_to(
    matrix.unsafe_get(result.t, 1, 1),
    1.0,
    1.0e-5,
  )
}

fn assert_hessenberg_reconstructs(
  original: matrix.Matrix,
  result: eigen.HessenbergResult,
  tolerance: Float,
) -> Nil {
  let assert Ok(qh) = matrix.mul(result.q, result.h)
  let assert Ok(reconstructed) = matrix.mul(qh, matrix.transpose(result.q))
  let assert Ok(qtq) = matrix.mul(matrix.transpose(result.q), result.q)
  let assert Ok(identity) = matrix.identity(matrix.cols(result.q))

  nla_weird_support.assert_matrix_close(reconstructed, original, tolerance)
  nla_weird_support.assert_matrix_close(qtq, identity, tolerance)
}

fn assert_tridiagonal_reconstructs(
  original: matrix.Matrix,
  result: eigen.TridiagonalResult,
  tolerance: Float,
) -> Nil {
  let assert Ok(qt) = matrix.mul(result.q, result.t)
  let assert Ok(reconstructed) = matrix.mul(qt, matrix.transpose(result.q))
  let assert Ok(qtq) = matrix.mul(matrix.transpose(result.q), result.q)
  let assert Ok(identity) = matrix.identity(matrix.cols(result.q))

  nla_weird_support.assert_matrix_close(reconstructed, original, tolerance)
  nla_weird_support.assert_matrix_close(qtq, identity, tolerance)
}

fn assert_eigendecomposition(
  a: matrix.Matrix,
  result: eigen.SymmetricEigenResult,
  tolerance: Float,
) -> Nil {
  let diagonal = nla_weird_support.diagonal_matrix_from_vector(result.diagonal)
  let assert Ok(av) = matrix.mul(a, result.eigenvectors)
  let assert Ok(vd) = matrix.mul(result.eigenvectors, diagonal)
  let assert Ok(vtv) =
    matrix.mul(matrix.transpose(result.eigenvectors), result.eigenvectors)
  let assert Ok(identity) = matrix.identity(matrix.cols(result.eigenvectors))

  nla_weird_support.assert_matrix_close(av, vd, tolerance)
  nla_weird_support.assert_matrix_close(vtv, identity, tolerance)
}

fn assert_hessenberg(a: matrix.Matrix, tolerance: Float) -> Nil {
  assert_hessenberg_cell(a, 0, 0, tolerance)
}

fn assert_hessenberg_cell(
  a: matrix.Matrix,
  row: Int,
  col: Int,
  tolerance: Float,
) -> Nil {
  case row >= matrix.rows(a) {
    True -> Nil
    False ->
      case col >= matrix.cols(a) {
        True -> assert_hessenberg_cell(a, row + 1, 0, tolerance)
        False -> {
          case row > col + 1 {
            True ->
              nla_weird_support.assert_close_to(
                matrix.unsafe_get(a, row, col),
                0.0,
                tolerance,
              )
            False -> Nil
          }
          assert_hessenberg_cell(a, row, col + 1, tolerance)
        }
      }
  }
}

fn assert_tridiagonal(a: matrix.Matrix, tolerance: Float) -> Nil {
  assert_tridiagonal_cell(a, 0, 0, tolerance)
}

fn assert_tridiagonal_cell(
  a: matrix.Matrix,
  row: Int,
  col: Int,
  tolerance: Float,
) -> Nil {
  case row >= matrix.rows(a) {
    True -> Nil
    False ->
      case col >= matrix.cols(a) {
        True -> assert_tridiagonal_cell(a, row + 1, 0, tolerance)
        False -> {
          case row > col + 1 || col > row + 1 {
            True ->
              nla_weird_support.assert_close_to(
                matrix.unsafe_get(a, row, col),
                0.0,
                tolerance,
              )
            False -> Nil
          }
          assert_tridiagonal_cell(a, row, col + 1, tolerance)
        }
      }
  }
}

fn assert_rotation_eigenvalues(values: List(eigen.Eigenvalue)) -> Nil {
  case values {
    [
      eigen.ComplexEigenvalue(real: real_pos, imaginary: imag_pos),
      eigen.ComplexEigenvalue(real: real_neg, imaginary: imag_neg),
    ] -> {
      nla_weird_support.assert_close_to(real_pos, 0.0, 1.0e-8)
      nla_weird_support.assert_close_to(real_neg, 0.0, 1.0e-8)
      nla_weird_support.assert_close_to(imag_pos, 1.0, 1.0e-8)
      nla_weird_support.assert_close_to(imag_neg, -1.0, 1.0e-8)
    }
    _ -> panic as "expected conjugate complex eigenvalues"
  }
}
