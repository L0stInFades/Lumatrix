import gleam/float
import gleeunit
import lumatrix/complex
import lumatrix/direct
import lumatrix/eigen
import lumatrix/error
import lumatrix/error_analysis
import lumatrix/iterative
import lumatrix/krylov
import lumatrix/least_squares
import lumatrix/matrix
import lumatrix/orthogonal
import lumatrix/sparse
import lumatrix/svd
import lumatrix/vector

const tolerance = 1.0e-8

pub fn main() -> Nil {
  gleeunit.main()
}

pub fn matrix_vector_product_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
  let x = vector.from_list([1.0, 1.0])

  let assert Ok(y) = matrix.mul_vec(a, x)
  let assert Ok(aty) = matrix.transpose_mul_vec(a, x)

  assert vector.approx_equal(y, vector.from_list([3.0, 7.0]), tolerance)
  assert vector.approx_equal(aty, vector.from_list([4.0, 6.0]), tolerance)
}

pub fn matrix_column_and_orientation_helpers_test() {
  let first = vector.from_list([1.0, 3.0])
  let second = vector.from_list([2.0, 4.0])
  let assert Ok(a) = matrix.from_columns([first, second])
  let assert [first_out, second_out] = matrix.to_columns(a)
  let assert Ok(first_column) = matrix.column(a, 0)
  let assert Ok(first_col_alias) = matrix.col(a, 0)
  let replacement = vector.from_list([9.0, 8.0])
  let assert Ok(updated) = matrix.set_column(a, 1, replacement)
  let assert Ok(column_matrix) = matrix.column_matrix(first)
  let assert Ok(row_matrix) = matrix.row_matrix(second)

  assert matrix.to_rows(a) == [[1.0, 2.0], [3.0, 4.0]]
  assert vector.approx_equal(first_out, first, tolerance)
  assert vector.approx_equal(second_out, second, tolerance)
  assert vector.approx_equal(first_column, first, tolerance)
  assert vector.approx_equal(first_col_alias, first, tolerance)
  assert matrix.to_rows(updated) == [[1.0, 9.0], [3.0, 8.0]]
  assert matrix.to_rows(column_matrix) == [[1.0], [3.0]]
  assert matrix.to_rows(row_matrix) == [[2.0, 4.0]]

  case matrix.set_row(a, 2, first) {
    Error(error.OutOfBounds(2, 0)) -> Nil
    _ -> panic as "set_row should report row bounds separately"
  }
  case matrix.set_column(a, 2, first) {
    Error(error.OutOfBounds(0, 2)) -> Nil
    _ -> panic as "set_column should report column bounds separately"
  }
  case matrix.set_column(a, 1, vector.from_list([1.0])) {
    Error(error.DimensionMismatch(expected: "2", actual: "1")) -> Nil
    _ -> panic as "set_column should report vector length mismatch"
  }
}

pub fn sparse_matrix_canonicalizes_and_multiplies_test() {
  let assert Ok(a) =
    sparse.from_entries(rows: 3, cols: 3, entries: [
      sparse.Entry(row: 2, col: 0, value: 4.0),
      sparse.Entry(row: 0, col: 1, value: 2.0),
      sparse.Entry(row: 0, col: 1, value: 3.0),
      sparse.Entry(row: 1, col: 2, value: -1.0),
      sparse.Entry(row: 1, col: 2, value: 1.0),
      sparse.Entry(row: 2, col: 2, value: 0.0),
    ])
  let x = vector.from_list([1.0, 2.0, 3.0])

  let assert Ok(y) = sparse.mul_vec(a, x)
  let assert Ok(aty) = sparse.transpose_mul_vec(a, x)
  let assert Ok(stored) = sparse.get(a, 0, 1)
  let assert Ok(implicit_zero) = sparse.get(a, 1, 2)

  assert sparse.rows(a) == 3
  assert sparse.cols(a) == 3
  assert sparse.nnz(a) == 2
  assert sparse.row_offsets(a) == [0, 1, 1, 2]
  assert sparse.column_indices(a) == [1, 0]
  assert sparse.values(a) == [5.0, 4.0]
  assert matrix.to_rows(sparse.to_dense(a))
    == [
      [0.0, 5.0, 0.0],
      [0.0, 0.0, 0.0],
      [4.0, 0.0, 0.0],
    ]
  assert vector.approx_equal(y, vector.from_list([10.0, 0.0, 4.0]), tolerance)
  assert vector.approx_equal(aty, vector.from_list([12.0, 5.0, 0.0]), tolerance)
  assert close(stored, 5.0)
  assert close(implicit_zero, 0.0)
}

pub fn sparse_matrix_dense_conversion_and_errors_test() {
  let assert Ok(dense) =
    matrix.from_rows([[1.0, 1.0e-14], [0.0, -2.0], [3.0, 0.0]])
  let assert Ok(a) = sparse.from_dense(dense, drop_tolerance: 1.0e-12)
  let assert Ok(combined_small) =
    sparse.from_entries_with_tolerance(
      rows: 1,
      cols: 1,
      entries: [
        sparse.Entry(row: 0, col: 0, value: 0.6e-12),
        sparse.Entry(row: 0, col: 0, value: 0.6e-12),
      ],
      drop_tolerance: 1.0e-12,
    )
  let transposed = sparse.transpose(a)
  let scaled = sparse.scale(a, 0.5)

  assert sparse.nnz(a) == 3
  assert sparse.values(combined_small) == [1.2e-12]
  assert matrix.to_rows(sparse.to_dense(a))
    == [[1.0, 0.0], [0.0, -2.0], [3.0, 0.0]]
  assert matrix.to_rows(sparse.to_dense(transposed))
    == [
      [1.0, 0.0, 3.0],
      [0.0, -2.0, 0.0],
    ]
  assert matrix.to_rows(sparse.to_dense(scaled))
    == [
      [0.5, 0.0],
      [0.0, -1.0],
      [1.5, 0.0],
    ]
  assert close_to(sparse.norm_inf(a), 3.0, tolerance)

  case sparse.get(a, 4, 0) {
    Error(error.OutOfBounds(4, 0)) -> Nil
    _ -> panic as "sparse.get should reject out-of-bounds coordinates"
  }
  case sparse.mul_vec(a, vector.from_list([1.0])) {
    Error(error.DimensionMismatch(expected: "2", actual: "1")) -> Nil
    _ -> panic as "sparse.mul_vec should reject incompatible vector length"
  }
  case
    sparse.from_entries(rows: 2, cols: 2, entries: [
      sparse.Entry(row: 0, col: 2, value: 1.0),
    ])
  {
    Error(error.OutOfBounds(0, 2)) -> Nil
    _ -> panic as "sparse.from_entries should reject invalid coordinates"
  }
}

pub fn complex_number_and_vector_operations_test() {
  let z = complex.new(real: 3.0, imaginary: 4.0)
  let w = complex.new(real: 1.0, imaginary: -2.0)
  let assert Ok(magnitude) = complex.abs(z)
  let product = complex.mul(z, w)
  let assert Ok(quotient) = complex.div(product, w)
  let v = complex.vector_from_list([z, w])
  let assert Ok(norm) = complex.vector_norm2(v)
  let assert Ok(unit) = complex.vector_normalize(v)
  let assert Ok(unit_norm) = complex.vector_norm2(unit)
  let assert Ok(axpy) =
    complex.vector_axpy(complex.new(real: 0.0, imaginary: 1.0), v, v)

  assert close_to(magnitude, 5.0, tolerance)
  assert complex.approx_equal(quotient, z, tolerance)
  assert close_to(norm, 5.477225575051661, 1.0e-12)
  assert close_to(unit_norm, 1.0, 1.0e-12)
  assert complex.vector_dimension(axpy) == 2
}

pub fn lu_solve_with_partial_pivoting_test() {
  let assert Ok(a) = matrix.from_rows([[0.0, 2.0], [1.0, 1.0]])
  let b = vector.from_list([4.0, 3.0])

  let assert Ok(x) = direct.solve(a, b)

  assert vector.approx_equal(x, vector.from_list([1.0, 2.0]), tolerance)
}

pub fn lu_reconstructs_permuted_matrix_test() {
  let assert Ok(a) =
    matrix.from_rows([[0.0, 2.0, 1.0], [2.0, 1.0, 3.0], [1.0, 1.0, 1.0]])

  let assert Ok(factors) = direct.lu_factor(a)
  let assert Ok(pa) = matrix.mul(factors.p, a)
  let assert Ok(lu) = matrix.mul(factors.l, factors.u)

  assert matrix.approx_equal(pa, lu, tolerance)
}

pub fn complete_lu_reconstructs_permuted_matrix_and_solves_test() {
  let assert Ok(a) = matrix.from_rows([[0.0, 2.0], [1.0, 3.0]])
  let b = vector.from_list([4.0, 7.0])
  let expected = vector.from_list([1.0, 2.0])

  let assert Ok(factors) = direct.complete_lu_factor(a)
  let assert Ok(pa) = matrix.mul(factors.p, a)
  let assert Ok(paq) = matrix.mul(pa, factors.q)
  let assert Ok(lu) = matrix.mul(factors.l, factors.u)
  let assert Ok(from_factors) = direct.complete_lu_solve(factors, b)
  let assert Ok(from_matrix) = direct.solve_complete_pivoting(a, b)
  let assert Ok(determinant) = direct.determinant_complete_pivoting(a)
  let assert Ok(inverse) = direct.inverse_complete_pivoting(a)
  let assert Ok(a_inverse) = matrix.mul(a, inverse)
  let assert Ok(identity) = matrix.identity(2)

  assert factors.swaps > 0
  assert matrix.approx_equal(paq, lu, tolerance)
  assert vector.approx_equal(from_factors, expected, tolerance)
  assert vector.approx_equal(from_matrix, expected, tolerance)
  assert close_to(determinant, -2.0, 1.0e-8)
  assert matrix.approx_equal(a_inverse, identity, 1.0e-8)
}

pub fn cholesky_factor_and_solve_spd_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 2.0], [2.0, 3.0]])
  let b = vector.from_list([6.0, 5.0])

  let assert Ok(factors) = direct.cholesky_factor(a)
  let assert Ok(reconstructed) =
    matrix.mul(factors.l, matrix.transpose(factors.l))
  let assert Ok(x) = direct.solve_spd(a, b)

  assert matrix.approx_equal(reconstructed, a, tolerance)
  assert vector.approx_equal(x, vector.from_list([1.0, 1.0]), tolerance)
}

pub fn householder_maps_vector_to_positive_axis_test() {
  let x = vector.from_list([2.0, 1.0])
  let assert Ok(#(h, a)) = orthogonal.householder_matrix(x)
  let assert Ok(y) = matrix.mul_vec(h, x)
  let assert Ok(expected_a) = float.square_root(5.0)

  assert close(a, expected_a)
  assert vector.approx_equal(y, vector.from_list([expected_a, 0.0]), tolerance)
}

pub fn givens_rotation_zeroes_second_component_test() {
  let assert Ok(rotation) = orthogonal.givens(3.0, 4.0)
  let assert Ok(g) = orthogonal.givens_matrix(2, 0, 1, rotation)
  let assert Ok(y) = matrix.mul_vec(g, vector.from_list([3.0, 4.0]))

  assert vector.approx_equal(y, vector.from_list([5.0, 0.0]), tolerance)
}

pub fn householder_qr_reconstructs_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

  let assert Ok(qr) = orthogonal.householder_qr(a)
  let assert Ok(reconstructed) = matrix.mul(qr.q, qr.r)
  let assert Ok(qtq) = matrix.mul(matrix.transpose(qr.q), qr.q)
  let assert Ok(identity) = matrix.identity(3)

  assert qr.form == orthogonal.FullQR
  assert matrix.approx_equal(reconstructed, a, tolerance)
  assert matrix.approx_equal(qtq, identity, tolerance)
}

pub fn givens_qr_reconstructs_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

  let assert Ok(qr) = orthogonal.givens_qr(a)
  let assert Ok(reconstructed) = matrix.mul(qr.q, qr.r)
  let assert Ok(qtq) = matrix.mul(matrix.transpose(qr.q), qr.q)
  let assert Ok(identity) = matrix.identity(3)

  assert qr.form == orthogonal.FullQR
  assert matrix.approx_equal(reconstructed, a, tolerance)
  assert matrix.approx_equal(qtq, identity, tolerance)
}

pub fn gram_schmidt_qr_reconstructs_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

  let assert Ok(classical) = orthogonal.classical_gram_schmidt_qr(a)
  let assert Ok(modified) = orthogonal.modified_gram_schmidt_qr(a)
  let assert Ok(classical_reconstructed) = matrix.mul(classical.q, classical.r)
  let assert Ok(modified_reconstructed) = matrix.mul(modified.q, modified.r)
  let assert Ok(classical_qtq) =
    matrix.mul(matrix.transpose(classical.q), classical.q)
  let assert Ok(modified_qtq) =
    matrix.mul(matrix.transpose(modified.q), modified.q)
  let assert Ok(identity) = matrix.identity(2)

  assert classical.form == orthogonal.ThinQR
  assert modified.form == orthogonal.ThinQR
  assert matrix.approx_equal(classical_reconstructed, a, tolerance)
  assert matrix.approx_equal(modified_reconstructed, a, tolerance)
  assert matrix.approx_equal(classical_qtq, identity, tolerance)
  assert matrix.approx_equal(modified_qtq, identity, tolerance)
}

pub fn error_analysis_residual_and_condition_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 3.0]])
  let x = vector.from_list([0.2, 0.6])
  let b = vector.from_list([1.0, 2.0])
  let rough_x = vector.from_list([0.19, 0.62])

  let assert Ok(residual_norm) = error_analysis.residual_norm2(a, x, b)
  let assert Ok(condition) = error_analysis.condition_number_inf(a)
  let assert Ok(refined) =
    error_analysis.iterative_refinement(a, b, rough_x, 3, 1.0e-10)
  let assert [rough_residual, refined_residual] = refined.residual_history

  assert residual_norm <=. tolerance
  assert close(condition, 3.2)
  assert refined.converged
  assert refined.iterations == 1
  assert refined_residual <. rough_residual
  assert refined.residual_norm <=. 1.0e-10
  assert vector.approx_equal(refined.solution, x, 1.0e-8)
}

pub fn ill_conditioned_forward_bound_uses_infinity_norm_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 1.000001]])
  let exact = vector.from_list([1.0, -1.0])
  let computed = vector.from_list([1.000001, -1.0])
  let assert Ok(b) = matrix.mul_vec(a, exact)

  let assert Ok(forward) = error_analysis.forward_error_inf(exact, computed)
  let assert Ok(relative_inf) =
    error_analysis.normwise_relative_residual_inf(a, computed, b)
  let assert Ok(kappa) = error_analysis.condition_number_inf(a)
  let assert Ok(bound) =
    error_analysis.residual_forward_bound_inf(a, computed, b)

  assert relative_inf >. 0.0
  assert close_to(bound, kappa *. relative_inf, 1.0e-6)
  assert bound >=. forward

  let assert Ok(identity) = matrix.identity(2)
  let zero = vector.from_list([0.0, 0.0])
  case error_analysis.normwise_relative_residual_inf(identity, zero, zero) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "infinity-norm relative residual should reject zero b"
  }
}

pub fn jacobi_gauss_seidel_and_sor_converge_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 1.0], [2.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(initial) = vector.zeros(2)
  let expected = vector.from_list([0.1, 0.6])

  let assert Ok(jacobi) = iterative.jacobi(a, b, initial, 80, 1.0e-10)
  let assert Ok(gs) = iterative.gauss_seidel(a, b, initial, 80, 1.0e-10)
  let assert Ok(sor) = iterative.sor(a, b, initial, 1.1, 80, 1.0e-10)
  let assert Ok(jacobi_diagnostics) =
    iterative.jacobi_convergence_diagnostics(a)
  let assert Ok(gs_diagnostics) =
    iterative.gauss_seidel_convergence_diagnostics(a)
  let assert Ok(sor_diagnostics) = iterative.sor_convergence_diagnostics(a, 1.1)

  assert jacobi.converged
  assert gs.converged
  assert sor.converged
  assert jacobi_diagnostics.sufficient_convergence
  assert gs_diagnostics.sufficient_convergence
  assert sor_diagnostics.sufficient_convergence
  assert vector.approx_equal(jacobi.solution, expected, 1.0e-6)
  assert vector.approx_equal(gs.solution, expected, 1.0e-6)
  assert vector.approx_equal(sor.solution, expected, 1.0e-6)
  assert close_to(jacobi_diagnostics.infinity_norm_bound, 2.0 /. 3.0, 1.0e-8)
  assert close_to(
    matrix.unsafe_get(jacobi_diagnostics.iteration_matrix, 0, 1),
    -0.25,
    1.0e-8,
  )
  assert close_to(
    matrix.unsafe_get(jacobi_diagnostics.iteration_matrix, 1, 0),
    -2.0 /. 3.0,
    1.0e-8,
  )
}

pub fn conjugate_gradient_family_converges_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 1.0], [1.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(initial) = vector.zeros(2)
  let expected = vector.from_list([0.09090909090909091, 0.6363636363636364])

  let assert Ok(sd) = iterative.steepest_descent(a, b, initial, 80, 1.0e-6)
  let assert Ok(cg) = iterative.conjugate_gradient(a, b, initial, 10, 1.0e-10)
  let assert Ok(practical_cg) =
    iterative.practical_conjugate_gradient(a, b, initial, 10, 1.0e-10, 2)
  let assert Ok(identity_pcg) =
    iterative.preconditioned_conjugate_gradient_with(
      a,
      b,
      initial,
      10,
      1.0e-10,
      fn(r) { Ok(r) },
    )
  let assert Ok(pcg) =
    iterative.preconditioned_conjugate_gradient(a, b, initial, 10, 1.0e-10)

  assert sd.converged
  assert cg.converged
  assert practical_cg.converged
  assert identity_pcg.converged
  assert pcg.converged
  assert vector.approx_equal(sd.solution, expected, 1.0e-5)
  assert vector.approx_equal(cg.solution, expected, 1.0e-8)
  assert vector.approx_equal(practical_cg.solution, expected, 1.0e-8)
  assert vector.approx_equal(identity_pcg.solution, expected, 1.0e-8)
  assert vector.approx_equal(pcg.solution, expected, 1.0e-8)
}

pub fn iterative_and_eigen_non_convergence_is_structured_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(initial) = vector.zeros(2)
  let eigen_initial = vector.from_list([1.0, 2.0])

  let assert Ok(jacobi) = iterative.jacobi(a, b, initial, 0, 1.0e-12)
  let assert Ok(gauss_seidel) =
    iterative.gauss_seidel(a, b, initial, 0, 1.0e-12)
  let assert Ok(sor) = iterative.sor(a, b, initial, 1.1, 0, 1.0e-12)
  let assert Ok(steepest) =
    iterative.steepest_descent(a, b, initial, 0, 1.0e-12)
  let assert Ok(cg) = iterative.conjugate_gradient(a, b, initial, 0, 1.0e-12)
  let assert Ok(practical_cg) =
    iterative.practical_conjugate_gradient(a, b, initial, 0, 1.0e-12, 2)
  let assert Ok(pcg) =
    iterative.preconditioned_conjugate_gradient(a, b, initial, 0, 1.0e-12)

  let assert Ok(gmres) = krylov.gmres(a, b, initial, 1, 1.0e-12)
  let assert Ok(restarted_gmres) =
    krylov.restarted_gmres(a, b, initial, 1, 1, 1.0e-12)

  let assert Ok(power) = eigen.power_method(a, eigen_initial, 0, 1.0e-12)
  let assert Ok(inverse_power) =
    eigen.inverse_power_method(a, eigen_initial, 0.8, 0, 1.0e-12)
  let assert Ok(qr) = eigen.qr_iteration(a, 0, 1.0e-12)
  let assert Ok(shifted_qr) = eigen.shifted_qr_iteration(a, 0, 1.0e-12)
  let assert Ok(qr_history) = eigen.qr_convergence_history(a, 0, 1.0e-12)
  let assert Ok(shifted_qr_history) =
    eigen.shifted_qr_convergence_history(a, 0, 1.0e-12)
  let assert Ok(symmetric_qr_history) =
    eigen.symmetric_qr_convergence_history(a, 0, 1.0e-12)
  let assert Ok(symmetric_qr) = eigen.symmetric_qr(a, 0, 1.0e-12)
  let assert Ok(implicit_qr) = eigen.implicit_qr_iteration(a, 0, 1.0e-12)
  let assert Ok(double_shift_qr) =
    eigen.double_shift_qr_iteration(a, 0, 1.0e-12)
  let assert Ok(symmetric_qr_eigen) = eigen.symmetric_qr_eigen(a, 0, 1.0e-12)
  let assert Ok(jacobi_eigen) = eigen.jacobi_eigen(a, 0, 1.0e-12)

  assert jacobi.converged == False
  assert jacobi.iterations == 0
  assert gauss_seidel.converged == False
  assert gauss_seidel.iterations == 0
  assert sor.converged == False
  assert sor.iterations == 0
  assert steepest.converged == False
  assert steepest.iterations == 0
  assert cg.converged == False
  assert cg.iterations == 0
  assert practical_cg.converged == False
  assert practical_cg.iterations == 0
  assert pcg.converged == False
  assert pcg.iterations == 0
  assert gmres.converged == False
  assert restarted_gmres.converged == False
  assert power.converged == False
  assert power.iterations == 0
  assert inverse_power.converged == False
  assert inverse_power.iterations == 0
  assert qr.converged == False
  assert qr.iterations == 0
  assert shifted_qr.converged == False
  assert shifted_qr.iterations == 0
  assert qr_history.result.converged == False
  assert shifted_qr_history.result.converged == False
  assert symmetric_qr_history.result.converged == False
  assert symmetric_qr.converged == False
  assert symmetric_qr.iterations == 0
  assert implicit_qr.converged == False
  assert implicit_qr.iterations == 0
  assert double_shift_qr.converged == False
  assert double_shift_qr.iterations == 0
  assert symmetric_qr_eigen.converged == False
  assert symmetric_qr_eigen.iterations == 0
  assert jacobi_eigen.converged == False
  assert jacobi_eigen.iterations == 0

  let assert Ok(general) =
    matrix.from_rows([[2.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 2.0]])
  let assert Ok(real_schur) = eigen.real_schur_basic(general, 0, 1.0e-12)
  assert real_schur.converged == False
  case eigen.real_schur_eigenvalues_of(general, 0, 1.0e-12) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "Schur eigenvalue extraction should reject non-convergence"
  }
  case eigen.complex_eigenpairs_of(general, 0, 1.0e-12) {
    Error(error.NoConvergence(iterations: 0, residual: residual)) -> {
      assert residual >. 0.0
    }
    _ -> panic as "complex eigenpair extraction should reject non-convergence"
  }
}

pub fn least_squares_normal_and_qr_agree_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
  let b = vector.from_list([1.0, 2.0, 2.0])
  let expected = vector.from_list([1.1666666666666667, 0.5])

  let assert Ok(default_qr) = least_squares.solve(a, b)
  let assert Ok(normal) = least_squares.normal_equations(a, b)
  let assert Ok(qr) = least_squares.householder_qr(a, b)
  let assert Ok(givens) = least_squares.givens_qr(a, b)
  let assert Ok(cgs) = least_squares.classical_gram_schmidt_qr(a, b)
  let assert Ok(mgs) = least_squares.modified_gram_schmidt_qr(a, b)
  let assert Ok(svd_solution) = least_squares.svd(a, b)
  let assert Ok(diagnostics) =
    least_squares.stability_diagnostics(a, b, qr.solution)

  assert vector.approx_equal(default_qr.solution, expected, 1.0e-8)
  assert vector.approx_equal(normal.solution, expected, 1.0e-8)
  assert vector.approx_equal(qr.solution, expected, 1.0e-8)
  assert vector.approx_equal(givens.solution, expected, 1.0e-8)
  assert vector.approx_equal(cgs.solution, expected, 1.0e-8)
  assert vector.approx_equal(mgs.solution, expected, 1.0e-8)
  assert vector.approx_equal(svd_solution.solution, expected, 1.0e-8)
  assert close(normal.residual_norm, qr.residual_norm)
  assert close(normal.residual_norm, givens.residual_norm)
  assert close(normal.residual_norm, cgs.residual_norm)
  assert close(normal.residual_norm, mgs.residual_norm)
  assert close(normal.residual_norm, svd_solution.residual_norm)
  assert close_to(diagnostics.residual_norm, qr.residual_norm, 1.0e-8)
  assert diagnostics.relative_residual >. 0.0
  assert diagnostics.normal_matrix_condition_inf >. 0.0
  assert diagnostics.normal_equation_residual_norm <=. 1.0e-8
}

pub fn svd_reconstructs_matrix_and_spectrum_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
  let assert Ok(expected_largest) = float.square_root(3.0)

  let assert Ok(result) = svd.decompose(a)
  let sigma = diagonal_matrix_from_vector(result.singular_values)
  let assert Ok(us) = matrix.mul(result.u, sigma)
  let assert Ok(reconstructed) = matrix.mul(us, result.vt)
  let assert Ok(utu) = matrix.mul(matrix.transpose(result.u), result.u)
  let assert Ok(vvt) = matrix.mul(result.vt, matrix.transpose(result.vt))
  let assert Ok(identity) = matrix.identity(2)
  let assert [largest, smallest] = vector.to_list(result.singular_values)
  let assert Ok(rank) = svd.rank(result, 1.0e-12)
  let assert Ok(condition) = svd.condition_number(result, 1.0e-12)
  let assert Ok(norm2) = svd.norm2(a)

  assert result.converged
  assert matrix.approx_equal(reconstructed, a, 1.0e-8)
  assert matrix.approx_equal(utu, identity, 1.0e-8)
  assert matrix.approx_equal(vvt, identity, 1.0e-8)
  assert close_to(largest, expected_largest, 1.0e-8)
  assert close_to(smallest, 1.0, 1.0e-8)
  assert rank == 2
  assert close_to(condition, expected_largest, 1.0e-8)
  assert close_to(norm2, expected_largest, 1.0e-8)
}

pub fn svd_pseudoinverse_handles_rank_deficiency_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
  let b = vector.from_list([1.0, 2.0, 3.0])
  let expected_minimum_norm = vector.from_list([0.2, 0.4])

  let assert Ok(solution) = svd.solve(a, b)
  let assert Ok(ls_solution) = least_squares.svd(a, b)
  let assert Ok(rank) = svd.numerical_rank(a, 1.0e-10)
  let assert Ok(a_plus) = svd.pseudoinverse(a)
  let assert Ok(projected) = matrix.mul_vec(a_plus, b)

  assert rank == 1
  assert vector.approx_equal(solution, expected_minimum_norm, 1.0e-8)
  assert vector.approx_equal(
    ls_solution.solution,
    expected_minimum_norm,
    1.0e-8,
  )
  assert vector.approx_equal(projected, expected_minimum_norm, 1.0e-8)
  assert ls_solution.residual_norm <=. 1.0e-8
  case svd.condition_number_2(a, 1.0e-10) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "rank-deficient SVD condition number should be rejected"
  }
}

pub fn svd_derived_operations_reject_non_convergence_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [0.0, 1.0]])
  let assert Ok(result) = svd.decompose_with(a, 0, 1.0e-16)

  assert result.converged == False
  assert result.off_diagonal_norm >. 0.0
  case svd.pseudoinverse_from(result, 1.0e-12) {
    Error(error.NoConvergence(iterations: 0, residual: residual)) -> {
      assert residual >. 0.0
    }
    _ -> panic as "derived SVD operations should reject non-convergence"
  }
}

pub fn rank_deficient_and_extreme_values_are_handled_test() {
  let assert Ok(rank_deficient) =
    matrix.from_rows([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
  let b = vector.from_list([1.0, 2.0, 3.0])

  case least_squares.normal_equations(rank_deficient, b) {
    Error(_) -> Nil
    _ -> panic as "normal equations should reject rank-deficient systems"
  }
  case least_squares.householder_qr(rank_deficient, b) {
    Error(_) -> Nil
    _ -> panic as "Householder QR should reject rank-deficient systems"
  }
  case least_squares.givens_qr(rank_deficient, b) {
    Error(_) -> Nil
    _ -> panic as "Givens QR should reject rank-deficient systems"
  }

  let huge = vector.from_list([1.0e150, -1.0e150])
  let scaled = vector.scale(huge, 1.0e-150)
  let assert Ok(diagonal) = matrix.diagonal([1.0e150, 1.0e-150])
  let x = vector.from_list([1.0e-150, 1.0e150])
  let assert Ok(y) = matrix.mul_vec(diagonal, x)

  assert close_to(vector.norm_inf(huge), 1.0e150, 1.0e136)
  assert vector.approx_equal(scaled, vector.from_list([1.0, -1.0]), 1.0e-12)
  assert vector.approx_equal(y, vector.from_list([1.0, 1.0]), 1.0e-12)
}

pub fn power_and_inverse_power_methods_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 0.0], [0.0, 1.0]])
  let initial = vector.from_list([1.0, 1.0])

  let assert Ok(dominant) = eigen.power_method(a, initial, 80, 1.0e-10)
  let assert Ok(near_one) =
    eigen.inverse_power_method(a, initial, 0.8, 80, 1.0e-10)

  assert dominant.converged
  assert near_one.converged
  assert close(dominant.value, 2.0)
  assert close(near_one.value, 1.0)
}

pub fn qr_iteration_and_hessenberg_reduction_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])
  let assert Ok(qr) = eigen.qr_iteration(a, 80, 1.0e-6)
  let assert Ok(history) = eigen.symmetric_qr_convergence_history(a, 20, 1.0e-8)
  let assert [first_step, ..] = history.steps
  let final_step = last_qr_step(history.steps)

  assert qr.converged
  assert close(matrix.unsafe_get(qr.t, 0, 0), 3.0)
  assert close(matrix.unsafe_get(qr.t, 1, 1), 1.0)
  assert history.result.converged
  assert first_step.iteration == 0
  assert first_step.off_diagonal_norm >. final_step.off_diagonal_norm
  assert final_step.off_diagonal_norm <=. 1.0e-8
  assert close_to(matrix.unsafe_get(history.result.t, 0, 0), 3.0, 1.0e-6)
  assert close_to(matrix.unsafe_get(history.result.t, 1, 1), 1.0, 1.0e-6)

  let assert Ok(general) =
    matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
  let assert Ok(reduced) = eigen.hessenberg_reduction(general)

  assert close(matrix.unsafe_get(reduced.h, 2, 0), 0.0)
}

pub fn shifted_qr_variants_converge_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])

  let assert Ok(symmetric) = eigen.symmetric_qr(a, 20, 1.0e-8)
  let assert Ok(implicit) = eigen.implicit_qr_iteration(a, 80, 1.0e-6)
  let assert Ok(double_shift) = eigen.double_shift_qr_iteration(a, 80, 1.0e-6)

  assert symmetric.converged
  assert implicit.converged
  assert double_shift.converged
  assert close_to(matrix.unsafe_get(symmetric.t, 0, 0), 3.0, 1.0e-6)
  assert close_to(matrix.unsafe_get(implicit.t, 0, 0), 3.0, 1.0e-6)
  assert close_to(matrix.unsafe_get(double_shift.t, 0, 0), 3.0, 1.0e-6)
  assert close_to(matrix.unsafe_get(symmetric.t, 1, 1), 1.0, 1.0e-6)
  assert close_to(matrix.unsafe_get(implicit.t, 1, 1), 1.0, 1.0e-6)
  assert close_to(matrix.unsafe_get(double_shift.t, 1, 1), 1.0, 1.0e-6)
}

pub fn symmetric_qr_rejects_nonsymmetric_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 2.0], [0.0, 1.0]])

  case eigen.symmetric_qr(a, 20, 1.0e-8) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "symmetric_qr should reject nonsymmetric matrices"
  }
}

pub fn symmetric_qr_eigen_decomposes_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 2.0]])

  let assert Ok(result) = eigen.symmetric_qr_eigen(a, 20, 1.0e-8)
  let diagonal = diagonal_matrix_from_vector(result.diagonal)
  let assert Ok(av) = matrix.mul(a, result.eigenvectors)
  let assert Ok(vd) = matrix.mul(result.eigenvectors, diagonal)
  let assert Ok(qtq) =
    matrix.mul(matrix.transpose(result.eigenvectors), result.eigenvectors)
  let assert Ok(identity) = matrix.identity(2)

  assert result.converged
  assert matrix.approx_equal(av, vd, 1.0e-6)
  assert matrix.approx_equal(qtq, identity, 1.0e-6)
}

pub fn real_schur_blocks_detect_complex_pair_test() {
  let assert Ok(rotation) = matrix.from_rows([[0.0, -1.0], [1.0, 0.0]])
  let assert Ok(schur) = eigen.real_schur_basic(rotation, 10, 1.0e-8)
  let assert Ok(blocks) = eigen.real_schur_blocks(schur.t, 1.0e-8)
  let assert Ok(values) = eigen.real_schur_eigenvalues(schur.t, 1.0e-8)
  let assert Ok(values_from_matrix) =
    eigen.real_schur_eigenvalues_of(rotation, 10, 1.0e-8)
  let assert Ok(pairs_from_schur) =
    eigen.real_schur_complex_eigenpairs(schur.q, schur.t, 1.0e-8)
  let assert Ok(pairs_from_matrix) =
    eigen.complex_eigenpairs_of(rotation, 10, 1.0e-8)

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
      assert close_to(real_part, 0.0, 1.0e-8)
      assert close_to(imaginary, 1.0, 1.0e-8)
      assert close_to(trace, 0.0, 1.0e-8)
      assert close_to(determinant, 1.0, 1.0e-8)
    }
    _ -> panic as "expected one complex conjugate Schur block"
  }
  assert_rotation_eigenvalues(values)
  assert_rotation_eigenvalues(values_from_matrix)
  assert_rotation_complex_eigenpairs(pairs_from_schur)
  assert_rotation_complex_eigenpairs(pairs_from_matrix)
}

pub fn generalized_eigenvalue_routines_reduce_regular_pencils_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 0.0], [0.0, 3.0]])
  let assert Ok(b) = matrix.from_rows([[1.0, 0.0], [0.0, 2.0]])
  let assert Ok(standard) = eigen.generalized_standard_matrix(a, b)
  let assert Ok(values) = eigen.generalized_eigenvalues(a, b, 20, 1.0e-10)

  assert matrix.approx_equal(
    standard,
    diagonal_matrix_from_vector(vector.from_list([2.0, 1.5])),
    1.0e-10,
  )
  assert has_real_eigenvalue(values, 2.0)
  assert has_real_eigenvalue(values, 1.5)

  let assert Ok(rotation_a) = matrix.from_rows([[0.0, -2.0], [2.0, 0.0]])
  let assert Ok(rotation_b) = matrix.from_rows([[2.0, 0.0], [0.0, 2.0]])
  let assert Ok(rotation_values) =
    eigen.generalized_eigenvalues(rotation_a, rotation_b, 20, 1.0e-10)
  let assert Ok(rotation_pairs) =
    eigen.generalized_complex_eigenpairs(rotation_a, rotation_b, 20, 1.0e-10)

  assert_rotation_eigenvalues(rotation_values)
  assert_rotation_complex_eigenpairs(rotation_pairs)
}

pub fn symmetric_tridiagonal_and_jacobi_eigen_test() {
  let assert Ok(a) =
    matrix.from_rows([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 2.0]])

  let assert Ok(reduced) = eigen.symmetric_tridiagonal_reduction(a)
  let assert Ok(qt) = matrix.mul(reduced.q, reduced.t)
  let assert Ok(reconstructed) = matrix.mul(qt, matrix.transpose(reduced.q))

  assert matrix.approx_equal(reconstructed, a, 1.0e-8)
  assert close_to(matrix.unsafe_get(reduced.t, 2, 0), 0.0, 1.0e-8)
  assert close_to(matrix.unsafe_get(reduced.t, 0, 2), 0.0, 1.0e-8)

  let assert Ok(jacobi) = eigen.jacobi_eigen(a, 80, 1.0e-8)
  let diagonal = diagonal_matrix_from_vector(jacobi.diagonal)
  let assert Ok(av) = matrix.mul(a, jacobi.eigenvectors)
  let assert Ok(vd) = matrix.mul(jacobi.eigenvectors, diagonal)

  assert jacobi.converged
  assert matrix.approx_equal(av, vd, 1.0e-6)
}

pub fn arnoldi_builds_krylov_relation_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 0.0], [0.0, 1.0]])
  let initial = vector.from_list([1.0, 1.0])

  let assert Ok(result) = krylov.arnoldi(a, initial, 1, 1.0e-12)
  let assert Ok(q0) = matrix.column(result.q, 0)
  let assert Ok(lhs) = matrix.mul_vec(a, q0)
  let assert Ok(h0) = matrix.column(result.h, 0)
  let assert Ok(rhs) = matrix.mul_vec(result.q, h0)

  assert vector.approx_equal(lhs, rhs, tolerance)
}

pub fn lanczos_builds_symmetric_krylov_relation_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 0.0], [0.0, 1.0]])
  let initial = vector.from_list([1.0, 1.0])

  let assert Ok(result) = krylov.lanczos(a, initial, 2, 1.0e-12)
  let assert Ok(aq) = matrix.mul(a, result.q)
  let assert Ok(qt) = matrix.mul(result.q, result.t)
  let assert Ok(qtq) = matrix.mul(matrix.transpose(result.q), result.q)
  let assert Ok(identity) = matrix.identity(result.steps)

  assert result.steps == 2
  assert result.happy_breakdown
  assert matrix.approx_equal(aq, qt, 1.0e-8)
  assert matrix.approx_equal(qtq, identity, 1.0e-8)
  assert close_to(
    matrix.unsafe_get(result.t, 1, 0),
    matrix.unsafe_get(result.t, 0, 1),
    1.0e-8,
  )
}

pub fn gmres_solves_nonsymmetric_system_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 1.0], [2.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(initial) = vector.zeros(2)
  let expected = vector.from_list([0.1, 0.6])

  let assert Ok(gmres) = krylov.gmres(a, b, initial, 2, 1.0e-10)
  let assert Ok(restarted) =
    krylov.restarted_gmres(a, b, initial, 2, 4, 1.0e-10)

  assert gmres.converged
  assert restarted.converged
  assert gmres.iterations <= 2
  assert restarted.iterations <= 4
  assert gmres.residual_norm <=. 1.0e-8
  assert restarted.residual_norm <=. 1.0e-8
  assert vector.approx_equal(gmres.solution, expected, 1.0e-8)
  assert vector.approx_equal(restarted.solution, expected, 1.0e-8)
}

pub fn bicg_family_solves_nonsymmetric_system_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 1.0], [2.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(initial) = vector.zeros(2)
  let shadow = vector.from_list([1.0, 2.0])
  let expected = vector.from_list([0.1, 0.6])

  let assert Ok(bicg) = krylov.bicg(a, b, initial, 4, 1.0e-10)
  let assert Ok(bicg_shadow) =
    krylov.bicg_with_shadow(a, b, initial, shadow, 4, 1.0e-10)
  let assert Ok(bicgstab) = krylov.bicgstab(a, b, initial, 4, 1.0e-10)

  assert bicg.converged
  assert bicg_shadow.converged
  assert bicgstab.converged
  assert bicg.residual_norm <=. 1.0e-8
  assert bicg_shadow.residual_norm <=. 1.0e-8
  assert bicgstab.residual_norm <=. 1.0e-8
  assert vector.approx_equal(bicg.solution, expected, 1.0e-8)
  assert vector.approx_equal(bicg_shadow.solution, expected, 1.0e-8)
  assert vector.approx_equal(bicgstab.solution, expected, 1.0e-8)
}

pub fn minres_solves_symmetric_indefinite_system_test() {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, -1.0]])
  let b = vector.from_list([1.0, 0.0])
  let assert Ok(initial) = vector.zeros(2)
  let expected = vector.from_list([0.3333333333333333, 0.3333333333333333])

  let assert Ok(minres) = krylov.minres(a, b, initial, 4, 1.0e-10)

  assert minres.converged
  assert minres.iterations <= 2
  assert minres.residual_norm <=. 1.0e-8
  assert vector.approx_equal(minres.solution, expected, 1.0e-8)
}

fn close(a: Float, b: Float) -> Bool {
  float.absolute_value(a -. b) <=. tolerance
}

fn close_to(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

fn diagonal_matrix_from_vector(values: vector.Vector) -> matrix.Matrix {
  let assert Ok(result) =
    matrix.from_fn(
      rows: vector.dimension(values),
      cols: vector.dimension(values),
      with: fn(i, j) {
        case i == j {
          True -> unsafe_vector_get(values, i)
          False -> 0.0
        }
      },
    )
  result
}

fn last_qr_step(
  steps: List(eigen.QrConvergenceStep),
) -> eigen.QrConvergenceStep {
  case steps {
    [step] -> step
    [_, ..rest] -> last_qr_step(rest)
    [] ->
      eigen.QrConvergenceStep(iteration: 0, shift: 0.0, off_diagonal_norm: 0.0)
  }
}

fn assert_rotation_eigenvalues(values: List(eigen.Eigenvalue)) -> Nil {
  case values {
    [
      eigen.ComplexEigenvalue(real: real_pos, imaginary: imag_pos),
      eigen.ComplexEigenvalue(real: real_neg, imaginary: imag_neg),
    ] -> {
      assert close_to(real_pos, 0.0, 1.0e-8)
      assert close_to(real_neg, 0.0, 1.0e-8)
      assert close_to(imag_pos, 1.0, 1.0e-8)
      assert close_to(imag_neg, -1.0, 1.0e-8)
    }
    _ -> panic as "expected conjugate complex eigenvalues"
  }
}

fn has_real_eigenvalue(values: List(eigen.Eigenvalue), target: Float) -> Bool {
  case values {
    [] -> False
    [value, ..rest] ->
      case value {
        eigen.RealEigenvalue(value: actual) ->
          close_to(actual, target, 1.0e-8) || has_real_eigenvalue(rest, target)
        eigen.ComplexEigenvalue(real: _, imaginary: _) ->
          has_real_eigenvalue(rest, target)
      }
  }
}

fn assert_rotation_complex_eigenpairs(
  pairs: List(eigen.ComplexEigenpair),
) -> Nil {
  case pairs {
    [positive, negative] -> {
      assert complex.approx_equal(
        positive.value,
        complex.new(real: 0.0, imaginary: 1.0),
        1.0e-8,
      )
      assert complex.approx_equal(
        negative.value,
        complex.new(real: 0.0, imaginary: -1.0),
        1.0e-8,
      )
      assert positive.converged
      assert negative.converged
      assert positive.residual_norm <=. 1.0e-8
      assert negative.residual_norm <=. 1.0e-8
      assert complex.vector_dimension(positive.vector) == 2
      assert complex.vector_dimension(negative.vector) == 2

      let assert Ok(positive_norm) = complex.vector_norm2(positive.vector)
      let assert Ok(negative_norm) = complex.vector_norm2(negative.vector)
      assert close_to(positive_norm, 1.0, 1.0e-8)
      assert close_to(negative_norm, 1.0, 1.0e-8)
    }
    _ -> panic as "expected conjugate complex eigenpairs"
  }
}

fn unsafe_vector_get(values: vector.Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
}
