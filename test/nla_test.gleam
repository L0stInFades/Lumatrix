import gleam/float
import gleeunit
import nla/direct
import nla/eigen
import nla/error_analysis
import nla/iterative
import nla/krylov
import nla/least_squares
import nla/matrix
import nla/orthogonal
import nla/vector

const tolerance = 1.0e-8

pub fn main() -> Nil {
  gleeunit.main()
}

pub fn matrix_vector_product_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 2.0], [3.0, 4.0]])
  let x = vector.from_list([1.0, 1.0])

  let assert Ok(y) = matrix.mul_vec(a, x)

  assert vector.approx_equal(y, vector.from_list([3.0, 7.0]), tolerance)
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

  let assert Ok(qr) = orthogonal.qr_householder(a)
  let assert Ok(reconstructed) = matrix.mul(qr.q, qr.r)
  let assert Ok(qtq) = matrix.mul(matrix.transpose(qr.q), qr.q)
  let assert Ok(identity) = matrix.identity(3)

  assert matrix.approx_equal(reconstructed, a, tolerance)
  assert matrix.approx_equal(qtq, identity, tolerance)
}

pub fn givens_qr_reconstructs_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

  let assert Ok(qr) = orthogonal.qr_givens(a)
  let assert Ok(reconstructed) = matrix.mul(qr.q, qr.r)
  let assert Ok(qtq) = matrix.mul(matrix.transpose(qr.q), qr.q)
  let assert Ok(identity) = matrix.identity(3)

  assert matrix.approx_equal(reconstructed, a, tolerance)
  assert matrix.approx_equal(qtq, identity, tolerance)
}

pub fn gram_schmidt_qr_reconstructs_matrix_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

  let assert Ok(classical) = orthogonal.qr_classical_gram_schmidt(a)
  let assert Ok(modified) = orthogonal.qr_modified_gram_schmidt(a)
  let assert Ok(classical_reconstructed) = matrix.mul(classical.q, classical.r)
  let assert Ok(modified_reconstructed) = matrix.mul(modified.q, modified.r)
  let assert Ok(classical_qtq) =
    matrix.mul(matrix.transpose(classical.q), classical.q)
  let assert Ok(modified_qtq) =
    matrix.mul(matrix.transpose(modified.q), modified.q)
  let assert Ok(identity) = matrix.identity(2)

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

pub fn least_squares_normal_and_qr_agree_test() {
  let assert Ok(a) = matrix.from_rows([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
  let b = vector.from_list([1.0, 2.0, 2.0])
  let expected = vector.from_list([1.1666666666666667, 0.5])

  let assert Ok(normal) = least_squares.normal_equations(a, b)
  let assert Ok(qr) = least_squares.householder_qr(a, b)
  let assert Ok(cgs) = least_squares.classical_gram_schmidt_qr(a, b)
  let assert Ok(mgs) = least_squares.modified_gram_schmidt_qr(a, b)
  let assert Ok(diagnostics) =
    least_squares.stability_diagnostics(a, b, qr.solution)

  assert vector.approx_equal(normal.solution, expected, 1.0e-8)
  assert vector.approx_equal(qr.solution, expected, 1.0e-8)
  assert vector.approx_equal(cgs.solution, expected, 1.0e-8)
  assert vector.approx_equal(mgs.solution, expected, 1.0e-8)
  assert close(normal.residual_norm, qr.residual_norm)
  assert close(normal.residual_norm, cgs.residual_norm)
  assert close(normal.residual_norm, mgs.residual_norm)
  assert normal.normal_matrix_condition_inf >. 0.0
  assert close_to(diagnostics.residual_norm, qr.residual_norm, 1.0e-8)
  assert diagnostics.relative_residual >. 0.0
  assert diagnostics.normal_matrix_condition_inf >. 0.0
  assert diagnostics.normal_equation_residual_norm <=. 1.0e-8
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

  assert close(matrix.unsafe_get(reduced.t, 2, 0), 0.0)
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
  let assert Ok(q0) = matrix.col(result.q, 0)
  let assert Ok(lhs) = matrix.mul_vec(a, q0)
  let assert Ok(h0) = matrix.col(result.h, 0)
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

fn close(a: Float, b: Float) -> Bool {
  float.absolute_value(a -. b) <=. tolerance
}

fn close_to(a: Float, b: Float, tol: Float) -> Bool {
  float.absolute_value(a -. b) <=. tol
}

fn diagonal_matrix_from_vector(values: vector.Vector) -> matrix.Matrix {
  let assert Ok(result) =
    matrix.from_fn(rows: values.size, cols: values.size, with: fn(i, j) {
      case i == j {
        True -> unsafe_vector_get(values, i)
        False -> 0.0
      }
    })
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

fn unsafe_vector_get(values: vector.Vector, index: Int) -> Float {
  let assert Ok(value) = vector.get(values, index)
  value
}
