import gleam/list
import lumatrix/error
import lumatrix/error_analysis
import lumatrix/iterative
import lumatrix/least_squares
import lumatrix/matrix
import lumatrix/vector
import nla_weird_matrix_tests/generated_cases
import nla_weird_support

pub fn error_analysis_exact_solutions_on_spd_cases_test() {
  nla_weird_support.each_matrix_case(generated_cases.spd_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let x = nla_weird_support.ramp_vector(matrix.rows(a))
    let assert Ok(b) = matrix.mul_vec(a, x)
    let assert Ok(residual) = error_analysis.residual(a, x, b)
    let assert Ok(residual_norm) = error_analysis.residual_norm2(a, x, b)
    let assert Ok(relative) = error_analysis.normwise_relative_residual(a, x, b)
    let assert Ok(backward) = error_analysis.backward_error_inf(a, x, b)
    let assert Ok(forward) = error_analysis.forward_error_inf(x, x)
    let assert Ok(condition) = error_analysis.condition_number_inf(a)
    let assert Ok(bound) = error_analysis.residual_forward_bound_inf(a, x, b)
    let assert Ok(perturbation) =
      error_analysis.perturbation_bound(1.0e-8, 2.0e-8, condition)

    nla_weird_support.assert_vector_close(
      residual,
      zero_like(x),
      nla_weird_support.loose_tolerance(),
    )
    assert residual_norm <=. nla_weird_support.loose_tolerance()
    assert relative <=. nla_weird_support.loose_tolerance()
    assert backward <=. nla_weird_support.loose_tolerance()
    assert forward <=. nla_weird_support.loose_tolerance()
    assert condition >=. 1.0
    assert bound <=. nla_weird_support.loose_tolerance()
    assert perturbation >. 0.0
  })
}

pub fn iterative_refinement_reduces_rough_residual_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 1.0], [1.0, 3.0]])
  let exact = vector.from_list([0.09090909090909091, 0.6363636363636364])
  let b = vector.from_list([1.0, 2.0])
  let rough = vector.from_list([0.08, 0.65])
  let assert Ok(rough_residual) = error_analysis.residual_norm2(a, rough, b)
  let assert Ok(refined) =
    error_analysis.iterative_refinement(a, b, rough, 5, 1.0e-10)

  assert refined.converged
  assert refined.residual_norm <. rough_residual
  assert refined.iterations <= 5
  assert list.length(refined.residual_history) >= 2
  nla_weird_support.assert_vector_close(
    refined.solution,
    exact,
    nla_weird_support.tolerance(),
  )
}

pub fn error_analysis_error_paths_test() {
  let assert Ok(identity) = matrix.identity(2)
  let assert Ok(zero_matrix) = matrix.zeros(rows: 2, cols: 2)
  let zero = vector.from_list([0.0, 0.0])
  let one = vector.from_list([1.0, 1.0])
  let assert Ok(singular) = matrix.from_rows([[1.0, 2.0], [2.0, 4.0]])

  case error_analysis.normwise_relative_residual(identity, zero, zero) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "relative residual should reject zero b"
  }
  case error_analysis.backward_error_inf(zero_matrix, zero, zero) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "backward error should reject a zero denominator"
  }
  case error_analysis.forward_error_inf(zero, one) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "forward error should reject zero exact solution"
  }
  case error_analysis.condition_number_inf(singular) {
    Error(error.SingularMatrix(_)) -> Nil
    _ -> panic as "condition number should reject singular matrices"
  }
  case error_analysis.perturbation_bound(0.5, 0.1, 4.0) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "perturbation bound should reject an invalid denominator"
  }
  case error_analysis.iterative_refinement(identity, one, one, -1, 1.0e-8) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "iterative refinement should reject negative iterations"
  }
}

pub fn least_squares_solvers_recover_exact_generated_systems_test() {
  nla_weird_support.each_matrix_case(generated_cases.tall_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let expected = nla_weird_support.alternating_vector(matrix.cols(a))
    let assert Ok(b) = matrix.mul_vec(a, expected)
    let assert Ok(normal) = least_squares.normal_equations(a, b)
    let assert Ok(householder) = least_squares.householder_qr(a, b)
    let assert Ok(givens) = least_squares.givens_qr(a, b)
    let assert Ok(classical) = least_squares.classical_gram_schmidt_qr(a, b)
    let assert Ok(modified) = least_squares.modified_gram_schmidt_qr(a, b)
    let assert Ok(diagnostics) =
      least_squares.stability_diagnostics(a, b, householder.solution)

    assert normal.residual_norm <=. 1.0e-3
    assert householder.residual_norm <=. 1.0e-3
    assert givens.residual_norm <=. 1.0e-3
    assert classical.residual_norm <=. 1.0e-3
    assert modified.residual_norm <=. 1.0e-3
    assert diagnostics.normal_matrix_condition_inf >. 0.0
    assert diagnostics.normal_equation_residual_norm <=. 1.0e-3
    nla_weird_support.assert_vector_close(normal.solution, expected, 1.0e-2)
    nla_weird_support.assert_vector_close(
      householder.solution,
      expected,
      1.0e-2,
    )
    nla_weird_support.assert_vector_close(givens.solution, expected, 1.0e-2)
    nla_weird_support.assert_vector_close(classical.solution, expected, 1.0e-2)
    nla_weird_support.assert_vector_close(modified.solution, expected, 1.0e-2)
  })
}

pub fn least_squares_noisy_problem_has_consistent_residuals_test() {
  let assert Ok(a) =
    matrix.from_rows([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
  let b = vector.from_list([1.0, 2.1, 2.9, 4.2])
  let assert Ok(normal) = least_squares.normal_equations(a, b)
  let assert Ok(householder) = least_squares.householder_qr(a, b)
  let assert Ok(givens) = least_squares.givens_qr(a, b)
  let assert Ok(classical) = least_squares.classical_gram_schmidt_qr(a, b)
  let assert Ok(modified) = least_squares.modified_gram_schmidt_qr(a, b)
  let assert Ok(residual) =
    least_squares.residual_norm(a, householder.solution, b)

  nla_weird_support.assert_close_to(
    normal.residual_norm,
    householder.residual_norm,
    nla_weird_support.loose_tolerance(),
  )
  nla_weird_support.assert_close_to(
    givens.residual_norm,
    householder.residual_norm,
    nla_weird_support.loose_tolerance(),
  )
  nla_weird_support.assert_close_to(
    classical.residual_norm,
    householder.residual_norm,
    nla_weird_support.loose_tolerance(),
  )
  nla_weird_support.assert_close_to(
    modified.residual_norm,
    householder.residual_norm,
    nla_weird_support.loose_tolerance(),
  )
  nla_weird_support.assert_close_to(
    residual,
    householder.residual_norm,
    nla_weird_support.tolerance(),
  )
}

pub fn least_squares_error_paths_test() {
  let assert Ok(wide) = matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let b2 = vector.from_list([1.0, 2.0])
  let b3 = vector.from_list([1.0, 2.0, 3.0])
  let wrong_x = vector.from_list([1.0])

  case least_squares.normal_equations(wide, b2) {
    Error(error.DimensionMismatch(
      expected: "m >= n and b length m",
      actual: "2x3, b=2",
    )) -> Nil
    _ -> panic as "least squares should reject wide matrices"
  }
  case least_squares.householder_qr(wide, b3) {
    Error(error.DimensionMismatch(
      expected: "m >= n and b length m",
      actual: "2x3, b=3",
    )) -> Nil
    _ -> panic as "least squares should reject wrong RHS length"
  }
  case least_squares.stability_diagnostics(wide, b2, wrong_x) {
    Error(error.DimensionMismatch(
      expected: "m >= n, b length m and x length n",
      actual: "2x3, b=2, x=1",
    )) -> Nil
    _ -> panic as "least squares diagnostics should validate x length"
  }
}

pub fn stationary_methods_converge_on_generated_diagonal_dominant_cases_test() {
  nla_weird_support.each_matrix_case(
    generated_cases.stationary_cases(),
    fn(sample) {
      let a = nla_weird_support.matrix_from_case(sample)
      let expected = nla_weird_support.ramp_vector(matrix.rows(a))
      let assert Ok(b) = matrix.mul_vec(a, expected)
      let assert Ok(initial) = vector.zeros(matrix.rows(a))
      let assert Ok(jacobi) = iterative.jacobi(a, b, initial, 250, 1.0e-8)
      let assert Ok(gs) = iterative.gauss_seidel(a, b, initial, 250, 1.0e-8)
      let assert Ok(sor) = iterative.sor(a, b, initial, 1.05, 250, 1.0e-8)
      let assert Ok(jacobi_diagnostics) =
        iterative.jacobi_convergence_diagnostics(a)
      let assert Ok(gs_diagnostics) =
        iterative.gauss_seidel_convergence_diagnostics(a)
      let assert Ok(sor_diagnostics) =
        iterative.sor_convergence_diagnostics(a, 1.05)
      let assert Ok(stationary) =
        iterative.stationary_convergence_diagnostics(
          a,
          iterative.JacobiIteration,
        )

      assert jacobi.converged
      assert gs.converged
      assert sor.converged
      assert jacobi.residual_norm <=. 1.0e-8
      assert gs.residual_norm <=. 1.0e-8
      assert sor.residual_norm <=. 1.0e-8
      assert jacobi_diagnostics.sufficient_convergence
      assert gs_diagnostics.sufficient_convergence
      assert sor_diagnostics.sufficient_convergence
      assert stationary.sufficient_convergence
      nla_weird_support.assert_vector_close(jacobi.solution, expected, 1.0e-6)
      nla_weird_support.assert_vector_close(gs.solution, expected, 1.0e-6)
      nla_weird_support.assert_vector_close(sor.solution, expected, 1.0e-6)
    },
  )
}

pub fn conjugate_gradient_family_converges_on_spd_cases_test() {
  nla_weird_support.each_matrix_case(generated_cases.spd_cases(), fn(sample) {
    let a = nla_weird_support.matrix_from_case(sample)
    let expected = nla_weird_support.alternating_vector(matrix.rows(a))
    let assert Ok(b) = matrix.mul_vec(a, expected)
    let assert Ok(initial) = vector.zeros(matrix.rows(a))
    let assert Ok(cg) = iterative.conjugate_gradient(a, b, initial, 100, 1.0e-9)
    let assert Ok(identity_pcg) =
      iterative.preconditioned_conjugate_gradient_with(
        a,
        b,
        initial,
        100,
        1.0e-9,
        fn(r) { Ok(r) },
      )
    let assert Ok(pcg) =
      iterative.preconditioned_conjugate_gradient(a, b, initial, 100, 1.0e-9)

    assert cg.converged
    assert identity_pcg.converged
    assert pcg.converged
    nla_weird_support.assert_vector_close(cg.solution, expected, 1.0e-6)
    nla_weird_support.assert_vector_close(
      identity_pcg.solution,
      expected,
      1.0e-6,
    )
    nla_weird_support.assert_vector_close(pcg.solution, expected, 1.0e-6)
  })
}

pub fn steepest_descent_converges_on_stable_spd_test() {
  let assert Ok(a) = matrix.from_rows([[4.0, 1.0], [1.0, 3.0]])
  let expected = vector.from_list([0.09090909090909091, 0.6363636363636364])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(initial) = vector.zeros(2)
  let assert Ok(result) = iterative.steepest_descent(a, b, initial, 80, 1.0e-6)
  let assert Ok(practical) =
    iterative.practical_conjugate_gradient(a, b, initial, 10, 1.0e-10, 2)

  assert result.converged
  assert practical.converged
  nla_weird_support.assert_vector_close(result.solution, expected, 1.0e-5)
  nla_weird_support.assert_vector_close(practical.solution, expected, 1.0e-8)
}

pub fn iterative_error_paths_test() {
  let assert Ok(non_square) =
    matrix.from_rows([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  let assert Ok(zero_diagonal) = matrix.from_rows([[0.0, 1.0], [1.0, 2.0]])
  let assert Ok(identity) = matrix.identity(2)
  let b2 = vector.from_list([1.0, 2.0])
  let b3 = vector.from_list([1.0, 2.0, 3.0])
  let assert Ok(initial2) = vector.zeros(2)

  case iterative.jacobi(non_square, b2, initial2, 10, 1.0e-8) {
    Error(error.NotSquare(rows: 2, cols: 3)) -> Nil
    _ -> panic as "iterative methods should reject non-square matrices"
  }
  case iterative.gauss_seidel(identity, b3, initial2, 10, 1.0e-8) {
    Error(error.DimensionMismatch(expected: "2", actual: "3 and 2")) -> Nil
    _ -> panic as "iterative methods should reject mismatched vector sizes"
  }
  case iterative.sor(identity, b2, initial2, 2.0, 10, 1.0e-8) {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "SOR should reject omega outside (0, 2)"
  }
  case
    iterative.practical_conjugate_gradient(
      identity,
      b2,
      initial2,
      10,
      1.0e-8,
      0,
    )
  {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "practical CG should reject nonpositive recompute interval"
  }
  case iterative.jacobi(zero_diagonal, b2, initial2, 10, 1.0e-8) {
    Error(error.SingularMatrix(0)) -> Nil
    _ -> panic as "Jacobi should reject a zero diagonal"
  }
  case
    iterative.preconditioned_conjugate_gradient_with(
      identity,
      b2,
      initial2,
      10,
      1.0e-8,
      fn(_) { Error(error.InvalidInput("bad preconditioner")) },
    )
  {
    Error(error.InvalidInput(_)) -> Nil
    _ -> panic as "custom preconditioner errors should propagate"
  }
}

fn zero_like(values: vector.Vector) -> vector.Vector {
  let assert Ok(zero) = vector.zeros(vector.dimension(values))
  zero
}
