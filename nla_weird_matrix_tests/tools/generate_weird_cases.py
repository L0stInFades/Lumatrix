#!/usr/bin/env python3
"""Generate deterministic Gleam matrix fixtures with NumPy."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "src" / "nla_weird_matrix_tests" / "generated_cases.gleam"


def gleam_float(value: float) -> str:
    text = format(float(value), ".15g")
    text = text.replace("e+", "e")
    if "e" in text:
        mantissa, exponent = text.split("e")
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}e{int(exponent)}"
    if "." not in text:
        text += ".0"
    return text


def gleam_rows(rows: np.ndarray) -> str:
    row_text = []
    for row in rows.tolist():
        row_text.append("[" + ", ".join(gleam_float(x) for x in row) + "]")
    return "[" + ", ".join(row_text) + "]"


def matrix_case(name: str, rows: np.ndarray) -> str:
    return f'  MatrixCase(name: "{name}", rows: {gleam_rows(rows)}),'


def vector_case(name: str, data: np.ndarray) -> str:
    values = ", ".join(gleam_float(x) for x in data.tolist())
    return f'  VectorCase(name: "{name}", data: [{values}]),'


def gleam_vector(data: np.ndarray) -> str:
    return "[" + ", ".join(gleam_float(x) for x in data.tolist()) + "]"


def hilbert(n: int) -> np.ndarray:
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def vandermonde(points: list[float], degree: int) -> np.ndarray:
    return np.vander(np.array(points, dtype=float), N=degree, increasing=True)


def spd(seed: int, n: int, ridge: float) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0, 1.0, (n, n))
    return base.T @ base + ridge * np.eye(n)


def diagonal_dominant(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.integers(-4, 5, (n, n)).astype(float)
    for i in range(n):
        a[i, i] = np.sum(np.abs(a[i])) + 2.0 + i
    return a


def tridiagonal(n: int) -> np.ndarray:
    a = np.zeros((n, n))
    for i in range(n):
        a[i, i] = 2.5 + 0.25 * i
        if i + 1 < n:
            a[i, i + 1] = -1.0 + 0.1 * i
            a[i + 1, i] = -0.75 - 0.05 * i
    return a


def render_cases() -> str:
    square_cases = [
        ("identity_3", np.eye(3)),
        ("scaled_diagonal_4", np.diag([1.0e-3, -2.0, 5.0, 11.0])),
        ("hilbert_4", hilbert(4)),
        ("vandermonde_3", vandermonde([-2.0, 0.5, 3.0], 3)),
        (
            "nearly_singular_3",
            np.array(
                [
                    [1.0, 1.0, 1.0],
                    [1.0, 1.0 + 1.0e-6, 1.0],
                    [1.0, 1.0, 1.0 + 2.0e-6],
                ]
            ),
        ),
        (
            "alternating_sign_3",
            np.array([[(-1.0) ** (i + j) * (i + 2 * j + 1) for j in range(3)] for i in range(3)]),
        ),
        ("nonsymmetric_3", np.array([[0.0, 2.0, -1.0], [3.0, 5.0, 4.0], [1.0, -2.0, 6.0]])),
        (
            "permutation_scaled_4",
            np.array([[0.0, 0.0, 3.0, 0.0], [2.0, 0.0, 0.0, 0.0], [0.0, -5.0, 0.0, 0.0], [0.0, 0.0, 0.0, 7.0]]),
        ),
        ("tridiagonal_5", tridiagonal(5)),
        ("rank_one_plus_diag_4", np.ones((4, 4)) + np.diag([0.5, 1.5, 2.5, 3.5])),
        ("negative_mix_3", np.array([[-4.0, 2.0, 0.5], [1.0, -3.0, 2.0], [0.25, -1.0, -2.0]])),
        ("companion_like_3", np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-6.0, 11.0, -6.0]])),
    ]

    spd_cases = [
        ("spd_2_random", spd(10, 2, 0.75)),
        ("spd_3_random", spd(11, 3, 1.0)),
        ("spd_4_random", spd(12, 4, 1.5)),
        ("hilbert_spd_3", hilbert(3) + 0.05 * np.eye(3)),
        ("laplacian_spd_5", 2.0 * np.eye(5) - np.eye(5, k=1) - np.eye(5, k=-1) + 0.2 * np.eye(5)),
    ]

    tall_cases = [
        ("tall_vandermonde_4x2", vandermonde([-1.0, 0.0, 1.0, 2.0], 2)),
        ("tall_poly_5x3", vandermonde([-2.0, -0.5, 0.25, 1.5, 3.0], 3)),
        ("tall_mixed_4x3", np.array([[1.0, -1.0, 2.0], [2.0, 0.0, -1.0], [3.0, 1.0, 0.5], [4.0, 2.0, 3.0]])),
        ("tall_near_collinear_5x2", np.array([[1.0, 1.0], [2.0, 2.0001], [3.0, 3.0004], [4.0, 4.0009], [5.0, 5.0016]])),
    ]

    stationary_cases = [
        ("dd_2", diagonal_dominant(20, 2)),
        ("dd_3", diagonal_dominant(21, 3)),
        ("dd_4", diagonal_dominant(22, 4)),
    ]

    eigen_cases = [
        ("diagonal_distinct_3", np.diag([7.0, -2.0, 0.5])),
        ("symmetric_2", np.array([[2.0, 1.0], [1.0, 2.0]])),
        ("symmetric_3", np.array([[4.0, 1.0, 2.0], [1.0, 3.0, 0.0], [2.0, 0.0, 2.0]])),
        ("rotation_2", np.array([[0.0, -1.0], [1.0, 0.0]])),
        ("upper_triangular_3", np.array([[4.0, 2.0, -1.0], [0.0, 3.0, 5.0], [0.0, 0.0, -2.0]])),
        ("hessenberg_4", np.array([[3.0, 2.0, 1.0, 0.0], [1.0, 4.0, -1.0, 2.0], [0.0, -2.0, 1.0, 1.0], [0.0, 0.0, 3.0, 2.0]])),
    ]

    vector_cases = [
        ("zero_3", np.array([0.0, 0.0, 0.0])),
        ("tiny_large_4", np.array([1.0e-9, -1.0e6, 3.5, -2.0e-4])),
        ("alternating_5", np.array([(-1.0) ** i * (i + 1) for i in range(5)])),
        ("dense_3", np.array([2.5, -4.0, 0.125])),
        ("almost_axis_3", np.array([1.0, 1.0e-8, -1.0e-8])),
    ]

    lines = [
        "//// This module is generated by tools/generate_weird_cases.py.",
        "//// Regenerate it after changing the Python fixture definitions.",
        "",
        "pub type MatrixCase {",
        "  MatrixCase(name: String, rows: List(List(Float)))",
        "}",
        "",
        "pub type VectorCase {",
        "  VectorCase(name: String, data: List(Float))",
        "}",
        "",
        "pub type SolveCase {",
        "  SolveCase(",
        "    name: String,",
        "    rows: List(List(Float)),",
        "    rhs: List(Float),",
        "    solution: List(Float),",
        "  )",
        "}",
        "",
        "pub type LeastSquaresCase {",
        "  LeastSquaresCase(",
        "    name: String,",
        "    rows: List(List(Float)),",
        "    rhs: List(Float),",
        "    solution: List(Float),",
        "    residual_norm: Float,",
        "  )",
        "}",
        "",
        "pub type SvdCase {",
        "  SvdCase(",
        "    name: String,",
        "    rows: List(List(Float)),",
        "    singular_values: List(Float),",
        "  )",
        "}",
        "",
        "pub type SymmetricEigenCase {",
        "  SymmetricEigenCase(",
        "    name: String,",
        "    rows: List(List(Float)),",
        "    eigenvalues: List(Float),",
        "  )",
        "}",
        "",
        "pub type CholeskyCase {",
        "  CholeskyCase(",
        "    name: String,",
        "    rows: List(List(Float)),",
        "    lower: List(List(Float)),",
        "  )",
        "}",
        "",
    ]

    for function_name, cases in [
        ("square_cases", square_cases),
        ("spd_cases", spd_cases),
        ("tall_cases", tall_cases),
        ("stationary_cases", stationary_cases),
        ("eigen_cases", eigen_cases),
    ]:
        lines.append(f"pub fn {function_name}() -> List(MatrixCase) {{")
        lines.append("  [")
        lines.extend(matrix_case(name, rows) for name, rows in cases)
        lines.append("  ]")
        lines.append("}")
        lines.append("")

    lines.append("pub fn vector_cases() -> List(VectorCase) {")
    lines.append("  [")
    lines.extend(vector_case(name, data) for name, data in vector_cases)
    lines.append("  ]")
    lines.append("}")
    lines.append("")

    solve_inputs = [
        ("solve_spd_2", np.array([[3.0, 1.0], [1.0, 2.0]]), np.array([9.0, 8.0])),
        (
            "solve_nonsymmetric_3",
            np.array([[4.0, -2.0, 1.0], [1.0, 6.0, -2.0], [1.0, 1.0, 5.0]]),
            np.array([1.0, -3.0, 7.0]),
        ),
        (
            "solve_spd_4",
            spd(101, 4, 2.0),
            np.array([1.5, -2.0, 0.25, 3.0]),
        ),
    ]
    lines.append("pub fn solve_oracles() -> List(SolveCase) {")
    lines.append("  [")
    for name, rows, rhs in solve_inputs:
        solution = np.linalg.solve(rows, rhs)
        lines.append(
            f'  SolveCase(name: "{name}", rows: {gleam_rows(rows)}, '
            f"rhs: {gleam_vector(rhs)}, solution: {gleam_vector(solution)}),"
        )
    lines.append("  ]")
    lines.append("}")
    lines.append("")

    least_squares_inputs = [
        (
            "least_squares_vandermonde_5x3",
            vandermonde([-2.0, -0.5, 0.25, 1.5, 3.0], 3),
            np.array([4.1, 1.2, 0.9, 2.8, 8.7]),
        ),
        (
            "least_squares_mixed_4x2",
            np.array([[1.0, -2.0], [2.0, 1.0], [3.0, 0.5], [4.0, 3.0]]),
            np.array([-1.0, 2.5, 4.0, 8.0]),
        ),
    ]
    lines.append("pub fn least_squares_oracles() -> List(LeastSquaresCase) {")
    lines.append("  [")
    for name, rows, rhs in least_squares_inputs:
        solution, _, _, _ = np.linalg.lstsq(rows, rhs, rcond=None)
        residual_norm = np.linalg.norm(rows @ solution - rhs)
        lines.append(
            f'  LeastSquaresCase(name: "{name}", rows: {gleam_rows(rows)}, '
            f"rhs: {gleam_vector(rhs)}, solution: {gleam_vector(solution)}, "
            f"residual_norm: {gleam_float(residual_norm)}),"
        )
    lines.append("  ]")
    lines.append("}")
    lines.append("")

    svd_inputs = [
        ("svd_tall_4x3", tall_cases[2][1]),
        ("svd_wide_2x4", np.array([[1.0, -2.0, 3.0, 0.5], [4.0, 1.0, -1.0, 2.0]])),
        ("svd_near_rank_one_3x2", np.array([[1.0, 1.0001], [2.0, 2.0001], [3.0, 3.0004]])),
    ]
    lines.append("pub fn svd_oracles() -> List(SvdCase) {")
    lines.append("  [")
    for name, rows in svd_inputs:
        singular_values = np.linalg.svd(rows, compute_uv=False)
        lines.append(
            f'  SvdCase(name: "{name}", rows: {gleam_rows(rows)}, '
            f"singular_values: {gleam_vector(singular_values)}),"
        )
    lines.append("  ]")
    lines.append("}")
    lines.append("")

    symmetric_inputs = [
        ("eigen_symmetric_2", eigen_cases[1][1]),
        ("eigen_symmetric_3", eigen_cases[2][1]),
        ("eigen_spd_4", spd_cases[2][1]),
    ]
    lines.append("pub fn symmetric_eigen_oracles() -> List(SymmetricEigenCase) {")
    lines.append("  [")
    for name, rows in symmetric_inputs:
        eigenvalues = np.linalg.eigvalsh(rows)
        lines.append(
            f'  SymmetricEigenCase(name: "{name}", rows: {gleam_rows(rows)}, '
            f"eigenvalues: {gleam_vector(eigenvalues)}),"
        )
    lines.append("  ]")
    lines.append("}")
    lines.append("")

    cholesky_inputs = [
        ("cholesky_spd_2", spd_cases[0][1]),
        ("cholesky_spd_3", spd_cases[1][1]),
        ("cholesky_spd_4", spd_cases[2][1]),
    ]
    lines.append("pub fn cholesky_oracles() -> List(CholeskyCase) {")
    lines.append("  [")
    for name, rows in cholesky_inputs:
        lower = np.linalg.cholesky(rows)
        lines.append(
            f'  CholeskyCase(name: "{name}", rows: {gleam_rows(rows)}, '
            f"lower: {gleam_rows(lower)}),"
        )
    lines.append("  ]")
    lines.append("}")
    lines.append("")

    raw = "\n".join(lines)
    formatted = subprocess.run(
        ["gleam", "format", "--stdin"],
        input=raw,
        text=True,
        check=True,
        capture_output=True,
    )
    return formatted.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check",
        action="store_true",
        help="fail if the committed Gleam fixture differs from NumPy output",
    )
    args = parser.parse_args()
    rendered = render_cases()

    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current == rendered:
            return 0
        diff = difflib.unified_diff(
            current.splitlines(),
            rendered.splitlines(),
            fromfile=str(OUT),
            tofile="NumPy/LAPACK regenerated output",
            lineterm="",
        )
        print("\n".join(diff), file=sys.stderr)
        return 1

    OUT.write_text(rendered, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
