<div align="center">

# 🧮 Lumatrix

**纯 Gleam 实现的数值稳定线性代数内核。**

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Pure Gleam](https://img.shields.io/badge/pure%20Gleam-no%20FFI-ffaff3.svg)](https://gleam.run)
[![Targets](https://img.shields.io/badge/targets-Erlang%20%7C%20JavaScript-ffaff3.svg)](https://gleam.run)

[English](README.md) · **简体中文**

</div>

---

Lumatrix 是一个完全用 Gleam 写成的数值线性代数库——没有 FFI，没有 NIF，没有需要伺候的 C 代码。Gleam 能跑到哪里它就能跑到哪里，Erlang 和 JavaScript 两个目标都支持。

它不打算和 BLAS / LAPACK 拼速度，也不假装能。它要做的是你 Gleam 应用里那个“诚实”的 `solve`：内核小到真的能读完，形状在构造时就被检查，默认就用稳定算法，求解器会告诉你答案到底有多准——而不是丢给你一个数，然后祝你好运。

## ✨ 盒子里有什么

- **构造即检查的稠密与稀疏类型** —— 坐标向量、行主序矩阵、canonical CSR 稀疏矩阵。参差不齐的行、越界下标、畸形数据在构造时就会被拒绝并返回错误；内部表示对外不可见。
- **经典工具箱** —— 带主元的 LU 与 Cholesky 直接法、Householder / Givens / Gram-Schmidt QR、最小二乘、one-sided Jacobi SVD、实/复特征值算法、定常迭代法和 Krylov 求解器。
- **处处可见的稳定性装备** —— 默认选主元、残差诊断、迭代改进、条件数估计、Krylov 方法显式处理 breakdown。失败是一个能被模式匹配的值，而不是一个悄悄算错的数。

## 🚀 快速开始

> [!NOTE]
> Lumatrix 还没有发布到 Hex。可以直接从 GitHub 拉取（git 依赖需要 Gleam ≥ 1.12），或者克隆仓库后用本地 path 依赖。

```toml
[dependencies]
lumatrix = { git = "https://github.com/L0stInFades/Lumatrix.git", ref = "main" }
# 或者克隆到本地之后：
# lumatrix = { path = "../Lumatrix" }
```

解一个线性方程组：

```math
\begin{bmatrix} 2 & 1 \\ 1 & 3 \end{bmatrix}
\begin{bmatrix} x_1 \\ x_2 \end{bmatrix}
=
\begin{bmatrix} 1 \\ 2 \end{bmatrix}
\quad\Longrightarrow\quad
x = \begin{bmatrix} 0.2 \\ 0.6 \end{bmatrix}
```

```gleam
import lumatrix/direct
import lumatrix/matrix
import lumatrix/vector

pub fn main() -> Nil {
  let assert Ok(a) = matrix.from_rows([[2.0, 1.0], [1.0, 3.0]])
  let b = vector.from_list([1.0, 2.0])
  let assert Ok(x) = direct.solve(a, b)

  assert vector.approx_equal(x, vector.from_list([0.2, 0.6]), 1.0e-8)
}
```

最小二乘求解器会同时给出解*和*它有多不准：

```gleam
import lumatrix/least_squares

let assert Ok(fit) = least_squares.solve(a, b)
// fit.solution      —— 最小二乘解 x̂
// fit.residual_norm —— ‖A·x̂ − b‖₂，残差小说明拟合得好
```

> [!TIP]
> 各模块里最朴素的 `solve` 就是稳定的默认路径（`direct.solve` 是部分选主元 LU，`least_squares.solve` 是 Householder QR）。那些更花哨的变体，等你明确知道为什么需要时再用。

## 🗺️ 模块地图

| 模块 | 里面有什么 |
| --- | --- |
| 🧱 `lumatrix/vector` · `lumatrix/matrix` | 稠密坐标向量与行主序矩阵 |
| 🚨 `lumatrix/error` | 所有可失败函数共用的 `NlaError` 错误类型 |
| 🌀 `lumatrix/complex` | 复数标量与复坐标向量 |
| 🕸️ `lumatrix/sparse` | canonical CSR 稀疏矩阵：稠密转换、矩阵-向量乘、转置、缩放、∞-范数 |
| 🔨 `lumatrix/direct` | 高斯消元、部分/完全选主元 LU、Cholesky、三角求解、行列式、逆矩阵 |
| 📐 `lumatrix/orthogonal` | Householder 变换、Givens 旋转、QR 分解 |
| 🎯 `lumatrix/least_squares` | 默认 Householder-QR 的 `solve`，外加正规方程、Givens QR、Gram-Schmidt QR、SVD 最小二乘与稳定性诊断 |
| 💎 `lumatrix/svd` | one-sided Jacobi 薄 SVD、伪逆、数值秩、2-范数、2-范数条件数 |
| 🩺 `lumatrix/error_analysis` | 残差、迭代改进、误差界、∞-范数条件数估计 |
| 🔁 `lumatrix/iterative` | Jacobi、Gauss-Seidel、SOR、最速下降、CG、预条件 CG |
| 🚄 `lumatrix/krylov` | Arnoldi、Lanczos、GMRES（普通与重启版）、BiCG、BiCGSTAB、MINRES |
| λ `lumatrix/eigen` | 幂法、Hessenberg/三对角化、Jacobi 与 QR 迭代、Schur 工具、对称特征分解、广义特征值、从实 Schur 形式提取复特征对 |

## 📐 值得知道的约定

- **构造即检查，内部不可见。** 用 `matrix.from_rows` / `from_columns` / `from_flat` / `from_fn` 和 `vector.from_list` / `zeros` / `basis` 创建值；用 `matrix.rows`、`matrix.cols`、`vector.dimension` 读形状。
- **向量就是坐标。** 不区分行/列向量：`matrix.mul_vec(a, x)` 把 `x` 当作 `A·x` 里的列向量。需要显式方向时，用 `matrix.row_matrix` 或 `matrix.column_matrix`。
- **QR 结果自带形状说明。** Householder 和 Givens QR 返回 `FullQR`（`q` 是 m×m，`r` 是 m×n）；classical / modified Gram-Schmidt 返回 `ThinQR`（`q` 是 m×n，`r` 是 n×n）。看 `form` 标签就行，不用猜。
- **两种下标读取。** 不可信的下标用 `matrix.get`；边界已经证明过的内部式代码用 `matrix.unsafe_get`。
- **求解器汇报质量。** 最小二乘结果自带残差范数；更深入的诊断量（条件数、正规方程残差）在 `least_squares.stability_diagnostics` 里。
- **SVD 从不构造 `AᵀA`。** one-sided Jacobi 避开了正规方程带来的条件数平方问题；`svd.rank`、`svd.pseudoinverse`、`svd.condition_number` 共用同一套奇异值截断规则。
- **稀疏是独立类型。** canonical CSR 存储：构造时检查坐标边界、按行列排序、合并重复坐标、丢弃显式零——是“构造时”，不是“以后再说”。
- **复特征值与广义特征值都有覆盖。** 实矩阵特征值例程能从实 Schur 形式提取复特征对，并对 `A·v = λ·v` 做残差检查；`B` 可逆的广义问题先用完全选主元直接法化为 `B⁻¹·A` 上的标准问题，再按原始 pencil 回算残差。

## 🔬 测试与质量

两层测试盯着这些内核：

- `test/` —— 各模块的单元测试与算法行为测试（gleeunit）。
- `nla_weird_matrix_tests/` —— 一个独立的测试包：把库当作只读，从外部攻击公开 API，所用的“刁钻”矩阵 fixture 由 Python + NumPy 确定性生成。

```sh
cd nla_weird_matrix_tests
python3 tools/generate_weird_cases.py   # 重新生成 fixture
gleam test
```

一条铁规矩：数值例程必须暴露收敛状态和残差质量，不许把失败藏在没人检查的返回值里。

## 🛠️ 开发

```sh
gleam format --check src test
gleam test
gleam docs build
```

CI 跑的就是同一套。欢迎贡献——见 [CONTRIBUTING.md](CONTRIBUTING.md)。

- `src/lumatrix/*.gleam` —— 库模块
- `test/lumatrix_test.gleam` —— 单元与算法行为测试
- `nla_weird_matrix_tests/` —— 外部对抗性测试包
- `gleam.toml` / `manifest.toml` —— 包元数据与锁文件
- `.github/workflows/*.yml` —— CI

## 📄 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。

---

<div align="center">

⎡ 纯 Gleam · 认真选主元 · 诚实的残差 ⎤

(=^･ω･^=)

</div>
