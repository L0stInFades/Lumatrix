# Lumatrix

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![Release](https://github.com/L0stInFades/Lumatrix/actions/workflows/release.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

English: [README.md](README.md)

Lumatrix 是一个纯 Gleam 编写的数值线性代数库，适合需要留在 Gleam 工具链内、
同时希望 NLA 接口小而可检查的应用。

它不打算在大型原生计算负载上替代 BLAS 或 LAPACK。它的重心是一个稳定、轻依赖
的核心：显式的稠密数据结构、可预期的错误返回，以及在公共 API 依赖稳定性时避开
已知不稳健捷径的算法路径。

## 它提供什么

- 稠密坐标向量与行主序矩阵，构造时检查数据，内部结构对外隐藏。
- 直接法、正交变换、最小二乘、迭代法、Krylov 方法、SVD 工具、特征值算法和基础
  误差分析。
- 通过 `lumatrix/svd` 提供面向秩亏最小二乘、伪逆、数值秩和 2-范数条件数估计的
  稳定路径。

## API 约定

`Matrix` 和 `Vector` 的构造器是隐藏的。外部代码应通过
`matrix.from_rows`、`matrix.from_columns`、`matrix.from_flat`、`matrix.from_fn`、
`vector.from_list`、`vector.zeros` 或 `vector.basis` 创建值；通过
`matrix.rows`、`matrix.cols` 和 `vector.dimension` 查看形状。

向量在这里是坐标数组，不区分行向量和列向量。`matrix.mul_vec(a, x)` 中的 `x`
按 `A * x` 里的列向量理解。如果需要显式表达方向，可以使用 `matrix.row_matrix`
或 `matrix.column_matrix`。

QR 分解结果会带有 `form` 标签。Householder QR 和 Givens QR 返回 `FullQR`
（`q` 是 m-by-m，`r` 是 m-by-n）；classical / modified Gram-Schmidt QR 返回
`ThinQR`（`q` 是 m-by-n，`r` 是 n-by-n）。

如果下标来自用户输入，建议使用 `matrix.get`。`matrix.unsafe_get` 更适合已经确认
边界正确的内部式代码。

最小二乘求解器返回解和残差范数；条件数、正规方程残差等诊断量放在
`least_squares.stability_diagnostics` 里。

SVD 使用 one-sided Jacobi 迭代，不通过显式构造 `A^T A` 来求奇异值，因此 SVD
最小二乘路径不会引入正规方程带来的条件数平方问题。`svd.rank`、`svd.pseudoinverse`
和 `svd.condition_number` 使用同一套奇异值 cutoff 规则。

## 模块

- `lumatrix/vector` 和 `lumatrix/matrix`：稠密坐标向量与行主序矩阵。
- `lumatrix/direct`：高斯消元、部分选主元 LU、Cholesky、三角求解、行列式与逆矩阵。
- `lumatrix/orthogonal`：Householder 变换、Givens 旋转和 QR 分解。
- `lumatrix/least_squares`：默认 Householder QR 的 `solve`，以及正规方程、
  Givens QR、Gram-Schmidt QR、SVD 最小二乘和稳定性诊断。
- `lumatrix/svd`：薄 SVD、伪逆、数值秩、2-范数、2-范数条件数和基于 SVD 的求解。
- `lumatrix/error_analysis`：残差、迭代改进、误差界和无穷范数条件数估计。
- `lumatrix/iterative`：Jacobi、Gauss-Seidel、SOR、最速下降、共轭梯度和预条件
  共轭梯度。
- `lumatrix/krylov`：Arnoldi、Lanczos、GMRES 和重启 GMRES。
- `lumatrix/eigen`：幂法、Hessenberg 与三对角化、Jacobi 特征值迭代、QR 迭代、
  Schur 块工具和对称特征分解。

## 示例

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

## 开发

```sh
gleam format --check src test
gleam test
gleam docs build
```

## CI/CD

CI workflow 会在 push、pull request 和手动触发时运行。它会下载依赖、检查格式、
运行测试，并构建生成文档。

Release workflow 可以通过 `vX.Y.Z` tag 触发，也可以手动触发。它会校验发布版本
是否和 `gleam.toml` 一致，并重新运行 CI 同级别的检查。

```sh
git tag v1.0.0
git push origin v1.0.0
```

## 仓库组织

- `src/lumatrix/*.gleam`：库代码。
- `test/lumatrix_test.gleam`：单元测试和算法行为测试。
- `gleam.toml` 和 `manifest.toml`：包元数据和锁文件。
- `.github/workflows/ci.yml`：CI 中的格式、测试和文档检查。
- `.github/workflows/release.yml`：通过 release tag 或手动触发做发布前检查。

## 许可证

本项目使用 Apache License 2.0 开源，详情见 [LICENSE](LICENSE)。
