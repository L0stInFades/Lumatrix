# Lumatrix

[![CI](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/ci.yml)
[![Release](https://github.com/L0stInFades/Lumatrix/actions/workflows/release.yml/badge.svg)](https://github.com/L0stInFades/Lumatrix/actions/workflows/release.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

English: [README.md](README.md)

Lumatrix 是一个纯 Gleam 编写的数值线性代数库。它希望把经典算法保留在接近
教材的形状里，让人可以读、可以检查，也可以温和地继续扩展。

它不打算替代 BLAS 或 LAPACK。它更关心另一件事：清楚的教材式实现、稳定的 API
边界，以及一个适合学习、测试和轻量数值计算的 Gleam 工具箱。

## 它提供什么

- 稠密坐标向量与行主序矩阵，构造时检查数据，内部结构对外隐藏。
- 直接法、正交变换、最小二乘、迭代法、Krylov 方法、特征值算法和基础误差分析。
- 小而清楚的实现，便于阅读、推导、验证和继续改进。

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

## 模块

- `lumatrix/vector` 和 `lumatrix/matrix`：稠密坐标向量与行主序矩阵。
- `lumatrix/direct`：高斯消元、部分选主元 LU、Cholesky、三角求解、行列式与逆矩阵。
- `lumatrix/orthogonal`：Householder 变换、Givens 旋转和 QR 分解。
- `lumatrix/least_squares`：默认 Householder QR 的 `solve`，以及正规方程、
  Givens QR、Gram-Schmidt QR 和稳定性诊断。
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

Release workflow 用来发布到 Hex.pm，可以通过 `vX.Y.Z` tag 触发，也可以手动触发。
发布版本必须和 `gleam.toml` 里的版本一致；发布前需要在仓库 secret 或受保护的
`hexpm` environment secret 中配置 `HEXPM_API_KEY`。

第一次发布前，需要先添加 Hex.pm API key：

```sh
gh secret set HEXPM_API_KEY --repo L0stInFades/Lumatrix
```

```sh
git tag v1.0.0
git push origin v1.0.0
```

## 仓库组织

- `src/lumatrix/*.gleam`：库代码。
- `test/lumatrix_test.gleam`：单元测试和算法行为测试。
- `gleam.toml` 和 `manifest.toml`：包元数据和锁文件。
- `.github/workflows/ci.yml`：CI 中的格式、测试和文档检查。
- `.github/workflows/release.yml`：通过 release tag 或手动触发发布到 Hex.pm。

## 许可证

本项目使用 Apache License 2.0 开源，详情见 [LICENSE](LICENSE)。
