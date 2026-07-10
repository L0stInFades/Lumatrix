pub type NlaError {
  DimensionMismatch(expected: String, actual: String)
  InvalidInput(message: String)
  NonFiniteInput(context: String)
  ArithmeticOverflow(operation: String)
  InternalInvariant(context: String)
  NotSquare(rows: Int, cols: Int)
  OutOfBounds(row: Int, col: Int)
  SingularMatrix(pivot: Int)
  ZeroNorm
  NoConvergence(iterations: Int, residual: Float)
}

pub type Error =
  NlaError
