import gleam/int
import gleam/list

/// Target-native immutable dense storage.
///
/// Erlang uses tuples and JavaScript uses `Float64Array`. Both provide O(1)
/// indexed reads. Mutation-style operations return a copied storage value so
/// Matrix and Vector remain immutable.
@internal
pub type Storage

@external(erlang, "erlang", "list_to_tuple")
@external(javascript, "./storage_ffi.mjs", "fromList")
fn from_list_ffi(values: List(Float)) -> Storage

@external(erlang, "erlang", "tuple_size")
@external(javascript, "./storage_ffi.mjs", "size")
fn size_ffi(storage: Storage) -> Int

@external(erlang, "erlang", "element")
@external(javascript, "./storage_ffi.mjs", "element")
fn element_ffi(position: Int, storage: Storage) -> Float

@external(erlang, "erlang", "setelement")
@external(javascript, "./storage_ffi.mjs", "setElement")
fn set_element_ffi(position: Int, storage: Storage, value: Float) -> Storage

@internal
pub fn from_list(values: List(Float)) -> Storage {
  from_list_ffi(values)
}

@internal
pub fn size(storage: Storage) -> Int {
  size_ffi(storage)
}

/// Read a known-valid zero-based index in O(1).
@internal
pub fn unsafe_get(storage: Storage, index: Int) -> Float {
  element_ffi(index + 1, storage)
}

/// Copy the storage and replace a known-valid zero-based index.
@internal
pub fn set(storage: Storage, index: Int, value: Float) -> Storage {
  set_element_ffi(index + 1, storage, value)
}

@internal
pub fn to_list(storage: Storage) -> List(Float) {
  indices(size(storage))
  |> list.map(fn(index) { unsafe_get(storage, index) })
}

@internal
pub fn map(storage: Storage, with f: fn(Float) -> Float) -> Storage {
  storage
  |> to_list
  |> list.map(f)
  |> from_list
}

fn indices(count: Int) -> List(Int) {
  int.range(from: 0, to: count, with: [], run: fn(acc, index) { [index, ..acc] })
  |> list.reverse
}
