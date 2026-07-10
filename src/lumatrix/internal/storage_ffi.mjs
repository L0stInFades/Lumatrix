export function fromList(values) {
  return Float64Array.from(values);
}

export function size(storage) {
  return storage.length;
}

export function element(position, storage) {
  return storage[position - 1];
}

export function setElement(position, storage, value) {
  const copy = storage.slice();
  copy[position - 1] = value;
  return copy;
}
