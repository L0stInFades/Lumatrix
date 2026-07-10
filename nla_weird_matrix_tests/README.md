# nla_weird_matrix_tests

This is a separate Gleam test package for the parent `lumatrix` library.

The checked-in source modules under `../src/lumatrix` are treated as read-only. This
package depends on the parent package through a local path dependency and tests
the public APIs from the outside.

Python 3 + NumPy generate deterministic "weird" matrix fixtures:

```sh
python3 tools/generate_weird_cases.py
gleam test
```

The generated Gleam fixture module is committed so the tests can run without
regenerating fixtures every time.
