# Releasing Lumatrix

1. Move the intended entries out of `Unreleased` in `CHANGELOG.md` and ensure
   `gleam.toml` has the same semantic version.
2. Run the complete commands documented in `README.md` plus both benchmark
   targets when the supported-size table changes.
3. Commit the release, create the exact tag `v<version>`, and push the tag.
4. The release workflow re-runs both root and external suites on Erlang and
   JavaScript, verifies NumPy/LAPACK fixtures, builds the Hex tarball, records a
   SHA-256 checksum, generates a GitHub build-provenance attestation, publishes
   through `gleam publish --yes`, and creates the GitHub release.

The repository secret `HEXPM_API_KEY` must contain a least-privilege Hex API
key allowed to publish `lumatrix`. A manual workflow run defaults to validation
only; publishing is accepted only when the selected ref is the exact version
tag.

Download a release asset and verify its source repository with GitHub CLI:

```sh
gh attestation verify lumatrix-1.0.0.tar --repo L0stInFades/Lumatrix
sha256sum --check SHA256SUMS
```

On macOS, use `shasum -a 256 lumatrix-1.0.0.tar` and compare it with
`SHA256SUMS` because the BSD userland does not provide `sha256sum` by default.
