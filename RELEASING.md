# Releasing Lumatrix

1. Move the intended entries out of `Unreleased` in `CHANGELOG.md` and ensure
   `gleam.toml` has the same semantic version.
2. Run the complete commands documented in `README.md` plus both benchmark
   targets when the supported-size table changes.
3. Commit the release, create the exact tag `v<version>`, and push the tag.
4. The release workflow re-runs both root and external suites on Erlang and
   JavaScript, verifies NumPy/LAPACK fixtures, builds the release tarball,
   records a SHA-256 checksum, generates a GitHub build-provenance attestation,
   and creates the GitHub release. A tag push does not publish to Hex.

Manual workflow runs default to validation only. `create_github_release` and
`publish_hex` are independent and are accepted only when the selected ref is
the exact version tag. If Hex publication is wanted later, configure a
least-privilege `HEXPM_API_KEY` and run:

```sh
gh workflow run release.yml \
  --ref v1.0.0 \
  -f version=1.0.0 \
  -f create_github_release=false \
  -f publish_hex=true
```

Download a release asset and verify its source repository with GitHub CLI:

```sh
gh release download v1.0.0 \
  --repo L0stInFades/Lumatrix \
  --pattern 'lumatrix-1.0.0.tar' \
  --pattern SHA256SUMS
sha256sum --check SHA256SUMS
gh attestation verify lumatrix-1.0.0.tar \
  --repo L0stInFades/Lumatrix \
  --signer-workflow L0stInFades/Lumatrix/.github/workflows/release.yml \
  --source-ref refs/tags/v1.0.0 \
  --deny-self-hosted-runners
```

On macOS, use `shasum -a 256 lumatrix-1.0.0.tar` and compare it with
`SHA256SUMS` because the BSD userland does not provide `sha256sum` by default.
