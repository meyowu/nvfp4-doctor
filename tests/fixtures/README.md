# Test Fixtures

Only small, public, hand-auditable fixtures belong here. Model weights, full
activations, profiler reports, and private data must remain outside Git.

- `e001_manifest_v1.json` is the schema-v1 environment-manifest fixture.
- `e002_format_oracle_v1.json` contains exhaustive hand-authored E2M1 values
  plus selected UE4M3, packing, layout, and reconstruction cases derived from
  the pinned public semantics record.

Tests must not regenerate expected values from production serializers or the
candidate FlashInfer implementation.
