# G49 N5 field deployment preparation

Date: 2026-07-17

## Outcome

G49 N5 now has a fail-closed deployment preparation path.  The checked-in
runtime configuration is `shadow_zero` only.  It samples the ACT policy at the
20 Hz training time base and lets the existing 50 Hz control pump repeat the
latest zero machine command.

The implementation does not authorize motion, choose field safety limits, or
claim that N5 is field-ready.

## Portable bundle

Build the ignored binary bundle from the completed training checkpoint:

```bash
PYTHONPATH=.:testbed python -m testbed.cli.package_act_runtime_bundle \
  --source-dir /path/to/completed/n5/ckpt \
  --output-dir policy_bundles/real_gmsl_fourcam_g49_n5_v1 \
  --candidate-id g49_n5_new_fourcam_camera_role_transition
```

The packager requires `run_metadata.json` to declare `status=completed`, copies
only the four runtime files, verifies each copy, and writes
`runtime_bundle_manifest.json` with file sizes and SHA-256 identities.

The locally prepared reference bundle has:

| File | SHA-256 |
| --- | --- |
| `policy_best.ckpt` | `0c9b755447f1c06a893394fb1111b9365eb47a8670523b6eeaef8b2df7e13b0e` |
| `dataset_stats.pkl` | `f6248e7a24eae5af9c0c758cba5eb21d19e74fea16571df58b2e008e21fe6361` |
| `resolved_config.yaml` | `ef5211342553d328e06891b013f9e7f641c2bd3617b958666669fff0a6a0822e` |
| `run_metadata.json` | `cc29bdd67fdc511eea15f21190bdd5c3bc7e753050855a44a8f87236bfdd5725` |

The binary bundle is ignored by git and must be delivered separately.

## Shadow preflight

Run the identity and semantic compatibility gate before starting the receiver:

```bash
PYTHONPATH=.:testbed python -m testbed.cli.preflight_act_shadow_deployment \
  --config testbed/testbed/configs/policy_real_gmsl_fourcam_g49_n5_shadow_v1.yaml \
  --bundle-dir policy_bundles/real_gmsl_fourcam_g49_n5_v1
```

The gate rejects:

- a bundle hash mismatch or incomplete training metadata;
- camera order differing from `video4, video5, video6, video7`;
- missing N5 eye/eye/stick/stick role encoding;
- action scaling that differs from the evaluated identity domain;
- policy sampling that differs from the 20 Hz training time base;
- `control` output, deadzone assist, or runtime gates;
- a non-strict receiver health mode or a control pump without zero-on-stop.

The packaged checkpoint also passed a two-step CUDA replay through the normal
runtime bundle loader on validation episode `10120`.

## Field-supplied adapter contract

`short_horizon_g49_n5_field_adapter_v1.yaml` is deliberately incomplete.  The
operator must supply, for the current machine and test pose:

- test ID;
- observation-gap and all-camera-age limits;
- per-axis command and delta limits;
- per-axis qvel abort limits;
- per-axis qpos lower and upper limits;
- per-axis allowed negative/positive directions.

The field axis order is `[swing, boom, stick, bucket]`.  Bundle hashes are
resolved automatically.  Once every field has been reviewed:

```bash
PYTHONPATH=.:testbed python -m testbed.cli.preflight_short_horizon_contract \
  --config /path/to/filled_field_adapter.yaml \
  --bundle-dir policy_bundles/real_gmsl_fourcam_g49_n5_v1 \
  --output-json /path/to/resolved_short_horizon_contract.json
```

This command validates and writes an immutable contract.  It still does not
send a command.  Runtime bounded-control arming and causal command-ID logging
remain separate promotion gates.
