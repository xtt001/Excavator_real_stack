# V1 policy release freeze

## Frozen release

`policy-v1.3.1` is the final V1 single-cycle policy release. It preserves the
checkpoint and behavior contract of the current highest-success four-camera
model and adds checkpoint-compatible camera-batched inference.

The release identity and all binary hashes are recorded in
`releases/v1/v1.3.1.yaml`. Historical V1 experiments are indexed in
`releases/v1/release_index.yaml`; they are not represented as independent Git
tags unless a reproducible runtime bundle exists.

## Boundary

V1 covers one excavation cycle and the existing policy-to-gohome boundary. It
does not claim explicit goal following, terrain depletion reasoning, or
continuous multi-cycle execution. Those capabilities belong to V2 and require
a new session/cycle/condition data contract.

## Performance compatibility result

The frozen checkpoint loaded strictly and passed 100-step replay on validation
episodes 10120, 10129, and 10139. Across the 300 executed actions, FP32
camera-batched inference changed action values by at most `5.37e-4` and caused
zero mechanical-deadzone class disagreements. The measured RTX 5070 Ti p95
speedup ranged from `1.43x` to `1.53x`.

Three of 24,000 unaggregated chunk elements crossed a diagnostic deadzone
boundary, although none changed the temporally aggregated executed-action
class. This is why the release claims numerical compatibility within the
recorded tolerance, not bitwise identity.

FP16 also passed the recorded-observation action tolerance and deadzone check,
but remains disabled by default. Jetson AGX Orin latency and real closed-loop
behavior still require field verification.

## Verification

- Full repository test suite: `581 passed`, plus `4` passing subtests.
- Runtime bundle SHA-256 identities: verified.
- Shadow deployment preflight: passed in fail-closed `shadow_zero` mode.
- Checkpoint loading and CUDA replay: passed on three validation episodes.
- Evidence scope: offline teacher-forced replay, not real closed-loop control.
