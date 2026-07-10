# E52 Real-Machine Experiment Plan and Commands

Status: prepared. No hardware motion is authorized by this document.

Runtime implementation source:

- repository: `/home/pingfan/Excavator_real_stack`;
- branch: `fs/online-qc-dev`;
- E52 base commit: `5f433bf` (`Add E52 gated ACT experiments and runtime integration`);
- supervised-trace implementation commit: `809afc2` (`Add E52 supervised control tracing`);
- Jetson repository: `/media/mundane/D/Excavator_real_stack`;
- Jetson deployment was not reachable when this plan was written, so C01-C03
  are mandatory before any field run.

This document is the human runbook. The machine-readable task state is in
`docs/e52_real_machine_experiment_tasks_20260710.csv`. Every task in that CSV
references a command ID in this document.

## Goal and claim boundary

The experiment must determine, in order:

1. whether live E52 inference produces stable raw ACT and gated intent while
   commanded motion remains zero;
2. how reviewed one-axis commands map to real qpos/qvel response;
3. whether supervised E52 output produces the expected physical direction and
   remains in a repeatedly observed trajectory region;
4. whether the same result survives repeated anchors and changed soil or
   lighting before attempting a longer segment.

Shadow success proves inference and trace integrity, not physical correctness.
Trace completeness proves that a run is reviewable, not that its motion was
correct. Failed or unused HDF5 episodes may be stress inputs, but are not clean
expert labels for a physical-success claim.

## 1. Files and ownership

| Purpose | File or directory | What to verify |
| --- | --- | --- |
| E52 deployment config | `testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml` | Checked-in `output_mode` remains `shadow_zero` |
| E52 policy and gate bundle | `policy_bundles/real_gmsl_eye2_e52_v1/` | C02 and C03 pass |
| Bundle contract verifier | `scripts/verify_e52_runtime_bundle.py` | Prints JSON with `ok: true` |
| No-motion shadow entrypoint | `scripts/run_e52_policy_shadow_check.sh` | Runs E52 with zero commanded action |
| Shadow log verifier | `scripts/e53_verify_no_motion_policy_log.py` | Produces `e53_no_motion_report.json` with `ok: true` |
| Scripted axis-response tool | `scripts/calibrate_axis_response.py` | Used only in P3 with approved inputs |
| Supervised E52 motion entrypoint | `scripts/run_e52_policy_control_trace.sh` | Requires `CONFIRM_HARDWARE_MOTION=YES` |
| Trace analyzer | `scripts/summarize_e52_control_trace.py` | Produces manifest, summary, context, and timeline |
| Main field commands | `docs/host_slave_start_commands.md` | Physical host/Jetson bring-up reference |
| Task status table | `docs/e52_real_machine_experiment_tasks_20260710.csv` | Update `status` after review, not merely after command exit |

Do not edit the E52 YAML to enable control. The supervised control script
applies a request-local `control` override after explicit operator
confirmation.

## 2. Execution model

Use these terminals:

| Terminal | Machine | Responsibility |
| --- | --- | --- |
| H1 | Host development computer | Deploy exact files, record host provenance, and run development tests |
| J1 | Jetson | Run `slave_real_stack.sh run --force --no-receiver`; keep it visible |
| J2 | Jetson | Run probes, shadow, scripted response, or supervised E52 trace |
| O1 | At the machine | Watch physical motion and use Ctrl+C, bridge stop, or physical emergency stop |

For this E52 runbook, J1 must use `--no-receiver`. The shadow or control-trace
script in J2 starts the only policy `record_real` process. Do not simultaneously
run `slave_real_stack.sh --policy-remote`, another receiver, or a generic
`record_real --policy-output-mode control` command.

The experiment order is strict:

```text
P0 provenance -> P1 no-motion platform -> P2 E52 shadow
-> P3 scripted axis response -> P4 supervised E52 trace
-> P5 supervised segment -> P6 full cycle and separate gohome decision
```

`stick` is intentionally a zero-action axis in E52. Do not create a stick
motion task. Effective stick output during P2-P6 is a stop condition.

## 3. Shared field root

Run once in J2. Reuse the printed path in every later Jetson terminal:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="/media/mundane/EXTERNAL_USB/policy_control_tests/e52_field_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${E52_FIELD_ROOT}"
printf '%s\n' "${E52_FIELD_ROOT}" | tee /tmp/e52_field_root
df -h "${E52_FIELD_ROOT}"
```

In a newly opened Jetson terminal, restore it with:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
test -d "${E52_FIELD_ROOT}"
printf 'E52_FIELD_ROOT=%s\n' "${E52_FIELD_ROOT}"
```

Do not reuse a field root from a previous day or machine session.

## 4. P0: provenance and deployment

### C00: record host provenance

Run in H1:

```bash
cd /home/pingfan/Excavator_real_stack
mkdir -p artifacts/e52_field_preflight
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git branch --show-current
  git rev-parse HEAD
  git status --short
  git rev-list --left-right --count origin/fs/online-qc-dev...HEAD
} | tee artifacts/e52_field_preflight/host_repository.txt
```

Requirement: the host source must include `809afc2` or a reviewed descendant.
Existing unrelated files shown by `git status` must not be copied, committed,
or deleted as part of this experiment.

### C01: deploy exact E52 runtime files and record Jetson provenance

Run in H1 after `ssh slave-jetson 'hostname'` succeeds:

```bash
cd /home/pingfan/Excavator_real_stack

RUNTIME_FILES=(
  docs/e52_real_machine_experiment_plan_20260710.md
  docs/e52_real_machine_experiment_tasks_20260710.csv
  docs/host_slave_start_commands.md
  scripts/run_e52_policy_control_trace.sh
  scripts/summarize_e52_control_trace.py
  testbed/testbed/backends/real/action_pump.py
  testbed/testbed/cli/record_real.py
)

rsync -avR "${RUNTIME_FILES[@]/#/./}" \
  slave-jetson:/media/mundane/D/Excavator_real_stack/

sha256sum "${RUNTIME_FILES[@]}" | ssh slave-jetson \
  'cd /media/mundane/D/Excavator_real_stack && sha256sum -c -'

ssh slave-jetson '
  cd /media/mundane/D/Excavator_real_stack
  git branch --show-current
  git rev-parse HEAD
  git status --short
  git merge-base --is-ancestor 5f433bf HEAD
'
```

Requirements:

- every `sha256sum -c` line is `OK`;
- the Jetson path is `/media/mundane/D/Excavator_real_stack`;
- the Jetson Git base contains `5f433bf`; C01 overlays the reviewed trace files
  and their exact hashes are the deployed-code evidence;
- no whole-repository sync, `--delete`, forced checkout, or overwrite of
  unrelated Jetson changes is allowed.

After C01, run Section 3 in J2 to create `E52_FIELD_ROOT`, then save Jetson
provenance:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
{
  date -u +%Y-%m-%dT%H:%M:%SZ
  git branch --show-current
  git rev-parse HEAD
  git status --short
} | tee "${E52_FIELD_ROOT}/jetson_repository.txt"
```

### C02: verify the E52 bundle contract

Run in J2:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
export PYTHON="$PWD/.venv/bin/python"
export PYTHONPATH="$PWD/testbed"

"${PYTHON}" scripts/verify_e52_runtime_bundle.py \
  --config testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml \
  --bundle-dir policy_bundles/real_gmsl_eye2_e52_v1 \
  | tee "${E52_FIELD_ROOT}/e52_bundle_preflight.json"
```

Requirement: the JSON has `ok: true`, `camera_names` equal to
`["video4", "video5"]`, `low_dim_keys` equal to `["qpos"]`,
`output_mode` equal to `shadow_zero`, and
`runtime_gate_stack_loaded: true`.

### C03: verify and preserve artifact hashes

Run in J2:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"

(
  cd policy_bundles/real_gmsl_eye2_e52_v1
  sha256sum -c SHA256SUMS
) | tee "${E52_FIELD_ROOT}/bundle_sha256_check.txt"

sha256sum \
  testbed/testbed/configs/policy_real_gmsl_eye2_e52_v1.yaml \
  policy_bundles/real_gmsl_eye2_e52_v1/candidate_package_manifest.json \
  policy_bundles/real_gmsl_eye2_e52_v1/deadzone_policy_raw_for_runtime_scale.json \
  | tee "${E52_FIELD_ROOT}/runtime_input_sha256.txt"
```

Requirement: every `SHA256SUMS` entry is `OK`. P0 fails on a missing file,
hash mismatch, wrong camera set, wrong low-dimensional contract, enabled
deadzone assist, or a checked-in E52 mode other than `shadow_zero`.

## 5. P1: no-motion platform preflight

Execution order is C10, C12, C11, C14, then the C13/S0 check at the start of
P2. C11 requires the CAN service started by C12.

### C10: physical stop-path signoff

There is no software test for T10. Before J1 starts, replace every placeholder
and run in J2:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
cat > "${E52_FIELD_ROOT}/operator_signoff.txt" <<'EOF'
operator=<name>
observer=<name>
machine_area=<area>
physical_estop_checked=yes/no
manual_power_cut_checked=yes/no
ctrl_c_operator=<name>
soil_condition=<description>
lighting_weather=<description>
EOF
```

Requirement: the on-machine operator confirms the physical paths. A config
flag or unit test is not evidence for T10.

### C11: four-IMU read-only probe

Run in J2 after C12 reports CAN and bridge startup:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
./scripts/imu_can_probe.py \
  --interface can3 \
  --duration-s 3 \
  --require-four \
  --output "${E52_FIELD_ROOT}/imu_can_probe.json"
```

Requirement: exit code 0, `missing_raw_addr_0_to_3` is empty, and captured IMU
frames are non-zero.

### C12: start and inspect the bottom stack

Run in J1 and leave this terminal visible:

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh run --force --no-receiver
```

Run in J2:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
./scripts/slave_real_stack.sh status \
  | tee "${E52_FIELD_ROOT}/slave_stack_status.txt"
ss -ltnp | grep -E ':(8765|8766)[[:space:]]' \
  | tee "${E52_FIELD_ROOT}/bridge_ports.txt"
```

Requirements:

- bridge and gateway are running;
- ports `8765` and `8766` are listening;
- GMSL SHM entries for the required cameras are present;
- receiver port `8770` is not started by J1;
- no stale-camera, bridge-health, or CAN fault is visible.

### C13: idle zero-command check

T13 uses the S0 execution in C20. It passes only if the generated E53 report
has `ok: true` and every `safe_action`, `commanded_action`, and
`policy_returned_action` remains exactly zero.

### C14: USB root check

Run in J2:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
findmnt /media/mundane/EXTERNAL_USB
df -h /media/mundane/EXTERNAL_USB
touch "${E52_FIELD_ROOT}/.write_test"
rm "${E52_FIELD_ROOT}/.write_test"
```

Requirement: the USB is mounted, the field root is writable, and free space is
enough for the planned image interval and run duration. Record the `df` output.

## 6. P2: E52 no-motion shadow matrix

### C20: run one named shadow anchor

J1 must still be running C12. In J2, set all context fields before each run:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
export ANCHOR="S0_home"                 # S0_home S1_pre_swing S2_pre_boom S3_pre_bucket S4_tail S5_shifted_visual
export EXPECTED_AXIS="none"             # none swing boom bucket
export EXPECTED_DIRECTION="zero"        # zero positive negative
export ANCHOR_NOTES="home idle check"   # describe pose soil and lighting
export MAX_STEPS=500
export TEST_LOG_DIR="${E52_FIELD_ROOT}/shadow/${ANCHOR}_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "${TEST_LOG_DIR}"

{
  printf 'anchor=%s\n' "${ANCHOR}"
  printf 'expected_axis=%s\n' "${EXPECTED_AXIS}"
  printf 'expected_direction=%s\n' "${EXPECTED_DIRECTION}"
  printf 'notes=%s\n' "${ANCHOR_NOTES}"
} | tee "${TEST_LOG_DIR}/anchor_context.txt"

./scripts/run_e52_policy_shadow_check.sh \
  2>&1 | tee "${TEST_LOG_DIR}/console.log"
```

Repeat C20 for T20-T25. The operator manually places the machine at each
anchor; the policy never commands motion in this phase.

Required review by anchor:

| Anchor | Expected review |
| --- | --- |
| S0 home | No effective action and no early gohome |
| S1 pre-swing | Raw and gated swing direction matches annotation |
| S2 pre-boom | Boom sign and qvel input are plausible |
| S3 pre-bucket | Bucket direction is plausible and stick stays inactive |
| S4 tail | Effective action is zero and gohome timing is reviewable |
| S5 shifted visual | No persistent wrong-axis or wrong-direction intent |

### C26: inspect the latest shadow result

Run in J2 after each C20:

```bash
export REPORT="$(find "${TEST_LOG_DIR}" -name e53_no_motion_report.json -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
export STEPS="${REPORT%/e53_no_motion_report.json}/steps.jsonl"
test -f "${REPORT}"
test -f "${STEPS}"

python3 - "${REPORT}" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "ok",
    "steps",
    "policy_nonzero_steps",
    "policy_error_steps",
    "wrong_output_mode_steps",
    "runtime_gate_missing_counts",
    "max_abs_by_key",
    "errors",
):
    print(f"{key}: {report.get(key)}")
PY
```

P2 pass requirements:

- every E53 report has `ok: true`;
- policy inference is live rather than constantly zero or failing;
- raw ACT and every E52 gate stage are present;
- commanded, safe, and returned actions remain zero;
- stick remains ineffective;
- the operator annotation and persistent inferred direction agree;
- no gohome request occurs outside the reviewed tail context.

Do not proceed to P3 because one isolated frame looks plausible. Review the
whole `steps.jsonl` interval and camera frames.

## 7. P3: scripted one-axis physical response

P3 does not execute the policy. It measures how a known one-axis command maps
to qpos/qvel response. Values must come from reviewed deadzone evidence and the
agreed trial setup; this runbook intentionally supplies no guessed amplitude.

### C30: run one axis and one direction

J1 must still be running C12. In J2, define every required input:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
export TASK_ID="T30"
export AXIS="swing"                       # swing boom bucket; never stick for E52
export DIRECTION="positive"              # positive or negative
export AMPLITUDES=""                      # required: approved comma-separated values
export DURATION_S=""                      # required: approved pulse duration
export SETTLE_S=""                        # required: approved settle duration
export ABORT_DELTA_RAD=""                 # required: approved physical delta

: "${AMPLITUDES:?set approved AMPLITUDES}"
: "${DURATION_S:?set approved DURATION_S}"
: "${SETTLE_S:?set approved SETTLE_S}"
: "${ABORT_DELTA_RAD:?set approved ABORT_DELTA_RAD}"

./scripts/calibrate_axis_response.py \
  --host 127.0.0.1 \
  --port 8766 \
  --axis "${AXIS}" \
  --direction "${DIRECTION}" \
  --amplitudes "${AMPLITUDES}" \
  --duration-s "${DURATION_S}" \
  --settle-s "${SETTLE_S}" \
  --abort-delta-rad "${ABORT_DELTA_RAD}" \
  --output "${E52_FIELD_ROOT}/axis_response/${TASK_ID}_${AXIS}_${DIRECTION}.jsonl" \
  --confirm-hardware-motion
```

Use C30 for T30-T35 with the matching task ID, axis, and direction. Stop after
each direction and review its JSONL before the next command.

Per-direction pass requirements:

- all controller acknowledgments are valid and fault codes are empty;
- active-axis qpos/qvel response has the expected direction;
- displacement remains within the approved trial range;
- no uncommanded cross-axis or post-zero motion requires intervention;
- response is repeatable, not a single sample;
- stick remains outside the E52 motion schedule.

### C36: aggregate P3 evidence

There is no automatic promotion command. Review all six JSONL files, replace
the placeholders, and run in J2:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
cat > "${E52_FIELD_ROOT}/axis_response/P3_review.txt" <<'EOF'
reviewer=<name>
operator=<name>
accepted_trials=<task IDs>
rejected_trials=<task IDs and reasons>
swing_direction_latency_range=<reviewed result>
boom_direction_latency_range=<reviewed result>
bucket_direction_latency_range=<reviewed result>
cross_axis_or_post_zero_findings=<reviewed result>
decision=pass|fail
EOF
```

P4 remains blocked until the operator and reviewer sign this evidence.

## 8. P4: supervised E52 control trace

P4 uses human observation, Ctrl+C, bridge stop, and the physical emergency
path. It does not add another automatic qpos/qvel state machine.

### C40: verify control-trace tooling before motion

Run development tests in H1:

```bash
cd /home/pingfan/Excavator_real_stack
bash -n scripts/run_e52_policy_control_trace.sh
PYTHONPATH=testbed:. pytest -q \
  testbed/tests/test_e52_control_trace.py \
  testbed/tests/test_policy_action_source.py \
  testbed/tests/test_realworld_v1.py \
  testbed/tests/test_e52_runtime_deployment.py \
  testbed/tests/test_e53_no_motion_policy_log_verifier.py \
  testbed/tests/test_runtime_gate_stack.py
```

Run the motion-confirmation refusal check in J2:

```bash
cd /media/mundane/D/Excavator_real_stack
unset REFUSAL_RC
env -u CONFIRM_HARDWARE_MOTION ./scripts/run_e52_policy_control_trace.sh \
  || REFUSAL_RC=$?
test "${REFUSAL_RC:-0}" -eq 2
```

The second line is expected to exit with code 2 without opening a control
session. C40 passes only after the development tests, C02, and this refusal
check pass.

### C41: run one supervised E52 trace

J1 must still be running C12. In J2, use a reviewed P2 anchor and set a finite
run length explicitly:

```bash
cd /media/mundane/D/Excavator_real_stack
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
export OPERATOR_ID=""                     # required
export SESSION_ID="T41_S0_home"           # task ID plus anchor
export NOTES=""                           # required: expected axis direction pose soil lighting
export MAX_STEPS=""                       # required: finite value approved for this trial
export IMAGE_INTERVAL_STEPS=5
export LOG_ROOT="${E52_FIELD_ROOT}/control/${SESSION_ID}"

: "${OPERATOR_ID:?set OPERATOR_ID}"
: "${NOTES:?set NOTES}"
: "${MAX_STEPS:?set reviewed finite MAX_STEPS}"

CONFIRM_HARDWARE_MOTION=YES \
  ./scripts/run_e52_policy_control_trace.sh
```

Use C41 for T41-T46 by changing `SESSION_ID`, `NOTES`, and the reviewed anchor.
The script performs bundle preflight, applies the request-local control mode,
captures trace/images, requests terminal zero on normal exit or Ctrl+C, and
runs the analyzer.

Stop immediately on wrong axis/direction, effective stick action, unexpected
gohome transition, unexpected continued motion, invalid sensors, bridge fault,
or an operator call. Preserve the output directory after every stop. Do not
rerun by loosening a condition in the same session.

### C47: inspect one generated control trace

Run in J2 after C41:

```bash
export TRACE_SUMMARY="$(find "${LOG_ROOT}" -path '*/e52_trace_analysis/trace_summary.json' -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
export RUN_DIR="${TRACE_SUMMARY%/e52_trace_analysis/trace_summary.json}"

test -f "${RUN_DIR}/metadata.json"
test -f "${RUN_DIR}/steps.jsonl"
test -f "${RUN_DIR}/summary.json"
test -f "${RUN_DIR}/termination.json"
test -f "${RUN_DIR}/e52_trace_analysis/run_manifest.json"
test -f "${RUN_DIR}/e52_trace_analysis/trace_context.json"
test -f "${RUN_DIR}/e52_trace_analysis/trace_timeline.png"

python3 - "${TRACE_SUMMARY}" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
for key in (
    "trace_complete",
    "trace_errors",
    "steps",
    "duration_s",
    "effective_hz",
    "policy_error_steps",
    "receiver_health_bad_steps",
    "controller_ack_bad_steps",
    "guard_triggered_steps",
    "gohome_requested_steps",
    "stick",
    "state",
    "termination",
):
    print(f"{key}: {summary.get(key)}")
PY
```

Trace integrity requires:

- `trace_complete: true`;
- no policy errors, bad receiver-health steps, or bad acknowledgments;
- `termination.zero_command_confirmed: true`;
- run manifest actual and observed output modes are `control`;
- raw ACT, each E52 stage, `safe_action`, `commanded_action`,
  `raw_low_level_command`, qpos, and qvel are reviewable.

Trace integrity is not physical success. The operator must also review
`trace_timeline.png`, captured frames, qpos/qvel response, expected direction,
and the exact stop event.

## 9. P5 and P6 promotion tasks

### C50: supervised task segment

Use C41 and C47 with `SESSION_ID=T50_<condition>` only after T41-T47 pass on
repeated shorter runs. The segment duration and start anchor must be written in
`NOTES`. T50-T55 remain evidence-gated, even though the command exists.

### C55: P5 promotion decision

After T50-T54 are reviewed, replace the placeholders and run in J2:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
cat > "${E52_FIELD_ROOT}/P5_decision.txt" <<'EOF'
reviewer=<name>
accepted_segment_runs=<absolute run directories>
rejected_segment_runs=<absolute run directories and reasons>
progress_review=pass|fail
tail_stop_review=pass|fail
gohome_log_only_review=pass|fail
decision=approve_full_cycle|remain_at_segment|reject
EOF
```

T55 passes only when `decision=approve_full_cycle` is supported by all cited
run directories. The file does not itself authorize motion.

### C60: supervised action-side cycle

Use C41 and C47 with `SESSION_ID=T60_<condition>` only after the signed P5
review. A longer `MAX_STEPS` is not itself authorization. Extend duration only
from reviewed shorter-run evidence.

### C62: separate gohome acceptance

There is currently no accepted C62 field command. Do not use the generic
go-home controller as an implicit extension of P4/P5. T62 remains blocked until
its start-state contract, operator procedure, expected trajectory, and review
criteria are written and approved separately.

### C64: raw ACT counterfactual comparison

Use the raw `policy_action` already present in E52 `steps.jsonl` as a logged
counterfactual against the executed E52 gate chain. Do not physically execute
raw ACT merely to populate T64. A physical raw-ACT A/B requires a separate
approved task and matched-state reset procedure.

Run this once for each matched E52 run after setting its absolute `RUN_DIR`:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
export RUN_DIR=""  # required: receiver run containing steps.jsonl
: "${RUN_DIR:?set absolute receiver RUN_DIR}"
export COUNTERFACTUAL_REPORT="${E52_FIELD_ROOT}/raw_counterfactual_$(basename "${RUN_DIR}").json"

python3 - "${RUN_DIR}/steps.jsonl" "${COUNTERFACTUAL_REPORT}" <<'PY'
import json
import sys
from pathlib import Path

steps_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
pairs = []
for line in steps_path.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    row = json.loads(line)
    raw = row.get("policy_action")
    gated = row.get("policy_temporal_direction_action")
    commanded = row.get("commanded_action")
    if not all(isinstance(value, list) and len(value) == 4 for value in (raw, gated, commanded)):
        continue
    pairs.append((raw, gated, commanded))

if not pairs:
    raise SystemExit("no complete raw/gated/commanded action rows")

def mean_abs(which):
    return [
        sum(abs(float(row[which][axis])) for row in pairs) / len(pairs)
        for axis in range(4)
    ]

payload = {
    "claim_boundary": "logged_counterfactual_only_not_physical_raw_act",
    "steps_jsonl": str(steps_path.resolve()),
    "paired_steps": len(pairs),
    "raw_mean_abs": mean_abs(0),
    "gated_mean_abs": mean_abs(1),
    "commanded_mean_abs": mean_abs(2),
    "raw_to_gated_mean_abs_delta": [
        sum(abs(float(row[0][axis]) - float(row[1][axis])) for row in pairs) / len(pairs)
        for axis in range(4)
    ],
    "raw_to_gated_changed_steps": sum(
        any(abs(float(raw[axis]) - float(gated[axis])) > 1e-7 for axis in range(4))
        for raw, gated, _ in pairs
    ),
}
output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2))
PY
```

T64 requires reports from matched reviewed runs plus a written comparison of
their anchor, soil, and visual conditions. This remains a counterfactual claim,
not evidence that raw ACT would physically succeed.

### C65: final verdict

There is no automatic promotion command. Replace every placeholder and run in
J2:

```bash
export E52_FIELD_ROOT="$(cat /tmp/e52_field_root)"
cat > "${E52_FIELD_ROOT}/final_verdict.txt" <<'EOF'
decision=promote|remain_shadow|reject
accepted_tasks=<IDs>
rejected_tasks=<IDs>
reviewed_run_dirs=<paths>
known_failures=<summary>
gohome_status=not_tested|accepted|rejected
reviewer=<name>
date_utc=<timestamp>
EOF
```

## 10. Stop and shutdown commands

Normal supervised stop:

1. Press Ctrl+C once in J2 to stop the current shadow/control process and let
   it write the terminal trace.
2. Review `termination.json` and `trace_summary.json` before another run.
3. When the phase is complete, press Ctrl+C once in J1 to stop the managed
   bottom stack.

If J1 is detached or unhealthy, run in J2:

```bash
cd /media/mundane/D/Excavator_real_stack
./scripts/slave_real_stack.sh stop --force
./scripts/slave_real_stack.sh status
```

Use the physical emergency stop or power path immediately when software stop
is not appropriate. After any abnormal stop, preserve the logs and record the
observed reason; absence of a confirmed terminal zero is a failed trace, not a
reason to delete it.

## 11. Required output layout

Expected field-root evidence:

```text
e52_field_<timestamp>/
  jetson_repository.txt
  e52_bundle_preflight.json
  bundle_sha256_check.txt
  runtime_input_sha256.txt
  operator_signoff.txt
  imu_can_probe.json
  slave_stack_status.txt
  bridge_ports.txt
  shadow/
    <anchor_timestamp>/
      anchor_context.txt
      console.log
      <receiver_run>/metadata.json
      <receiver_run>/steps.jsonl
      <receiver_run>/summary.json
      <receiver_run>/camera_frames/
      <receiver_run>/e53_no_motion_report.json
  axis_response/
    T30_swing_positive.jsonl
    ...
    P3_review.txt
  P5_decision.txt
  raw_counterfactual_<receiver_run>.json
  final_verdict.txt
  control/
    <task_anchor>/
      e52_control_trace_<timestamp>/e52_bundle_preflight.json
      e52_control_trace_<timestamp>/<receiver_run>/metadata.json
      e52_control_trace_<timestamp>/<receiver_run>/steps.jsonl
      e52_control_trace_<timestamp>/<receiver_run>/summary.json
      e52_control_trace_<timestamp>/<receiver_run>/termination.json
      e52_control_trace_<timestamp>/<receiver_run>/camera_frames/
      e52_control_trace_<timestamp>/<receiver_run>/e52_trace_analysis/run_manifest.json
      e52_control_trace_<timestamp>/<receiver_run>/e52_trace_analysis/trace_summary.json
      e52_control_trace_<timestamp>/<receiver_run>/e52_trace_analysis/trace_context.json
      e52_control_trace_<timestamp>/<receiver_run>/e52_trace_analysis/trace_timeline.png
```

The host provenance file remains under
`/home/pingfan/Excavator_real_stack/artifacts/e52_field_preflight/` unless it is
copied into the field root after SSH is available.

## 12. Readiness boundary

- P0-P2 are executable only after C01 confirms exact Jetson file hashes.
- P3 is executable only with operator-approved command inputs.
- P4 tooling exists, but motion remains blocked until P0-P3 evidence is
  reviewed.
- P5-P6 are not blocked by missing trace code; they are blocked by missing
  lower-phase evidence.
- C62 gohome acceptance has no approved command and remains a separate gap.
- Command exit code alone never changes a task to `passed`; update the task CSV
  only after its required files and pass condition are reviewed.
