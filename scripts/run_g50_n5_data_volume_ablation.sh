#!/usr/bin/env bash
set -euo pipefail

REPO=/home/pingfan/Excavator_real_stack_e52_deadlock_eval
RUN_ROOT=/data/pingfan/Excavator_real_stack_data/runs/g50_n5_data_volume_ablation_20260715
DATASET=/data/pingfan/Excavator_real_stack_data/g48_new_trainval_view_v1
VAL_MANIFEST="$REPO/testbed/testbed/configs/manifests/g50_g49_validation20.json"
POLICY_CONFIG="$REPO/testbed/testbed/configs/policy_real_gmsl_fourcam_camera_role_eval.yaml"
EVENT_DIR=/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/expert_intent_events_h40_train120_val20
DEADZONE_JSON=/data/pingfan/Excavator_real_stack_data/runs/g49_new_data_first_batch_20260714/evaluation/n0_state_hold_raw_val20_h20/resolved_direct_output_deadzone.json

cd "$REPO"
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/evaluation"

metadata_status() {
  local metadata=$1
  if [[ ! -f "$metadata" ]]; then
    printf '%s\n' missing
    return
  fi
  python -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "$metadata"
}

wait_for_n20() {
  local metadata="$RUN_ROOT/training/n20_equal_steps/ckpt/run_metadata.json"
  while true; do
    local status
    status=$(metadata_status "$metadata")
    if [[ "$status" == completed ]]; then
      return
    fi
    if [[ "$status" == failed ]]; then
      printf 'N20 training metadata reports failed\n' >&2
      exit 1
    fi
    if ! pgrep -f 'python -m testbed.cli.train --config testbed/testbed/configs/act_real_gmsl_fourcam_g50_n5_data20_equal_steps.yaml' >/dev/null; then
      printf 'N20 training is not running and is not complete (status=%s)\n' "$status" >&2
      exit 1
    fi
    sleep 30
  done
}

run_train() {
  local size=$1
  local config="testbed/testbed/configs/act_real_gmsl_fourcam_g50_n5_data${size}_equal_steps.yaml"
  local metadata="$RUN_ROOT/training/n${size}_equal_steps/ckpt/run_metadata.json"
  local status
  status=$(metadata_status "$metadata")
  if [[ "$status" == completed ]]; then
    return
  fi
  if [[ "$status" != missing ]]; then
    printf 'Refusing ambiguous N%s output with metadata status=%s\n' "$size" "$status" >&2
    exit 1
  fi
  PYTHONPATH=.:testbed python -m testbed.cli.train --config "$config" \
    2>&1 | tee "$RUN_ROOT/logs/n${size}_train.log"
  [[ $(metadata_status "$metadata") == completed ]]
}

run_eval() {
  local size=$1
  local bundle="$RUN_ROOT/training/n${size}_equal_steps/ckpt"
  local eval_root="$RUN_ROOT/evaluation/n${size}"
  local model="G50_N${size}"
  local -a episode_args=()
  local episode_id
  for episode_id in {10120..10139}; do
    episode_args+=(--episode-id "episode_${episode_id}")
  done

  if [[ ! -f "$eval_root/open_loop_val20/collection_summary.json" ]]; then
    PYTHONPATH=.:testbed python -m testbed.cli.offline_policy_eval \
      --bundle-dir "$bundle" \
      --dataset-dir "$DATASET" \
      --manifest "$VAL_MANIFEST" \
      --all-train-ready \
      --output-dir "$eval_root/open_loop_val20" \
      --device cuda \
      --progress-every 200 \
      2>&1 | tee "$RUN_ROOT/logs/n${size}_open_loop.log"
  fi

  if [[ ! -f "$eval_root/startup_activation_val20_h20/startup_activation_report.json" ]]; then
    PYTHONPATH=.:testbed python -m testbed.cli.offline_startup_activation \
      --model "$model" \
      --config "$POLICY_CONFIG" \
      --bundle-dir "$bundle" \
      --dataset-dir "$DATASET" \
      --event-dir "$EVENT_DIR" \
      --deadzone-json "$DEADZONE_JSON" \
      --hold-horizon-steps 20 \
      --sampling-hz 20 \
      --output-dir "$eval_root/startup_activation_val20_h20" \
      --device cuda \
      2>&1 | tee "$RUN_ROOT/logs/n${size}_startup_activation.log"
  fi

  if [[ ! -f "$eval_root/state_hold_raw_val20_h20/run_summary.json" ]]; then
    PYTHONPATH=.:testbed python -m testbed.cli.offline_state_hold_liveness \
      --config "$POLICY_CONFIG" \
      --bundle-dir "$bundle" \
      --pipeline-mode raw \
      --candidate-id "g50_n${size}_equal_steps" \
      --dataset-dir "$DATASET" \
      "${episode_args[@]}" \
      --hold-horizon-steps 20 \
      --trace-full-horizon-after-recovery \
      --output-dir "$eval_root/state_hold_raw_val20_h20" \
      --device cuda \
      --assist-mode disabled \
      2>&1 | tee "$RUN_ROOT/logs/n${size}_state_hold.log"
  fi

  if [[ ! -f "$eval_root/expert_intent_val20_h40/expert_intent_eval_report.json" ]]; then
    PYTHONPATH=.:testbed python -m testbed.cli.evaluate_expert_intent \
      --eval "$model=$eval_root/open_loop_val20" \
      --event-dir "$EVENT_DIR" \
      --deadzone-json "$DEADZONE_JSON" \
      --output-dir "$eval_root/expert_intent_val20_h40" \
      --split validation \
      --sampling-hz 20 \
      2>&1 | tee "$RUN_ROOT/logs/n${size}_expert_intent.log"
  fi
}

wait_for_n20
run_train 40
run_train 80
run_eval 20
run_eval 40
run_eval 80
