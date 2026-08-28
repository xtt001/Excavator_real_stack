#!/usr/bin/env bash
# Resolve the accepted target-release bundle and log root for field scripts.
# This file is sourced by the runner and log checker.

real_transition_collect_external_roots() {
  REAL_TRANSITION_EXTERNAL_ROOTS=()
  local -a raw_roots=()
  local restore_nullglob=0

  if [[ -n "${EXCAVATOR_RUNTIME_SEARCH_ROOTS:-}" ]]; then
    IFS=: read -r -a raw_roots <<<"${EXCAVATOR_RUNTIME_SEARCH_ROOTS}"
  else
    if ! shopt -q nullglob; then
      shopt -s nullglob
      restore_nullglob=1
    fi
    raw_roots=(
      /media/*/EXTERNAL_USB*
      /run/media/*/EXTERNAL_USB*
      /mnt/EXTERNAL_USB*
    )
    if [[ "${restore_nullglob}" -eq 1 ]]; then
      shopt -u nullglob
    fi
  fi

  local -A seen=()
  local raw_root resolved_root
  for raw_root in "${raw_roots[@]}"; do
    [[ -n "${raw_root}" && -d "${raw_root}" ]] || continue
    resolved_root="$(realpath -- "${raw_root}")"
    if [[ -z "${seen[${resolved_root}]:-}" ]]; then
      seen["${resolved_root}"]=1
      REAL_TRANSITION_EXTERNAL_ROOTS+=("${resolved_root}")
    fi
  done
}

real_transition_resolve_runtime_paths() {
  local repo_root="$(realpath -- "$1")"
  local bundle_name="real_transition_target_release_v2"
  local explicit_bundle="${BUNDLE_DIR:-}"
  local explicit_log_root="${LOG_ROOT:-}"
  local resolved_bundle=""
  local resolved_drive=""
  local bundle_source=""

  real_transition_collect_external_roots

  if [[ -n "${explicit_bundle}" ]]; then
    resolved_bundle="$(realpath -m -- "${explicit_bundle}")"
    bundle_source="explicit"
    local external_root
    for external_root in "${REAL_TRANSITION_EXTERNAL_ROOTS[@]}"; do
      case "${resolved_bundle}/" in
        "${external_root}/"*)
          resolved_drive="${external_root}"
          break
          ;;
      esac
    done
  else
    local -a bundle_candidates=()
    local -a candidate_drives=()
    local -A seen_candidates=()
    local restore_nullglob=0
    local external_root candidate resolved_candidate

    if ! shopt -q nullglob; then
      shopt -s nullglob
      restore_nullglob=1
    fi
    for external_root in "${REAL_TRANSITION_EXTERNAL_ROOTS[@]}"; do
      for candidate in \
        "${external_root}/Excavator_real_stack_runtime/${bundle_name}/policy_bundles/${bundle_name}" \
        "${external_root}"/Excavator_real_stack_runtime/${bundle_name}_*/policy_bundles/${bundle_name}; do
        [[ -f "${candidate}/policy_accepted.ckpt" ]] || continue
        resolved_candidate="$(realpath -- "${candidate}")"
        if [[ -z "${seen_candidates[${resolved_candidate}]:-}" ]]; then
          seen_candidates["${resolved_candidate}"]=1
          bundle_candidates+=("${resolved_candidate}")
          candidate_drives+=("${external_root}")
        fi
      done
    done
    if [[ "${restore_nullglob}" -eq 1 ]]; then
      shopt -u nullglob
    fi

    if [[ "${#bundle_candidates[@]}" -eq 1 ]]; then
      resolved_bundle="${bundle_candidates[0]}"
      resolved_drive="${candidate_drives[0]}"
      bundle_source="external"
    elif [[ "${#bundle_candidates[@]}" -gt 1 ]]; then
      echo "Multiple target-release runtime bundles were found; set BUNDLE_DIR explicitly:" >&2
      printf '  %s\n' "${bundle_candidates[@]}" >&2
      return 2
    elif [[ -f "${repo_root}/policy_bundles/${bundle_name}/policy_accepted.ckpt" ]]; then
      resolved_bundle="${repo_root}/policy_bundles/${bundle_name}"
      bundle_source="local"
    else
      echo "No accepted target-release runtime bundle was found." >&2
      echo "Expected it below an inserted drive at:" >&2
      echo "  Excavator_real_stack_runtime/${bundle_name}_*/policy_bundles/${bundle_name}" >&2
      echo "Set BUNDLE_DIR only when intentionally overriding automatic discovery." >&2
      return 2
    fi
  fi

  if [[ ! -f "${resolved_bundle}/policy_accepted.ckpt" ]]; then
    echo "Resolved runtime bundle has no policy_accepted.ckpt: ${resolved_bundle}" >&2
    return 2
  fi

  local resolved_log_root
  if [[ -n "${explicit_log_root}" ]]; then
    resolved_log_root="$(realpath -m -- "${explicit_log_root}")"
  elif [[ -n "${resolved_drive}" ]]; then
    resolved_log_root="${resolved_drive}/policy_control_tests"
  elif [[ "${#REAL_TRANSITION_EXTERNAL_ROOTS[@]}" -eq 1 ]]; then
    resolved_drive="${REAL_TRANSITION_EXTERNAL_ROOTS[0]}"
    resolved_log_root="${resolved_drive}/policy_control_tests"
  else
    resolved_log_root="${repo_root}/artifacts/policy_control_tests"
  fi

  REAL_TRANSITION_BUNDLE_DIR="${resolved_bundle}"
  REAL_TRANSITION_LOG_ROOT="${resolved_log_root}"
  REAL_TRANSITION_DRIVE_ROOT="${resolved_drive}"
  REAL_TRANSITION_BUNDLE_SOURCE="${bundle_source}"
}
