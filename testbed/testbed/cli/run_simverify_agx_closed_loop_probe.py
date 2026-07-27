"""Run a bounded, non-promotable SimVerify checkpoint against live AGX."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from testbed.policies.offline_eval import load_policy_for_episode
from testbed.simverify.agx_closed_loop_probe import (
    ExternalAgxWorker,
    ObservableDumpEndCommitDetector,
    ObservableReadyBoundaryDetector,
    external_git_provenance,
    load_action_prefix,
    run_bounded_closed_loop_probe,
    validate_probe_bundle,
)
from testbed.simverify.contracts import git_provenance
from testbed.simverify.habit_runtime_ready import (
    ObservableHabitReadyBoundaryDetector,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="tb-run-simverify-agx-closed-loop-probe")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--pact-root", type=Path, required=True)
    parser.add_argument("--unity-root", type=Path, required=True)
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--current-sector",
        choices=("left", "center", "right"),
        required=True,
    )
    parser.add_argument(
        "--next-sector",
        choices=("left", "center", "right"),
        required=True,
    )
    parser.add_argument(
        "--second-next-sector",
        choices=("left", "center", "right"),
    )
    parser.add_argument("--m0-root", type=Path)
    parser.add_argument(
        "--runtime-ready-root",
        type=Path,
        help=(
            "Accepted-v11 runtime ready calibration. May be used for a "
            "single-cycle endpoint check or with --second-next-sector."
        ),
    )
    parser.add_argument("--resnet18-checkpoint", type=Path)
    parser.add_argument(
        "--definition-root",
        type=Path,
        help=(
            "Frozen habit-definition artifacts; required by B1/B2 to causally "
            "activate the target only after observable dump-end"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="Inference RNG seed; paired diagnostics must keep this fixed",
    )
    parser.add_argument(
        "--action-prefix-policy-ticks",
        type=Path,
        help="Prior probe policy_ticks.jsonl providing actual sent actions",
    )
    parser.add_argument(
        "--action-prefix-count",
        type=int,
        help="Number of leading policy ticks to override from the shared prefix",
    )
    parser.add_argument("--policy-ticks", type=int, default=10)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5057)
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--inference-precision", default="fp32")
    parser.add_argument(
        "--action-selection",
        choices=(
            "legacy_temporal_aggregation",
            "recency_temporal_aggregation_diagnostic",
            "newest_chunk_head_diagnostic",
        ),
        default="legacy_temporal_aggregation",
    )
    parser.add_argument("--no-save-images", action="store_true")
    parser.add_argument("--allow-dirty-external", action="store_true")
    args = parser.parse_args()

    repository = args.repo_root.resolve(strict=True)
    current_git = git_provenance(repository)
    if (
        not current_git.get("git_available")
        or current_git.get("branch") != "v2.0.0-simVerify"
        or bool(current_git.get("dirty"))
    ):
        raise ValueError("AGX probe requires a clean v2.0.0-simVerify worktree")
    pact = external_git_provenance(args.pact_root)
    unity = external_git_provenance(args.unity_root)
    if (pact["dirty"] or unity["dirty"]) and not args.allow_dirty_external:
        raise ValueError(
            "external PACT/Unity checkout is dirty; pass "
            "--allow-dirty-external only for explicitly non-promotable diagnostics"
        )
    bundle = validate_probe_bundle(args.bundle_root)
    gated_condition = bundle["condition_input"] == (
        "cycle_condition_v1_dump_end_gated_low_dim"
    )
    if gated_condition != (args.definition_root is not None):
        raise ValueError(
            "--definition-root is required exactly for dump-end-gated B1/B2 bundles"
        )
    condition_commit_detector = (
        None
        if args.definition_root is None
        else ObservableDumpEndCommitDetector.from_definition_artifacts(
            definition_root=args.definition_root,
        )
    )
    prefix_arguments = (
        args.action_prefix_policy_ticks,
        args.action_prefix_count,
    )
    if any(value is not None for value in prefix_arguments) and not all(
        value is not None for value in prefix_arguments
    ):
        raise ValueError(
            "--action-prefix-policy-ticks and --action-prefix-count "
            "must be provided together"
        )
    action_prefix, action_prefix_provenance = (
        (None, None)
        if args.action_prefix_policy_ticks is None
        else load_action_prefix(
            args.action_prefix_policy_ticks,
            count=args.action_prefix_count,
        )
    )
    if args.m0_root is not None and args.runtime_ready_root is not None:
        raise ValueError("--m0-root and --runtime-ready-root are mutually exclusive")
    if args.runtime_ready_root is not None:
        if args.resnet18_checkpoint is None:
            raise ValueError(
                "--runtime-ready-root requires --resnet18-checkpoint"
            )
        ready_boundary_detector = (
            ObservableHabitReadyBoundaryDetector.from_calibration_artifacts(
                calibration_root=args.runtime_ready_root,
                weights_path=args.resnet18_checkpoint,
                device=args.device,
            )
        )
    elif args.m0_root is not None:
        if args.second_next_sector is None or args.resnet18_checkpoint is None:
            raise ValueError(
                "legacy --m0-root requires --second-next-sector and "
                "--resnet18-checkpoint"
            )
        ready_boundary_detector = ObservableReadyBoundaryDetector.from_m0_artifacts(
            m0_root=args.m0_root,
            resnet18_checkpoint=args.resnet18_checkpoint,
            device=args.device,
        )
    else:
        if args.second_next_sector is not None or args.resnet18_checkpoint is not None:
            raise ValueError(
                "--second-next-sector/--resnet18-checkpoint require "
                "--runtime-ready-root or legacy --m0-root"
            )
        ready_boundary_detector = None
    from testbed.policies.base import set_seed

    set_seed(args.policy_seed)
    policy = load_policy_for_episode(
        bundle_dir=args.bundle_root,
        ckpt_path=args.bundle_root / "policy_best.ckpt",
        resolved_config_path=None,
        stats_path=None,
        max_episode_len=8000,
        temporal_agg=True,
        temporal_aggregation_diagnostics=(
            args.action_selection != "legacy_temporal_aggregation"
        ),
        device=args.device,
        inference_precision=args.inference_precision,
    )
    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    worker_script = repository / "scripts/simverify_agx_env_worker.py"
    environment = ExternalAgxWorker(
        repo_root=repository,
        pact_root=args.pact_root,
        worker_script=worker_script,
        python_executable=sys.executable,
        host=args.host,
        port=args.port,
        timeout_s=args.timeout,
    )
    try:
        result = run_bounded_closed_loop_probe(
            policy=policy,
            environment=environment,
            output_root=args.output_root,
            bundle_contract=bundle,
            current_git=current_git,
            external_provenance={"pact": pact, "unity": unity},
            current_sector=args.current_sector,
            next_sector=args.next_sector,
            second_next_sector=args.second_next_sector,
            ready_boundary_detector=ready_boundary_detector,
            condition_commit_detector=condition_commit_detector,
            action_selection=args.action_selection,
            seed=args.seed,
            policy_seed=args.policy_seed,
            deterministic_inference=True,
            action_prefix=action_prefix,
            action_prefix_provenance=action_prefix_provenance,
            policy_ticks=args.policy_ticks,
            save_images=not args.no_save_images,
        )
    finally:
        environment.close()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
