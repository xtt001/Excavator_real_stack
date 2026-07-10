# Policy Gate Experiments Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Find a policy training and handoff scheme that is good enough by three gates: action closeness, action intent / deadzone-crossing ability, and tail stop stability.

**Architecture:** Keep training experiments, offline replay artifacts, deadzone gates, and handoff label work tied to one durable ledger. Every model is evaluated with the same train-ready manifest, deadzone table, and window definitions before it is considered better. Gohome awareness is treated as conservative end-of-cycle eligibility, not as gohome trajectory imitation.

**Tech Stack:** Python, PyTorch ACT policy, HDF5 datasets, `testbed.cli.train`, `testbed.cli.offline_policy_eval`, `scripts/deadzone_window_eval.py`, repo-local `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed`.

---

## Reference Base

- User objective: improve real excavator policy behavior using offline training and gates before live testing.
- Primary metrics:
  1. Action closeness: all-train-ready replay MAE/RMSE and per-axis errors.
  2. Action intent: policy actions must cross directional deadzones in the same axis/direction as the expert in startup and main-motion windows.
     Deadzone-aware training must therefore both promote effective same-direction motion when the expert moves and suppress effective motion when the expert is idle; it is not only an idle-suppression loss.
  3. Stop stability: after human action stops and before gohome, policy must not cross effective deadzones or create unsafe extra motion.
- Safety preference: late gohome is acceptable if the policy is stable; early gohome is unsafe.
- Handoff semantics: model should learn `gohome_eligible`, meaning "candidate state where runtime may request gohome", not "must press gohome now" and not "imitate gohome trajectory".
- Existing protocol sources:
  - `docs/policy_model_effect_eval_protocol.md`
  - `docs/policy_to_gohome_handoff_notes_20260616.md`
  - `docs/gmsl_hfov110_act_texture_generalization_risks_20260630.md`
- Current target lock:
  - Repo: `/home/pingfan/Excavator_real_stack`
  - Branch: `fs/online-qc-dev`
  - Reference HEAD at plan creation: `31eab44`
  - Dataset root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104`
  - Train-ready manifest: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz/qc_batch_ref_72_87/train_ready_manifest.json`
  - Deadzone table: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/deadzone_policy_raw_for_runtime_scale.json`
- Non-goals:
  - Do not direct-connect to Jetson from this machine.
  - Do not promote any model to live control from MAE alone.
  - Do not train on gohome automation actions as ordinary policy actions.
  - Do not use fixed qpos envelopes as the main gohome eligibility definition; they may only be conservative runtime guards.

## Current Artifacts

| Artifact | Path |
| --- | --- |
| 20Hz train-ready dataset | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz` |
| raw HDF5 copy | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104` |
| handoff W30 census | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_request_20hz_w30` |
| handoff eligibility dwell10 dataset | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_eligibility_20hz_dwell10` |
| deadzone intent census | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/deadzone_intent_census_train_ready_runtime_scaled` |
| four-model deadzone gate | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/deadzone_gate_four_models_runtime_scaled` |
| E15 low-dimensional intent probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e15_intent_probe` |
| E27 four-view visual domains | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/visual_domain_clusters_four_k6` |
| E27 eye2 visual domains | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/visual_domain_clusters_eye2_k6` |
| E27 domain-5 held-out split | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/splits/visual_domain_four_k6_domain5_heldout.yaml` |
| E27 held-out manifest | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/manifests/e27_domain5_heldout_manifest.json` |
| E27 train-domain manifest | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/manifests/e27_domain5_train_domain_manifest.json` |
| E36 candidate package manifest | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e36_e34_runtime_gate_candidate` |
| E37 full ACT gate smoke | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e37_full_act_gate_smoke_ep73_ep79` |
| E38 full ACT all-train-ready gate smoke | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e38_full_act_gate_smoke_all_train_ready` |
| E39 startup failure diagnostics | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e39_startup_failure_diagnostics` |
| E40 deadzone snap calibration probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e40_deadzone_snap_probe` |
| E41 intent-targeted snap calibration probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e41_intent_targeted_snap_probe` |
| E42 startup intent-overlap diagnostic | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e42_startup_intent_overlap_diagnostic` |
| E43 direction release gate probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e43_direction_gate_probe` |
| E44 E41 snap + E43 direction release probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e44_e41snap_e43_direction_release_probe` |
| E45 temporal release training | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e45_temporal_release_eye2` |
| E45 all-train-ready replay | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e45_temporal_release_eye2_all_train_ready_best` |
| E45 startup/tail gate | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e45_temporal_release_eye2_gate_runtime_scaled` |
| E46 weak temporal release training | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e46_temporal_release_weak_eye2` |
| E46 all-train-ready replay | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e46_temporal_release_weak_eye2_all_train_ready_best` |
| E46 startup/tail gate | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e46_temporal_release_weak_eye2_gate_runtime_scaled` |
| E47 temporal direction gate probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e47_temporal_direction_gate_probe` |
| E47b temporal direction gate s75 probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e47b_temporal_direction_gate_probe_s75` |
| E47b reference window comparison | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e47b_t50_s75_vs_refs_deadzone_window_eval` |
| E48 non-causal E47b + E33 combined candidate | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e48_e47b_action_e33_gohome_combined_candidate` |
| E49 causal temporal direction gate s75 probe | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e49_causal_temporal_direction_gate_probe_s75` |
| E49 causal reference window comparison | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e49_causal_t50_s75_vs_refs_deadzone_window_eval` |
| E50 causal E49 + E33 combined candidate | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e50_e49_causal_action_e33_gohome_combined_candidate` |
| E51 full-ACT causal temporal gate smoke | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready` |
| E51 full-ACT startup/tail gate | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready_deadzone_runtime_scaled` |
| E51 full-ACT window comparison | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_fullact_vs_e38_e49_deadzone_window_eval` |
| E52 causal temporal package manifest | `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e52_e51_causal_temporal_gate_candidate` |

## Gate Definitions

| Gate | Decision Role | Required Evidence |
| --- | --- | --- |
| Action closeness | Reject regressions in imitation quality | `collection_summary.json`, `episode_metrics.csv`, per-axis MAE/RMSE/bias |
| Startup intent | Check whether policy can begin real machine motion | first expert-effective + 40 step gate, same-axis same-direction coverage, wrong/extra rate |
| Main-motion intent | Check whether policy preserves effective task motion | longest expert-effective segment gate |
| Tail stop stability | Check whether late gohome would be safe | `t_stop -> t_go` deadzone crossing rate, policy max abs in tail |
| Visual sensitivity | Check texture overfit and view dependence | fixed qpos / multi-FPV swap gate, domain-held-out split when available |
| Gohome awareness | Learn candidate gohome eligibility only | event-level precision/recall/delay; no automation action imitation |

## Deadzone Intent Contract

Deadzone-aware training is an intent-matching objective, not a generic small-action penalty or a tail-only quieting trick. The first question is whether the policy crosses the relevant directional deadzone when the expert intended motion; only after that should we reward staying quiet in stop windows.

Positive intent:
- If the expert action would produce real machine motion on an axis/direction, the policy should cross the same directional deadzone with enough margin.
- A model that matches MAE but stays inside the deadzone in these windows fails the startup/main-motion intent gate.

Negative intent:
- If the expert is inactive in the step, the policy should stay inside all directional deadzones.
- If the expert is active in one axis/direction, policy crossings on other axes or the opposite direction count as extra/wrong intent.

The loss and gates therefore need three separate terms: same-direction promotion, idle suppression, and wrong/extra suppression. A deadzone-aware candidate cannot pass just because tail output is quiet; it must also preserve or improve startup/main should-move coverage. Any future deadzone-aware variant must report all three loss terms, plus startup, main-motion, and tail window summaries.

Clarification from E16 discussion:
- Deadzone-aware means matching when the model should move and when it should not move.
- It should promote above-deadzone same-axis/same-direction action in should-move frames, not only suppress action in should-stop frames.
- It must avoid indiscriminate "move more" behavior: a crossing in the wrong window, wrong axis, or wrong direction is still a failure even if it proves the model can cross the deadzone.

Current train-ready intent census under runtime-scaled deadzones:
- 24 episodes, 16,529 total 20Hz steps.
- `should_move`: 9,812 frames, 59.36%.
- `should_stop`: 6,717 frames, 40.64%.
- Multi-direction move frames: 589, 6.00% of should-move frames; most move labels are single-axis/single-direction.
- Axis/direction event rates over all steps: swing pos 13.41%, swing neg 13.69%, boom pos 12.31%, boom neg 6.71%, bucket pos 6.73%, bucket neg 10.07%, stick pos/neg 0.00%.
- Implication for E14: balance move vs stop and direction classes, but do not train stick deadzone promotion from this batch because the expert never crosses the runtime-scaled stick threshold.

## Completed Experiment Ledger

Reporting convention: every new experiment should be reported with a human-readable title first, then the short ID in parentheses. The ID is only for artifact lookup.

| ID | Model | Views | Training Change | Best Epoch | Best Val Loss | Replay MAE | Startup Policy Effective | Startup Same Dir | Startup Extra/Wrong | Main Policy Effective | Main Same Dir | Main Extra/Wrong | Tail Effective | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E00 | baseline_qpos_no_transform | video4,5,6,7 | none | 1255 | 0.117843 | 0.0445 | 70.0% | 74.7% | 16.1% | 88.1% | 85.9% | 3.1% | 0.0% | Keep as 4-view baseline |
| E01 | baseline_qpos_no_transform_eye2 | video4,5 | none | 1750 | 0.100744 | 0.0473 | 71.9% | 76.6% | 16.2% | 96.6% | 96.5% | 1.1% | 0.0% | Strong intent baseline despite worse MAE |
| E02 | baseline_qpos_no_transform_deadzone_four_camera | video4,5,6,7 | deadzone same-dir loss + wrong penalty | 1960 | 0.124359 | 0.0430 | 60.7% | 66.2% | 12.9% | 97.3% | 95.0% | 3.2% | 0.0% | Improves main motion, hurts startup |
| E03 | baseline_qpos_no_transform_deadzone_eye2 | video4,5 | deadzone same-dir loss + wrong penalty | 1235 | 0.100853 | 0.0475 | 44.4% | 44.9% | 20.4% | 96.7% | 96.5% | 1.2% | 0.0% | Reject as-is; startup too weak |
| E04a | baseline_qpos_no_transform_motion_intent_deadzone_four_camera | video4,5,6,7 | same-dir promote + idle suppress + wrong suppress | 1960 | 0.111353 | 0.0414 | 60.6% | 63.9% | 16.2% | 95.7% | 94.1% | 2.5% | 0.0% | Best MAE so far, but startup still below baseline |
| E04b | baseline_qpos_no_transform_motion_intent_deadzone_eye2 | video4,5 | same-dir promote + idle suppress + wrong suppress | 1995 | 0.072163 | 0.0485 | 54.1% | 55.5% | 16.9% | 96.5% | 96.3% | 1.2% | 0.0% | Recovers part of broad-loss startup regression, still below eye2 baseline |
| E10a | baseline_qpos_no_transform_promote_biased_deadzone_four_camera | video4,5,6,7 | promote-biased deadzone: same 0.25, idle 0.01, wrong 0.02 | 1915 | 0.121568 | 0.0534 | 65.0% | 74.9% | 9.7% | 97.6% | 96.2% | 2.3% | 0.0% | Better startup than E04, but MAE/p95 too poor and still below baseline startup |
| E10b | baseline_qpos_no_transform_promote_biased_deadzone_eye2 | video4,5 | promote-biased deadzone: same 0.25, idle 0.01, wrong 0.02 | 1415 | 0.098275 | 0.0500 | 59.4% | 59.0% | 20.4% | 97.2% | 97.1% | 1.2% | 0.0% | Reject; startup still below eye2 baseline and extra/wrong high |
| E12a | baseline_qpos_downsample060_eye2, infer downsample060 | video4,5 | train/infer deterministic downsample_060, no deadzone loss | 1520 | 0.100795 | 0.0523 | 63.0% | 71.3% | 11.1% | 96.8% | 96.7% | 1.1% | 0.0% | Reject; main/tail ok but startup and MAE below eye2 baseline |
| E12b | baseline_qpos_downsample060_eye2, infer none | video4,5 | train downsample_060, infer original images | 1520 | 0.100795 | 0.0546 | 34.1% | 34.0% | 4.4% | 96.7% | 96.3% | 1.3% | 0.0% | Reject; transform mismatch badly hurts startup |
| E13a | baseline_qpos_downsample080_eye2, infer downsample080 | video4,5 | train/infer deterministic downsample_080, no deadzone loss | 1645 | 0.107668 | 0.0500 | 43.5% | 45.5% | 13.6% | 96.5% | 95.1% | 2.3% | 0.0% | Reject; lighter low-pass still collapses startup |
| E13b | baseline_qpos_downsample080_eye2, infer none | video4,5 | train downsample_080, infer original images | 1645 | 0.107668 | 0.0536 | 30.3% | 31.3% | 6.4% | 96.2% | 95.5% | 1.6% | 0.0% | Reject; original-image inference is worse than consistent transform |
| E14a | baseline_qpos_no_transform_transition_deadzone_eye2 | video4,5 | same-dir promotion only in expert transition windows; idle/wrong suppression unchanged | 1960 | 0.100651 | 0.0447 | 58.0% | 61.0% | 15.4% | 95.1% | 94.9% | 1.2% | 0.0% | Reject; tail is quieter and MAE is good, but startup intent is still far below eye2 baseline |
| E15b | baseline_qpos_no_transform_intent_head_eye2 | video4,5 | auxiliary 8-way intent head, unweighted BCE, no action deadzone loss | 1915 | 0.119857 | 0.0420 | 39.4% | 40.1% | 13.2% | 95.1% | 94.8% | 1.3% | 0.0% | Reject; best MAE so far but startup collapses and intent head predicts no-move everywhere |
| E16 | baseline_qpos_no_transform_weighted_intent_head_eye2 | video4,5 | auxiliary 8-way intent head with positive_weight 8.0, no action deadzone loss | 1805 | 0.081496 | 0.0448 | 67.3% | 80.4% | 6.9% | 95.5% | 95.2% | 1.2% | 0.0% | Useful signal, not final; fixes all-negative head and improves direction quality, but still below E01 startup effective and has high any-move false positives |
| E17 | baseline_qpos_no_transform_e17_balanced_deadzone_eye2 | video4,5 | E16 weighted intent head + frequency-sensitive idle/wrong deadzone loss | 1805 | 0.094252 | 0.0394 | 48.1% | 52.6% | 9.3% | 94.4% | 94.3% | 1.1% | 0.0% | Reject despite best MAE; too much should-move startup suppression |
| E18a | baseline_qpos_no_transform_e18a_recall_priority_deadzone_eye2 | video4,5 | E16 weighted intent head + recall-priority deadzone loss: same 0.12, idle 0.02, wrong 0.03 | 1730 | 0.104615 | 0.0435 | 59.8% | 64.9% | 12.4% | 92.7% | 91.9% | 1.6% | 0.0% | Reject as final; recovers part of E17 startup loss but still worse than E16/E01 and weakens main motion |
| E25 | baseline_qpos_random_downsample060100_weighted_intent_head_eye2 | video4,5 | E16 weighted intent head plus stochastic train-time `random_downsample_060_100_seed25` | 1695 | 0.114224 | 0.0556 | 49.1% | 51.1% | 14.5% | 94.5% | 93.5% | 2.1% | 0.0% | Reject as final; recovers low-pass main motion but hurts original startup/MAE and creates low-pass tail crossings |
| E26 | baseline_qpos_random_downsample080100_weighted_intent_head_eye2 | video4,5 | E16 weighted intent head plus stochastic train-time `random_downsample_080_100_seed26` | 1695 | 0.092107 | 0.0494 | 45.5% | 47.7% | 13.6% | 93.8% | 93.7% | 1.1% | 0.0% | Reject as final; narrower augmentation improves MAE vs E25 but further weakens startup and still has low-pass tail crossings |
| E27 | e27_domain5_heldout_weighted_intent_head_eye2 | video4,5 | E16-style weighted intent head, four-view `texture_domain_5` held out from training | 1750 | 0.122229 | 0.0488 | 50.6% | 54.0% | 14.3% | 96.8% | 96.2% | 1.5% | 0.0% | Reject as candidate; domain-held-out split exposes visual-domain MAE and extra/wrong risk, while all-data startup is far below E16/E22b |
| E28 | e28_weighted_intent_head_four_camera | video4,5,6,7 | Four-view E16-style weighted intent head, no action deadzone loss | 1999 | 0.084474 | 0.0433 | 62.0% | 69.2% | 10.3% | 96.8% | 94.3% | 3.3% | 1.2% | Reject as candidate; MAE/RMSE close to E22b but startup is weaker and tail has 6 / 504 effective frames |
| E29 | e29_e28_phase_gate_soft_scale_probe/hyst_o0.25_c0.10_s0.75 | video4,5,6,7 | E28 plus learned phase gate, inactive scale 0.75 | n/a | n/a | 0.0404 | 59.9% | 68.5% | 8.1% | 95.9% | 93.5% | 3.3% | 0.2% | Reject as candidate; best MAE/RMSE so far but startup/main intent and tail are worse than E22b |
| E29b | e29b_e28_phase_gate_simple015_s050_probe/simple_0.15_s0.50 | video4,5,6,7 | E28 plus E22b-style phase gate `simple_0.15_s0.50` | n/a | n/a | 0.0413 | 58.9% | 67.3% | 7.9% | 96.3% | 93.9% | 3.3% | 0.2% | Reject as candidate; same scale as E22b still leaves weaker startup and 1 / 504 tail crossing |
| E30 | e30_window_deadzone_intent_eye2 | video4,5 | E16 weighted intent head + handoff-mask action stats + window-conditioned deadzone intent loss | 1155 | 0.125042 | 0.0604 | 51.8% | 53.4% | 17.6% | 97.2% | 96.3% | 1.8% | 2.0% | Reject; start40 is quiet, but action closeness, startup movement intent, and tail stop stability all regress |
| E31 | e31_masked_handoff_intent_eye2 | video4,5 | E16 weighted intent head + handoff-mask action stats/action-loss mask, no window deadzone loss | 1935 | 0.105888 | 0.0500 | 45.7% | 49.3% | 10.1% | 95.4% | 93.9% | 2.4% | 3.2% | Reject; removing window loss recovers MAE vs E30 but startup and tail are still worse than E22b, so the mask-aware handoff path is not a free improvement |
| E45 | e45_temporal_release_eye2 | video4,5 | E16 weighted intent head + same-direction temporal release persistence penalty | 1999 | 0.082703 | 0.0448 | 58.9% | 65.4% | 11.1% | n/a | n/a | n/a | 1.2% | Reject; training-time release penalty worsens startup, extra/wrong, and tail versus E38/E41/E44 |
| E46 | e46_temporal_release_weak_eye2 | video4,5 | E16 weighted intent head + weaker temporal release penalty | 1580 | 0.110259 | 0.0425 | 49.0% | 52.0% | 11.8% | n/a | n/a | n/a | 0.0% | Reject; weak release recovers tail but collapses startup should-move intent |

## Gohome Eligibility Probe Ledger

| ID | Probe | Inputs | Gate | Event Recall | Early FP Episodes | Pre-tail FP Episodes | Pre-tail FP Frames | Mean Detection Delay | Mean Steps Before t_go | Decision |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E32 | e32_gohome_eligibility_probe | E16 intent probabilities + qpos/qvel | `thr_0.95_c5` | 95.8% | 79.2% | 16.7% | 5 | 1.57 steps | 13.61 steps | Not deployable as a standalone request gate; promising signal exists, but it needs a conservative tail/candidate-phase guard or stronger labels before runtime use |
| E33 | e33_two_stage_gohome_gate_probe | learned tail candidate + E32 eligibility | `tail0.97_c10 && elig0.80_c3` | 95.8% | 25.0% | 0.0% | 0 | 2.87 steps | 12.30 steps | Best gohome-awareness probe so far; two-stage learned gate removes unsafe pre-tail triggers while preserving E32 event recall |

## Combined Offline Candidate Ledger

| ID | Action Candidate | Gohome Candidate | Action MAE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Main Effective | Main Same Dir | Main Extra/Wrong | Tail Effective Frames | Gohome Recall | Gohome Pre-tail FP | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E34 | E22b `simple_0.15_s0.50` | E33 `tail0.97_c10 && elig0.80_c3` | 0.0414 | 65.1% | 78.6% | 5.7% | 94.8% | 94.5% | 1.2% | 0 | 95.8% | 0 / 24 episodes, 0 frames | Best integrated offline candidate so far; package as action policy plus auxiliary gohome request gate for runtime/replay smoke |
| E48 | E41 `snap_m0200_i70` + non-causal temporal gate `tdir_t50_s75` | E33 `tail0.97_c10 && elig0.80_c3` | 0.0406 | 67.3% | 81.4% | 5.3% | 95.3% | 95.2% | 1.1% | 0 | 95.8% | 0 / 24 episodes, 0 frames | Offline upper-bound diagnostic only; E47b uses future context offsets and is not real-time deployable |
| E50 | E41 `snap_m0200_i70` + causal temporal gate `tdir_t50_s75` | E33 `tail0.97_c10 && elig0.80_c3` | 0.0408 | 67.9% | 81.9% | 5.8% | 95.3% | 95.2% | 1.1% | 0 | 95.8% | 0 / 24 episodes, 0 frames | Best real-time-causal action+gohome offline candidate so far, but full-window extra/wrong is not improved versus E38 |

## Runtime/Replay Smoke Ledger

| ID | Candidate | Scope | Action MAE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Tail Effective Frames | Gohome Recall | Gohome Pre-tail FP | Gate Latency P95 | Decision |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E35 | E34 final gate models | final phase-gate model + final tail/eligibility models on cached features | 0.0414 | 64.9% | 78.4% | 5.6% | 0 | 95.8% | 0 / 24 episodes, 0 frames | 0.035 ms for 3 gate MLPs | Pass lightweight runtime gate smoke; next blocker is full ACT image inference package and visual-domain robustness |
| E37 | E36 packaged full ACT path | full ACT image inference + same-frame intent + final gates on ep73/ep79 | 0.0354 | 77.5% | 96.7% | 0.0% | 0 | 100.0% | 0 / 2 episodes, 0 frames | 0.0006 ms batched gate eval | Pass sampled full-ACT smoke; not a full-dataset or visual-domain robustness result |
| E38 | E36 packaged full ACT path | full ACT image inference + same-frame intent + final gates on all 24 train-ready episodes | 0.0414 | 64.9% | 78.4% | 5.6% | 0 | 95.8% | 0 / 24 episodes, 0 frames | ACT p95 20.3 ms in offline HDF5 loop | Pass full-dataset full-ACT mechanics; visual-domain/startup weak spots remain |
| E51 | E36 packaged ACT + E41 snap + final causal temporal direction gate + E33 gohome | full ACT image inference on all 24 train-ready episodes | 0.0408 | 68.4% | 82.6% | 5.7% | 0 | 95.8% | 0 / 24 episodes, 0 frames | ACT p95 20.3 ms; gate p95 0.0075 ms | Best runtime-smoke candidate so far; needs package manifest that includes the temporal gate model/evidence |
| E54 | E52/E51 candidate on 6 QC-excluded episodes | full ACT image inference on episodes 72/77/81/95/101/103 with separately generated dwell10 handoff labels | 0.0548 | 69.2% | 79.7% | 7.5% | 0 | 83.3% | 0 / 6 episodes, 0 frames | ACT p95 20.2 ms; gate p95 0.0063 ms | Useful stress validation: action closeness degrades and one no-positive gohome label episode is missed, but tail stop and pre-tail gohome safety still pass |

## Package Ledger

| ID | Candidate | Scope | Artifacts Checked | SHA Verification | Action MAE | Gohome Recall | Gohome Pre-tail FP | Gate Latency P95 | Decision |
| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| E36 | E35/E34 package manifest | action checkpoint bundle + phase gate + gohome gates + E34/E35 evidence | 25 | pass | 0.0414 | 95.8% | 0 / 24 episodes, 0 frames | 0.035 ms | Pass packaging/verifier gate; this is a traceability artifact, not a new policy |
| E52 | E51 causal temporal package manifest | E36 ACT/phase/gohome base plus E49 temporal model and E50/E51 evidence | 23 | pass | 0.0408 | 95.8% | 0 / 24 episodes, 0 frames | 0.0075 ms | Best package artifact so far; still requires field-side integration wiring before live motion |

## Field Integration / No-Motion Ledger

| ID | Scope | Evidence Checked | No-Motion Gate | Policy-Running Gate | Decision |
| --- | --- | --- | --- | --- | --- |
| E53 | `shadow_zero` dry-run log verifier and gate-stack diagnostics pass-through | `steps.jsonl` fields for raw/scaled/assisted/returned policy action, optional phase/snap/temporal gate action, safe action, commanded action, gohome request probabilities, output mode, and policy errors | Requires `safe_action`, `commanded_action`, and `policy_returned_action` to stay within tolerance | Requires non-zero finite `policy_action` on at least the configured number of steps, so an all-zero log cannot pass as a policy test | Pass as an offline verifier/tooling gate; still needs a real field-side `shadow_zero` run log before live motion |

## Diagnostic Ledger

| ID | Scope | Key Finding | Decision |
| --- | --- | --- | --- |
| E39 | E38 startup weak episodes, raw ACT vs phase-gated action | Phase gate is not the main cause: `episode_74` raw bucket+ is just below the deadzone threshold, `episode_100/90` over-extend bucket+, and `episode_98` under-shoots boom- | Next work should target action amplitude/timing calibration, not another phase-gate-only threshold sweep |
| E40 | E38 phase-gated replay, post-hoc near-deadzone action snap | Snapping phase-active actions just below runtime deadzones improves startup effective/same-dir without changing MAE materially, but larger margins also increase extra/wrong and do not solve over-duration cases | Keep as diagnostic evidence and possible calibration primitive; do not deploy global snap without direction/window targeting |
| E41 | E40 snap plus same-frame intent-probability direction filter | Intent-targeted snap keeps tail at 0 / 504 and improves startup with less extra/wrong than global snap; best extra<=6.0 scan is `snap_m0200_i70` at 68.5% effective / 82.6% same-dir / 5.9% extra | Better calibration primitive than E40 global snap, but still not final because over-duration episodes remain |
| E47/E47b | E41 `snap_m0200_i70` plus non-causal temporal-context direction gate | Temporal features over intent/qpos/qvel can reduce extra/wrong while preserving tail; `tdir_t50_s75` gives MAE 0.0406, startup 67.3% / 81.4% / 5.3%, main 95.3% / 95.2% / 1.1%, tail 0 | Useful offline upper bound, but not runtime-ready because the default offsets include future frames |
| E49/E50 | E41 `snap_m0200_i70` plus causal-only temporal-context direction gate | Using only offsets `[-10, -5, -2, -1, 0]`, `tdir_t50_s75` gives MAE 0.0408, startup 67.9% / 81.9% / 5.8%, main 95.3% / 95.2% / 1.1%, tail 0, and E33 gohome remains 23 / 24 recall with zero pre-tail FP | Best real-time-causal offline candidate so far; package/runtime validation is the next gate, while full-window extra/wrong remains a residual risk |
| E54 | E52/E51 candidate on the 6 QC-excluded episodes | Excluded samples are usable as a stress validation set, but not as normal train/val: failures include step-count outliers, bucket semantic outlier, and qpos jump/bucket-reference warnings. E54 generated an isolated excluded6 manifest plus dwell10 handoff labels, then replayed the full ACT temporal gate stack. Result: MAE 0.0548, startup 69.2% / 79.7% / 7.5%, longest-main 89.1% / 88.9% / 0.6%, full-available extra/wrong 9.9%, tail 0 / 114, gohome recall 5 / 6, pre-tail FP 0 | Strengthens the safety evidence for stop/gohome, but weakens confidence in action closeness and main-motion generalization. Episode 77 has no dwell10 eligible positive label, so its gohome miss should be treated as label/data pathology rather than a clean model miss |
| E55 | E51/E52 expert-effective frame miss decomposition | Recomputed all-train-ready per-frame deadzone intent over every expert-effective frame, not only the startup aggregate. Final E51 has 93.0% same-axis/same-direction coverage over all expert-effective frames, 81.9% in first-effective startup, 95.1% in the longest main-motion segment, and 93.0% in end80. Among final no-same frames, 88.0% already lacked same-direction deadzone crossing in raw ACT, 12.0% were phase-gate suppression, and snap/temporal gates added no further same-direction suppression. First40 miss frames have low gohome/tail evidence: tail-candidate mean 0.008, p95 0.035, gohome active 0%. | The 68% startup number was a conservative 40-step window occupancy metric, not the model's overall should-move intent success rate. The next optimization should target raw ACT amplitude/phase confidence on should-move frames, not simply raise all outputs or loosen the gohome/tail gate |
| E42 | E38 startup same/extra/missing frames vs same-frame intent probabilities | Extra/wrong startup frames are often high-intent too: mean episode extra-intent is 0.72, 60.4% of extra frames are >=0.70, and `episode_76` extra is 100% >=0.80 | Do not rely on intent threshold alone for release; next training/gate work needs temporal duration or stop-after-intent-end control |
| E43 | Episode-heldout 8-way direction gate on E38 phase-gated replay | Direction gate can lower extra/wrong by suppressing directions, but it cannot recover missing startup amplitude and worsens MAE as thresholds rise | Not a better standalone action candidate; use only as evidence for release control |
| E44 | E41 `m0200_i70` snap followed by E43 direction release | `dir_t70_s50` keeps most E41 startup gain and reduces extra below baseline, but MAE regresses to 0.0448 and high-intent over-duration episodes remain | Useful tradeoff evidence, not final; training-time temporal release is still required |

Notes:
- Tail stability passes for most completed candidates under runtime-scaled deadzone thresholds, except infer-only visual downsample060 baseline produced a small tail crossing rate (3 / 504 frames), and E25/E26 low-pass evals produced 6 / 504 effective frames under both downsample080 and downsample060.
- Earlier broad deadzone losses were too global and suppressed startup, especially for eye2.
- The active training direction is motion-intent aware: promote same-axis same-direction deadzone crossing when the expert is effective, and suppress deadzone crossing when the expert is ineffective or in the tail idle region.
- E14a shows that local transition-window promotion still behaves too much like a quieting regularizer: it improves tail p95/max and keeps MAE good, but it does not recover should-move startup coverage. Do not spend four-view GPU time on E14b unless we first change the objective structure.
- Deterministic low-pass is not a good next default: downsample_060 preserves main/tail but hurts startup and MAE; downsample_080 improves MAE versus downsample_060 but collapses startup further. Continue texture robustness work through stochastic/mixed augmentation or domain split, not fixed deterministic blur/downsample as the only training view.

## Active Hypotheses

1. A useful deadzone-aware model must match motion intent in both directions: promote crossings in true should-move windows and suppress crossings in should-stop windows.
2. Eye2 is often enough for action intent and may reduce visual shortcut risk, but eye2-deadzone needs a better loss schedule.
3. Four-view deadzone improves main motion but may use auxiliary views in a way that does not help startup.
4. Handoff should be an eligibility classifier with conservative runtime gate, not part of continuous action imitation.
5. Tail stability is currently acceptable; gohome delay is not the immediate offline blocker if policy remains below deadzone.
6. E04 proves the corrected loss semantics are directionally better than broad deadzone for eye2 startup, but the current weights still over-regularize startup.
7. E10 shows that simply increasing global same-direction promotion is not enough: startup recovers partly, but MAE and action p95 degrade. Global deadzone losses should not be the next default path. If we continue deadzone-aware training, the loss must be local to first-motion/startup windows, balanced by intent class, or scheduled, not applied uniformly over the whole trajectory.
8. The train-ready batch is not dominated by stop frames. E14 should not be framed as countering a stop-majority dataset; it should instead avoid direction-frequency bias and avoid inventing stick motion intent absent from the expert deadzone census.
9. E14a rejects the idea that a transition-window scalar penalty is enough. The next deadzone-aware attempt should either add an explicit should-move/should-stop intent head or train a separate lightweight intent gate, then condition action acceptance or action loss on that intent.
10. A qpos/qvel-only intent gate is not enough. E15 low-dimensional probes cannot simultaneously keep startup recall and suppress stop false positives, so the next intent experiment must use visual features or ACT shared representations.
11. E15b shows that simply adding an unweighted auxiliary intent head is not enough. The head collapses to all-negative predictions because axis-direction positives are sparse. E16 must weight positive direction labels before deciding whether the intent-head route is useful.
12. E16 shows the weighted intent head can learn positive should-move labels, but high any-move false positives mean the next deadzone-aware objective must be explicitly window- and direction-matched. The target is not "larger action"; it is "cross the correct directional deadzone when expert intent is active, and stay below all deadzones when expert intent is inactive."
13. E17 confirms that better MAE and quieter tail are not enough. A frequency-sensitive suppression-heavy loss can still erase startup should-move recall, so the next action-loss experiment must prioritize same-direction recall in expert-effective startup/main windows and treat idle suppression as a bounded safety term, not the dominant objective.
14. E18a shows that simply rebalancing action deadzone weights is still not enough. It recovers startup from E17 but remains below E16 and weakens main-motion intent, so the next step should evaluate E16-style action imitation with an external or auxiliary intent gate rather than keep adding action-head deadzone regularization.
15. E19 shows that the early bucket+ problem is not separable by a naive axis-only rule. Suppressing all solo bucket+ crossings removes start40 false motion but also destroys startup should-move coverage, so any deployable gate must infer phase/intent from visual-state context rather than hard-code a bucket-axis filter.
16. E20 shows that E16's existing auxiliary intent head is not a sufficient runtime gate. Thresholding the head reduces some extra/wrong motion and tail magnitude, but it does not reduce start40 bucket+ materially before it damages startup/main should-move coverage.
17. E21a shows that the deadzone-aware target should be framed as phase/intent matching: predict when motion is appropriate, then require the action to cross the correct deadzone in those windows. A small episode-held-out probe using E16 intent probabilities plus qpos/qvel suppresses early start40 bucket+ much better than the raw E16 intent head, but it is still an offline proxy and not a deployable visual gate.
18. E22a turns the E21a idea into a reproducible phase-gate artifact and scan. The best operating point is a lower simple probability threshold, not hysteresis: it suppresses early start40 bucket+ almost as well as E21a threshold 0.2 while recovering action closeness and a little startup coverage. The remaining blocker is the hard zeroing action-closeness penalty, not phase separability.
19. E22b shows the hard-zero penalty is avoidable in this batch. Scaling inactive-phase actions by 0.5 keeps them below runtime-scaled deadzones in start40 and tail, so deadzone intent metrics match E22a while MAE/RMSE improve beyond E16. This makes E22b the current best offline tradeoff, with the next risk shifted to deployability and held-out visual-domain robustness.
20. E23a/E23b show a real visual sensitivity risk for the current E16/E22 line. Infer-only downsample suppresses early false bucket+ motion, but it also suppresses true startup/main motion. Even the lighter downsample_080 keeps MAE close while cutting startup same-direction coverage sharply, so texture robustness cannot be inferred from original-image gates.
21. E24a/E24b show the phase gate cannot recover movement that the base visual policy no longer outputs under transformed images. With transformed-image intent probabilities, the gate still improves MAE/RMSE and stop quietness, but startup/main should-move coverage remains essentially bounded by the degraded base E16 actions. The next fix must target visual-domain robustness in the policy/features, not another phase-gate threshold.
22. E25 shows broad stochastic low-pass augmentation is too blunt. It restores low-pass main-motion coverage, but startup stays around 49%, same-direction startup is only about 49-51%, extra/wrong rises to about 15-18%, and low-pass tail gets 6 / 504 effective frames. The next visual robustness attempt should narrow the augmentation strength before changing architecture or adding another gate.
23. E26 shows narrowing stochastic low-pass from 0.60-1.00 to 0.80-1.00 does not solve the startup problem. It improves MAE versus E25 but reduces original startup to 45.5%, keeps low-pass startup around 48%, and still produces 6 / 504 low-pass tail effective frames. The current best path should not be more low-pass augmentation sweeps; return to E16/E22b phase-gate as the action-control candidate and treat visual-domain robustness as a separate representation/domain-split problem.
24. E27 visual clustering shows that the batch is not a single texture domain. Four-view clustering produces three episode-dominant domains: `texture_domain_3` with 12 episodes, `texture_domain_4` with 7 episodes, and `texture_domain_5` with 5 episodes. Eye2 clustering is more collapsed: `texture_domain_2` has 17 dominant episodes, `texture_domain_4` has 6, and `episode_102` alone is `texture_domain_5`. Therefore domain-held-out validation must record which view set produced the domain labels; a four-view domain split is not automatically an eye2 domain split.
25. E27 held-out training confirms the visual-domain risk but is not a better policy. Train-domain replay is acceptable on action closeness (MAE 0.0442) and main motion (97.2% same-dir), but startup remains weak (47.1% effective, 51.1% same-dir). Held-out domain MAE worsens to 0.0663 and startup extra/wrong rises to 35.2%, so the current data/model does not generalize cleanly to the held-out texture domain. E22b remains the best original-image offline action-control candidate; the next improvement should target representation/domain robustness or a window-conditioned should-move objective, not another global deadzone suppressor.
26. E28 shows that simply adding the rear/auxiliary GMSL views to the weighted-intent-head route is not enough. Action closeness improves versus E16 and is close to E22b (MAE 0.0433, RMSE 0.0869), but startup drops to 62.0% effective / 69.2% same-dir, main extra/wrong rises to 3.3%, and tail stability fails with 6 / 504 effective frames. Extra views may help value loss, but they do not solve the "should move when appropriate and stop when required" contract by themselves.
27. E29/E29b show that applying the existing E22-style phase gate to E28 is not enough either. The gate improves action closeness and reduces early false motion, but it cannot recover E28's weak startup should-move coverage and still leaves at least 1 / 504 tail crossing. This reinforces that the next training change should be an explicit window-conditioned should-move/should-stop objective or data/representation improvement, not another post-hoc threshold search on E28.
28. E30 shows that using handoff masks and a window-conditioned deadzone loss is mechanically valid but not a better objective as weighted here. It makes start40 completely quiet, but worsens replay MAE/RMSE, drops startup should-move coverage far below E16/E22b, and introduces tail crossings. The next E30-style attempt should not just increase suppression; it must either reduce the window loss weight, schedule it later, or restrict stop/wrong terms so startup same-direction recall remains protected.
29. E31 isolates the handoff-mask/masked-action-stat path from the window deadzone loss. It recovers action closeness compared with E30, but startup effective/same-dir falls to 45.7%/49.3% and tail effective frames rise to 16 / 504. This means the E30 regression is not only the extra window loss weights; changing the action distribution, loss mask, and sampling window also changes the learned policy enough to damage the three user metrics.
30. E32 shows gohome eligibility is learnable from the existing E16 intent probabilities plus qpos/qvel, but not yet safe enough as a standalone request gate. The selected `thr_0.95_c5` gate detects 23 / 24 episodes with mean detection delay 1.57 steps, but still has pre-tail early triggers in 4 / 24 episodes. Those unsafe triggers are short, only 5 total frames, while 73 / 78 early frames are inside the tail/dwell candidate region. The next gohome-aware step should separate tail/candidate-phase detection from final gohome eligibility instead of training the ACT action head on automation-tainted handoff windows.
31. E33 validates the two-stage gohome-awareness structure. The oracle tail gate proves the structure can remove all pre-tail triggers, and the learned tail-candidate probe can match that safety after using a stricter operating point (`candidate_threshold=0.97`, `candidate_consecutive_steps=10`). The selected learned two-stage gate keeps E32's 23 / 24 event recall, removes pre-tail false positives entirely, and leaves only 6 dwell-region early frames. This is the first gohome-awareness result that satisfies the "do not request early" offline gate, though it remains a probe on cached E16 intent probabilities rather than integrated runtime policy logic.
32. E34 combines the current best action candidate (E22b) and current best gohome-awareness candidate (E33) into one offline package. This is the first candidate that passes all four tracked offline gates at once: action closeness, should-move startup/main intent, tail stop stability, and conservative gohome request timing. It is still an offline artifact composition, not a field-ready deployment, because runtime wiring, inference latency, and visual-domain robustness remain unproven.
33. E35 reduces the deployability risk from OOF-only gates. Replaying the final saved phase-gate model and final saved gohome MLPs keeps the same safety properties as E34: action tail crossings remain 0 / 504, gohome pre-tail requests remain 0 / 24 episodes and 0 frames, and event recall stays 23 / 24. The three lightweight gate MLPs are not the latency blocker on CPU (p95 about 0.035 ms per step for all three); the next runtime risk is the full ACT image inference path and artifact packaging, not these auxiliary gates.
34. E36 closes the artifact-traceability gap for the current best offline candidate. The package manifest verifies 25 required files by SHA-256, including `policy_best.ckpt`, `dataset_stats.pkl`, `resolved_config.yaml`, final phase/tail/eligibility gate model bundles, and the E34/E35 evidence files. This reduces handoff/version risk but does not prove full ACT image runtime inference or visual-domain robustness.
35. E37 confirms that the E36 package can run a real HDF5 image path rather than only cached actions/probabilities. On `episode_73` and the previously E32-risky `episode_79`, full ACT image inference reproduces the cached raw E16 replay within about 1.5e-5 max action difference, then the same-frame intent probabilities drive the final phase and gohome gates without pre-tail gohome false positives or tail action crossings. This is still sampled evidence; the next blocker is broader coverage and visual-domain stress, not the mechanics of loading the packaged artifacts.
36. E38 extends E37 to all 24 train-ready episodes. Full ACT image inference reproduces cached E16 raw replay within a worst-case max action difference of 6.53e-4, and the final phase/gohome gates reproduce E35's all-data metrics: phase-gated MAE 0.0414, startup 64.9% effective / 78.4% same-dir / 5.6% extra-wrong, tail 0 / 504 effective frames, and gohome pre-tail false positives 0 / 24. The remaining problem is not package mechanics; it is uneven startup behavior across episodes/domains, especially `episode_74` startup suppression and high extra/wrong in `episode_100`.
37. E39 shows the E38 startup weak spots are mostly base-action amplitude/timing issues, not phase-gate-only failures. In `episode_74`, phase is active 87.5% of the startup window, but raw ACT bucket+ never crosses the runtime deadzone: max raw bucket+ is 0.5438 versus threshold 0.544. In `episode_100` and `episode_90`, raw ACT crosses bucket+ for too many frames compared with expert. In `episode_98`, raw ACT boom- never reaches the 0.357 threshold. The next candidate should correct axis/direction amplitude and duration near the deadzone boundary while preserving tail and gohome gates.
38. E40 confirms that some startup failures are recoverable by tiny action calibration near the runtime deadzone, but not by an unconstrained global boost. A phase-active snap margin of 0.02 raises startup effective from 64.9% to 71.4% and same-dir from 78.4% to 85.2% while keeping tail at 0 / 504, but extra/wrong also rises from 5.6% to 6.8%. The safer 0.003-0.005 margins move startup more modestly with less extra/wrong increase. The next candidate should snap or train only the intended axis/direction and should include duration limits, because `episode_100` is already a same-direction over-duration problem.
39. E41 validates the intended-direction filter as a better calibration shape than E40 global snap. With margin 0.02 and intent threshold 0.70, startup rises to 68.5% effective / 82.6% same-dir while extra/wrong stays at 5.9% and tail remains 0 / 504. The permissive 0.50 threshold recovers more `episode_74` startup, but crosses the extra/wrong 6.0% line. This makes intent-targeted snap a useful primitive, but it still needs duration control or training-time shaping for over-duration episodes such as `episode_100`.
40. E42 shows why E41 cannot be fixed by just raising or lowering the intent threshold. In E38 startup windows there are 583 same-direction, 40 extra/wrong, and 169 missing direction events. Mean extra-frame intent is still 0.72, and 60.4% of extra frames are above 0.70. `episode_100` has 18 extra bucket+ frames with 72.2% above 0.70, while `episode_76` has 13 extra boom- frames with 100% above 0.80. The ACT intent head is therefore useful for amplitude targeting, but it is not a reliable release signal by itself.
41. E43 shows that a learned direction-level gate is not enough as a standalone post-hoc fix. It can reduce startup extra/wrong from 5.6% to about 5.3% at threshold 0.70 or to 5.0% at threshold 0.80, but startup effective/same-dir do not improve and MAE worsens because the gate only suppresses action; it cannot push near-deadzone intended motion over threshold.
42. E44 combines E41 amplitude snap with E43 direction release. The best current combined tradeoff, `dir_t70_s50`, keeps startup at 68.2% effective / 82.5% same-dir and lowers extra/wrong to 5.6%, with tail still 0 / 504. However, MAE rises to 0.0448, and `episode_100` / `episode_76` remain high-extra over-duration cases. This confirms that post-hoc gates can improve one slice of the problem but do not replace a training-side temporal release objective.
43. E45/E46 are a negative sensitivity result for the direct `temporal_release_loss` training direction. The stronger release penalty creates non-zero tail crossings and weaker startup, while the weaker penalty recovers tail stability but collapses startup even further. Do not continue with a simple release weight/window sweep.
44. E47 shows that temporal context is useful when it is applied as a separate direction gate on top of the E41 amplitude primitive. Context over intent/qpos/qvel can reduce extra/wrong without relying on same-frame intent alone, but aggressive inactive scaling still hurts MAE.
45. E47b `tdir_t50_s75` is the best non-causal offline tradeoff so far: MAE 0.0406, startup 67.3% effective / 81.4% same-dir / 5.3% extra-wrong, main 95.3% effective / 95.2% same-dir / 1.1% extra-wrong, and tail 0 / 504. It uses future context offsets (`+1/+2/+5/+10`), so it must not be treated as runtime deployable.
46. E49 removes the future-frame leak by using only offsets `[-10, -5, -2, -1, 0]`. The causal `tdir_t50_s75` keeps most of the temporal-gate benefit: MAE 0.0408, startup 67.9% effective / 81.9% same-dir / 5.8% extra-wrong, longest-main 95.3% effective / 95.2% same-dir / 1.1% extra-wrong, and tail 0 / 504.
47. E50 combines the causal E49 action candidate with the E33 gohome request gate. The combined offline gates pass: gohome recall remains 23 / 24, pre-tail false positives remain 0 / 24 episodes and 0 frames, and action tail crossings remain 0 / 504. The residual risk is full-window extra/wrong, which is close to E41 and slightly higher than E38.
48. E51 proves the causal temporal gate through the full ACT image path and final saved temporal gate model, not only cached OOF probabilities. Final full-ACT metrics are MAE 0.0408, startup 68.4% effective / 82.6% same-dir / 5.7% extra-wrong, longest-main 95.3% effective / 95.2% same-dir / 1.1% extra-wrong, tail 0 / 504, and gohome pre-tail false positives 0. The ACT offline loop p95 is 20.3 ms and the four lightweight gates add about 0.0075 ms p95 in batched CPU evaluation.
49. E52 closes the traceability gap for E51. The package manifest verifies 23 / 23 artifacts by SHA-256, including the ACT checkpoint bundle, phase/gohome models, causal temporal direction model, E50 combined evidence, E51 full-ACT smoke summary, startup/tail gate, and window comparison. It is the best packaged offline candidate so far, but field-side runtime integration still needs explicit wiring for phase-gated action, snap action, temporal-direction action, safe action, and commanded action logging.
50. E53 defines the first field-side no-motion acceptance contract. A `shadow_zero` dry-run log is not useful unless it proves both sides simultaneously: the policy stack actually produced finite non-zero `policy_action`, and the machine-facing `safe_action`, `commanded_action`, and `policy_returned_action` remained zero. Gate-stack diagnostics must be visible as separate fields so action closeness, deadzone intent, tail stop, and gohome request behavior can be audited without enabling live motion.
51. E54 uses the six QC-excluded episodes as a held-out stress validation set rather than training data. The result is mixed: tail stop remains strong at 0 / 114 deadzone-effective frames and gohome has 0 pre-tail false-positive episodes, but action closeness degrades to MAE 0.0548 and longest-main same-direction drops to 88.9%. This makes E52 less overconfident: it is still the best candidate, but the credible claim is now "safe enough for no-motion dry-run," not "generalizes strongly to failed/outlier episodes."
52. E55 corrects the interpretation of "deadzone startup ability." The reported 68.4% is not the global probability that the model moves when it should; it is policy-effective occupancy in a fixed 40-frame window after first expert motion. On all expert-effective frames, E51/E52 gives 93.0% same-axis/same-direction deadzone crossing, with 95.1% in the main segment and 81.9% in first-effective startup. Most missed should-move frames are already missing at raw ACT output, so more gate tuning cannot solve the main issue.

## Experiment Queue

| ID | Status | Purpose | Model / Data Change | Expected Gate Movement |
| --- | --- | --- | --- | --- |
| E04 | Completed | Train motion-intent-aware deadzone loss | Promote same-axis same-direction crossing when expert is effective; suppress crossing only when expert is ineffective/tail idle | Tail passed, main retained, startup still below baseline |
| E05 | Completed | Compare view count under scoped loss | Train E04 as four-view and eye2 variants | Four-view has better MAE; eye2 has stronger main intent but both fail startup-vs-baseline gate |
| E06 | In progress | Solidify gohome eligibility labels | Add `gohome_eligible_label`, `gohome_loss_mask`, `tail_idle_mask`, automation filtering; sweep dwell_min 5/10/15/20 raw steps | Positive label starts only after human stop + dwell; no early unsafe labels |
| E07 | Planned | Prove gohome awareness learnability | Independent classifier: state-only, eye2+state, four-view+state | Event recall near `t_go` high; early false positives near zero |
| E08 | Planned | Visual texture sensitivity | Fixed qpos / multi-FPV gate on best candidate models | No major degradation under image swap; tail extra/wrong does not increase |
| E09 | Planned | Domain split validation | Texture-domain or episode-domain held-out split when labels/clusters are ready | Held-out intent gates remain close to train-domain gates |
| E10 | Completed | Recover startup while preserving tail | Promote-biased deadzone weights: same_dir 0.25, idle 0.01, wrong 0.02, four-view and eye2 | Startup partly recovered, but action closeness regressed too much |
| E11 | Planned | Avoid global deadzone-loss regression | Either startup-window-only deadzone loss or no-deadzone baseline plus visual/domain gates | Do not accept any model that improves one gate by damaging MAE or startup-vs-baseline |
| E12 | Completed | Test texture-detail suppression during training | Eye2 baseline with training/inference `downsample_060`, no deadzone loss | Main/tail preserved with consistent transform, but startup and MAE still below baseline |
| E13 | Completed | Test lighter texture suppression | Eye2 baseline with training/inference `downsample_080`, no deadzone loss | Rejected: MAE improves over E12 but startup collapses below both E12 and eye2 baseline |
| E14a | Completed | Test startup-targeted deadzone-aware training | Eye2 no-transform baseline plus `same_dir_window: expert_transition_window`, 4-step local transition window; idle/wrong suppression unchanged | Rejected: startup effective/same-dir stayed below eye2 baseline despite good MAE and quiet tail |
| E14b | Skipped | Compare view count after E14a | Four-view variant only if E14a improves startup-vs-baseline or gives a useful tradeoff | Skipped because E14a failed the movement-first gate |
| E15a | Completed | Test whether low-dimensional state is enough for should-move intent | Episode-held-out linear probes on qpos, qpos+qvel, qpos+dqpos using runtime-scaled deadzone labels | Rejected: low threshold preserves startup but fires during stop; safe threshold kills startup |
| E15b | Completed | Separate visual/state intent recognition from action magnitude | ACT auxiliary 8-way axis-direction intent head on shared decoder features; no action deadzone penalty | Rejected: action MAE improves but startup collapses; intent head all-negative |
| E16 | Completed | Fix sparse-label collapse in intent head | E15b plus `intent_loss.positive_weight: 8.0`; still no action deadzone penalty | Head recall recovered and startup direction quality improved, but high false positives require a more explicit intent-matching loss |
| E17 | Completed | Align deadzone-aware training with should-move/should-stop intent | E16 weighted intent head plus frequency-sensitive deadzone action loss: same-direction promotion, `all_idle_axes` idle denominator, and `all_wrong_candidate_axes` wrong denominator; eye2 first | Rejected: MAE/tail improved but startup should-move recall collapsed |
| E18a | Completed | Prioritize should-move recall without losing stop stability | Eye2 first; keep weighted intent head, set `same_dir=0.12`, `idle=0.02`, `wrong=0.03`, and keep frequency-sensitive denominators | Rejected as final: startup partially recovered from E17 but stayed below E16/E01, and main intent regressed |
| E19 | Completed | Separate action imitation from stop/start gating | Post-hoc E16 gate: zero bucket axis when only bucket+ crosses deadzone | Rejected: start40 false motion removed, but startup effective collapsed to 31.5% |
| E20 | Completed | Test existing E16 intent head as visual-state gate | Cache E16 query-0 intent probabilities; scan any-move and direction thresholds; materialize direction threshold 0.7 gate | Rejected: start40 bucket+ remains while startup/main are suppressed |
| E21a | Completed | Test whether a learned phase/intent gate can separate should-move from should-stop | Episode-held-out MLP probe on E16 intent probabilities plus qpos/qvel; post-hoc gate E16 actions at threshold 0.2 | Strongly reduces early start40 bucket+ while preserving more startup/main than E20, but worsens MAE and remains offline-only |
| E22a | Completed | Build a reproducible should-move phase-gate artifact | Train episode-held-out MLP on E16 intent probabilities plus qpos/qvel; save final model artifact; scan simple and hysteresis gates; materialize best replay | Strong improvement over E16 early false motion with modest startup/main loss; better MAE than E21a threshold 0.2 |
| E22b | Completed | Reduce hard-gate action-closeness penalty | Scan inactive-phase action scale values with the same phase gate; materialize `simple_0.15_s0.50` | Keeps E22a deadzone gates unchanged while improving MAE/RMSE beyond E16 |
| E23a | Completed | Stress E16 under stronger infer-only low-pass visual transform | Replay E16 weighted-intent eye2 with `--image-transform downsample_060`; run standard gates | Rejected as robust candidate: startup/main collapse despite start40 false motion removal |
| E23b | Completed | Stress E16 under lighter infer-only low-pass visual transform | Replay E16 weighted-intent eye2 with `--image-transform downsample_080`; run standard gates | Still startup-sensitive: MAE close to original but startup same-dir drops sharply |
| E24a | Completed | Test phase gate under lighter transformed visual inputs | Cache E16 intent probabilities under `downsample_080`, apply E22b-style soft phase gate to transformed E16 actions | Improves MAE/RMSE and stop quietness, but does not recover startup/main motion |
| E24b | Completed | Test phase gate under stronger transformed visual inputs | Cache E16 intent probabilities under `downsample_060`, apply E22b-style soft phase gate to transformed E16 actions | Same conclusion as E24a; phase gate cannot restore missing base motion |
| E25 | Completed | Train or evaluate a visual-domain-robust policy candidate | Eye2 E16-style weighted intent head with stochastic `random_downsample_060_100_seed25` train augmentation; evaluate original plus low-pass/domain gates after completion | Rejected: low-pass main recovers, but startup, original MAE, extra/wrong, and low-pass tail regress |
| E26 | Completed | Test narrower stochastic texture suppression | Same as E25 but stochastic `random_downsample_080_100_seed26`; keep split, views, intent head, and no action deadzone loss unchanged | Rejected: MAE improves over E25, but startup is worse and low-pass tail crossings remain |
| E27 | Completed | Test visual-domain held-out generalization directly | Eye2 E16-style weighted intent head, no image transform; train on 19 episodes and validate on four-view `texture_domain_5` held-out episodes 82/98/99/100/102 | Rejected as candidate: held-out MAE and extra/wrong degrade, and all-data startup is below E16/E22b |
| E28 | Completed | Test whether four GMSL views improve weighted-intent action control | Four-view version of E16: weighted 8-way intent head, no action deadzone loss, no image transform, full train-ready split reused from E00 | Rejected: action closeness is good, but startup and tail gates regress versus E22b |
| E29 | Completed | Test whether phase gate can rescue E28 | Cache E28 intent probabilities; train E22-style phase gate on E28 actions; scan inactive scales and materialize auto gate plus E22b-style `simple_0.15_s0.50` | Rejected: gate improves MAE/start40 but not startup/main intent or tail enough to beat E22b |
| E30 | Completed | Train the policy to match should-move and should-stop windows directly | Eye2 E16-style weighted intent head plus action-aligned handoff masks, masked action stats, and window-conditioned same-dir/stop/wrong deadzone loss | Rejected: start40 quiets to 0%, but replay MAE/RMSE, startup intent, extra/wrong, and tail stability all regress versus E22b |
| E31 | Completed | Isolate whether E30 failed because of window loss or because of the masked handoff data path | Eye2 E16-style weighted intent head plus action-aligned handoff masks, masked action stats, and action-loss sampling, with `window_deadzone_loss.enabled: false` | Rejected: MAE improves over E30, but startup and tail remain worse than E22b; future handoff/gohome-aware training needs a different target or curriculum, not just this mask path |
| E32 | Completed | Test whether gohome request eligibility is learnable without imitating automation actions | Episode-heldout MLP on cached E16 intent probabilities plus qpos/qvel, evaluated with event-level recall/early-trigger metrics | Not deployable standalone: high recall and short delay, but 4 / 24 episodes still trigger before the tail candidate region; use as evidence for a staged gohome gate, not as current runtime logic |
| E33 | Completed | Test the staged gohome-awareness fix suggested by E32 | Train a tail/cycle-complete candidate probe on cached E16 intent probabilities plus qpos/qvel, then AND it with E32 eligibility probabilities | Best gohome-awareness probe so far: learned two-stage gate has 95.8% event recall, zero pre-tail early triggers, and only one missed episode |
| E34 | Completed | Combine the best action and gohome-awareness candidates into one offline deployability report | E22b action gate plus E33 two-stage gohome request gate; write `combined_candidate_summary.json/csv` and artifact manifest | First integrated offline candidate that passes action tail stop and gohome pre-tail safety gates together; next step is runtime package/replay smoke, not another action retrain |
| E35 | Completed | Verify E34 with final saved gate models rather than OOF probabilities | Load final phase/tail/eligibility MLP artifacts, replay final phase-gated actions, recompute final gohome events, and measure gate latency | Pass: final models preserve E34 safety gates; next step is full ACT runtime package smoke with image inference and visual-domain checks |
| E36 | Completed | Make the current best candidate reproducible and hard to misdeploy | Build SHA-256 manifest and verifier for the E16 action bundle, E22b phase gate, E32/E33 gohome gates, and E34/E35 evidence | Pass: 25 / 25 required artifacts verified; next step remains full ACT image inference smoke and visual-domain robustness |
| E37 | Completed | Verify the packaged candidate can run full ACT image inference before applying gates | Load E36 manifest, run ACT on real HDF5 images for ep73/ep79, collect same-frame intent logits, then apply final phase and gohome gates | Pass sampled smoke: raw action matches cached replay, gated startup/tail/gohome checks pass on the two selected episodes; next step is broader episode/domain coverage |
| E38 | Completed | Verify full ACT image inference over all train-ready episodes | Run the E37 package smoke over all 24 train-ready episodes and rerun startup/tail/gohome/domain summaries | Pass mechanics and safety gates: full image path matches cached replay, tail and pre-tail gohome gates pass; next step should target startup/domain weak spots rather than packaging |
| E39 | Completed | Diagnose whether E38 startup weak spots come from ACT, phase gate, or domain/window definitions | Compare expert, raw ACT, phase-gated action, and phase probabilities in startup windows, with domain labels | Phase gate is not the primary blocker; next candidate should target action amplitude/duration near deadzone boundaries |
| E40 | Completed | Test whether near-deadzone action calibration can recover startup without retraining | Post-hoc snap phase-active policy actions that are just below runtime deadzone thresholds; sweep margins 0.001-0.020 and rerun startup/tail gates | Diagnostic pass, deploy reject as global rule: startup improves and tail stays quiet, but extra/wrong rises with margin and over-duration cases remain |
| E41 | Completed | Add intent direction targeting to the near-deadzone snap probe | Snap only if the same-frame ACT intent head supports the same axis/direction; sweep margins 0.003-0.020 and intent thresholds 0.50-0.90 | Better tradeoff than global snap: `m0200_i70` keeps extra/wrong under 6.0% and improves startup, but still does not solve over-duration |
| E42 | Completed | Check whether same-frame intent probabilities can distinguish same-dir motion from over-duration extra motion | Compare intent distribution for startup same / extra / missing direction events under E38 | Reject threshold-only release: many extra frames have high intent, so the next candidate needs temporal release/duration control or a training-time persistence penalty |
| E43 | Completed | Test a learned per-axis-direction release gate without amplitude boost | Train 8-way episode-heldout MLP over intent probabilities plus qpos/qvel, then scale inactive policy directions | Reject standalone: lowers extra only by suppressing action, does not improve startup movement, and worsens MAE |
| E44 | Completed | Combine E41 amplitude snap with E43 release gate | Apply E43 direction release to E41 `snap_m0200_i70` replay and rerun gates | Mixed result: `dir_t70_s50` has 68.2% startup effective / 82.5% same-dir / 5.6% extra and tail 0, but MAE regresses to 0.0448 |
| E45 | Completed | Train a temporal release / persistence objective directly in ACT | E16 eye2 weighted intent head plus `temporal_release_loss`: penalize same-direction policy persistence for 4 steps after expert direction release | Rejected: all-train-ready replay has worse MAE, weaker startup, higher extra/wrong, and non-zero tail crossings versus E38/E41/E44 |
| E46 | Completed | Test whether E45 failed because the temporal-release penalty was too strong | Same as E45 but weaker release penalty: `weight=0.01`, `release_window_steps=2` | Rejected: tail recovers to 0 / 504, but startup effective and same-dir collapse below E45/E38/E41 |
| E47 | Completed | Test whether a temporal-context direction gate can solve over-duration without retraining ACT | Add context-window intent/qpos/qvel features to E43-style episode-heldout direction gate, applied on E41 `snap_m0200_i70` replay | Positive non-causal diagnostic: `tdir_t50_s75` is strong, but future offsets make it an offline upper bound |
| E49 | Completed | Remove future-frame leakage from E47 | Repeat E47 with causal-only offsets `[-10, -5, -2, -1, 0]` and inactive scale 0.75 | Positive: causal `tdir_t50_s75` keeps better MAE/startup than E38 and preserves tail, with extra/wrong slightly above E38 but below E41 |
| E50 | Completed | Combine causal E49 action with conservative gohome awareness | Use E49 `tdir_t50_s75` action plus E33 two-stage gohome request gate | Best real-time-causal integrated offline candidate so far; next gate is runtime package/full ACT smoke, not live deployment |
| E51 | Completed | Verify E50 through final saved causal temporal gate and full ACT image inference | Run ACT on all train-ready HDF5 images, apply final phase gate, E41 snap, final causal temporal direction gate, and E33 gohome gates | Best runtime-smoke candidate so far; next gate is package manifest/update with temporal model and E51 evidence |
| E52 | Completed | Package the E51 candidate with traceable artifact verification | Build SHA-256 manifest over E36 base package, E49 temporal model, E50 combined summary, and E51 full-ACT evidence | Pass: 23 / 23 artifacts verified; next gate is field-side runtime integration wiring and a no-motion/logging dry run |
| E53 | Completed | Define the no-motion dry-run acceptance gate before live motion | Add a `steps.jsonl` verifier for `shadow_zero` logs and pass through E52 gate-stack diagnostics in the receiver test logger | Pass as tooling: verifies policy ran, no commanded/safe/returned motion escaped, and required gate fields are present; next gate is collecting a real field-side dry-run log |
| E54 | Completed | Use QC-excluded episodes as stress validation instead of training data | Build excluded6 manifest, generate isolated dwell10 handoff labels, run E52/E51 full-ACT temporal gate replay and deadzone/window summaries | Mixed but useful: MAE/main-motion generalization weakens, while tail stop and no-pre-tail gohome safety still pass |
| E55 | Completed | Reinterpret deadzone intent success over all should-move frames | Decompose final E51 misses into raw-action miss vs phase/snap/temporal gate suppression and check tail/gohome probabilities on missed first-effective frames | Main issue is raw ACT under-motion/phase confidence, not gohome confusion or post-hoc gate suppression |

### E30 Window-Conditioned Motion Intent Loss

Reason:
- E22b is still the current best offline tradeoff, but it is a post-hoc gate on top of E16 actions.
- E30 tests whether the ACT action head can learn the corrected deadzone-intent contract directly: move through the correct directional deadzone when the expert moves, and stay below all deadzones in expert-idle / tail / automation-owned windows.

Implementation:
- Added optional `deadzone_intent` batches that preserve the legacy four-tuple dataset contract when disabled.
- Added mask-aware action normalization so the handoff dataset's automation tail does not shift the human-action action stats.
- Added `window_deadzone_loss` with separate same-direction promotion, stop suppression, and wrong-direction suppression terms.
- Added `require_action_loss_in_chunk` sampling so pure automation-tail chunks are not sampled as ordinary action-imitation windows.

Smoke:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e30_window_deadzone_intent_eye2_smoke.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2_smoke`.
- Split path: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2_smoke/train_val_split.yaml`.
- Result: 1 epoch completed, best epoch 0, best val loss 76.10833358764648.
- Smoke emitted non-zero `window_deadzone_same_dir_loss`, `window_deadzone_stop_loss`, and `window_deadzone_wrong_loss`; `dataset_stats.pkl` action mean was `[-0.009317, 0.020309, -0.001789, -0.041707]`, matching the masked human-action distribution rather than the full automation-tainted handoff action distribution.

Full training:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e30_window_deadzone_intent_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2`.
- Split path: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e30_window_deadzone_intent_eye2/train_val_split.yaml`.
- Started at 2026-07-09 11:58 CST in the current training session with PID 835212. Completed at 2026-07-09 12:19 CST.
- Training result: best epoch 1155, best val loss 0.12504172697663307.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e30_window_deadzone_intent_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e30_window_deadzone_intent_eye2_gate_runtime_scaled`.

E30 results:
- Replay: MAE 0.0604, RMSE 0.1186, policy p95 abs 0.8918, policy max abs 0.9792.
- Startup first-effective 40: policy effective 51.8%, same-axis/same-direction 53.4%, extra/wrong 17.6%.
- Main-motion longest expert-effective segment: policy effective 97.2%, same-axis/same-direction 96.3%, extra/wrong 1.8%.
- Start40: policy effective 0.0%, extra/wrong 0.0%.
- Tail: 10 / 504 effective frames, tail effective rate 2.0%, mean tail p95 max abs 0.5861, max policy max abs 0.7610.

Decision:
- Reject as candidate and do not launch E30-four with these weights. The implementation path works, but the objective overfits the wrong tradeoff: it suppresses early false motion while damaging should-move startup and action closeness, and it fails the tail stop-stability gate.
- Next E30-style direction should keep the mask-aware data path but change the objective schedule/weights before spending four-view GPU time.

### E31 Masked-Handoff Intent Baseline Without Window Loss

Reason:
- E30 combined two changes: a mask-aware handoff dataset path and a new `window_deadzone_loss`.
- E31 isolates the first part by keeping the handoff masks, masked action stats, action-loss mask, and valid-start filtering, but disabling `window_deadzone_loss`.
- This answers whether E30 failed mainly because the added window loss was too strong, or whether the handoff/mask training path itself changes the learned policy tradeoff.

Implementation:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e31_masked_handoff_intent_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e31_masked_handoff_intent_eye2`.
- Split path: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e31_masked_handoff_intent_eye2/train_val_split.yaml`.
- Smoke confirmed all `window_deadzone_*` losses remained zero when the feature was disabled.
- Full training first failed during `torch.save` because `/data` was 100% full and `policy_epoch_249_seed_0.ckpt` was truncated. The generated intermediate `policy_epoch_*_seed_*.ckpt` files under this batch's ckpt root were removed, freeing about 806 GB while preserving `policy_best.ckpt`, `policy_latest.ckpt`, stats, configs, metadata, and plots.
- E31 was resumed from `policy_latest.ckpt`, and the config was changed to `save_latest_every: 50`, `checkpoint_every: 2000`, `plot_every: 2000` to prevent repeated 961 MB epoch snapshots from filling `/data` again.
- Training result: completed at 2026-07-09 12:47 CST, best epoch 1935, best val loss 0.10588763654232025.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e31_masked_handoff_intent_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e31_masked_handoff_intent_eye2_gate_runtime_scaled`.

E31 results:
- Replay: MAE 0.0500, RMSE 0.1039, policy p95 abs 0.8462, policy max abs 0.9626.
- Startup first-effective 40: policy effective 45.7%, same-axis/same-direction 49.3%, extra/wrong 10.1%.
- Main-motion longest expert-effective segment: policy effective 95.4%, same-axis/same-direction 93.9%, extra/wrong 2.4%.
- Start40: policy effective 0.0%, extra/wrong 0.0%.
- Tail: 16 / 504 effective frames, tail effective rate 3.2%, mean tail p95 max abs 0.6055, max policy max abs 0.7676.

Decision:
- Reject as current candidate. E31 proves the disabled-window-loss path is mechanically stable and recovers part of E30's MAE damage, but it still violates the user's three metrics compared with E22b: weaker action closeness, much weaker startup should-move intent, and worse tail stop stability.
- Do not treat the handoff-mask path as a harmless preprocessing improvement. Future gohome/handoff-aware training should likely separate action imitation from gohome eligibility prediction or use a staged/curriculum objective, rather than training one masked ACT action head on this path.

### E32 Independent Gohome Eligibility Probe

Reason:
- The user-defined gohome problem is not "imitate automation gohome motion"; it is "request gohome after the human-operated cycle is complete, preferably late rather than early, and do not keep producing unsafe random tail actions."
- E30/E31 showed that mixing handoff masks into the ACT action path damages action closeness, startup should-move intent, and tail stop stability.
- E32 tests the narrower question: can the existing learned motion representation plus state predict conservative gohome eligibility at the end of a cycle?

Implementation:
- Added `testbed/testbed/policies/gohome_eligibility.py` with runtime-causal consecutive-frame activation and event-level metrics.
- Added `scripts/e32_gohome_eligibility_probe.py`, an episode-heldout MLP over cached E16 intent probabilities plus qpos/qvel.
- The probe uses `handoff/gohome_eligible_label`, `handoff/gohome_loss_mask`, and `handoff/tail_idle_mask` from `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_eligibility_20hz_dwell10`.
- It deliberately does not imitate automation actions and does not use the fixed `go_home_acceptable_position` diagnostic as its main training target.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e32_gohome_eligibility_probe`.
- Fold summary: `fold_summary.json`.
- Threshold scan: `threshold_scan.csv`.
- Selected events: `thr_0.95_c5_events.csv`.
- Final probe bundle: `gohome_eligibility_model.pt` plus `gohome_eligibility_model_metadata.json`.

E32 results:
- Fold-level frame classifier: recall 98.9-100.0%, precision 52.0-68.3%, false-positive rate 1.29-1.79%, accuracy 98.24-98.73%.
- Auto-selected event gate: `thr_0.95_c5`.
- Event recall: 23 / 24 episodes, 95.8%.
- Mean detection delay after `eligible_start`: 1.57 steps; median delay 0.0 steps.
- Mean margin before `t_go`: 13.61 steps.
- Early false positives before `eligible_start`: 19 / 24 episodes, but split by tail/candidate phase this is 78 early frames total: 73 dwell/tail-candidate frames and 5 true pre-tail frames.
- True pre-tail early triggers: 4 / 24 episodes, 5 total frames. The affected episodes are `episode_79`, `episode_82`, `episode_85`, and `episode_86`; each starts only 1-2 frames before the current `tail_idle_mask`.
- No threshold/consecutive setting in the scanned grid achieved both high event recall and zero pre-tail false-positive episodes.

Decision:
- Reject E32 as a standalone deployable gohome request gate. The key safety rule is "late is acceptable; early can be unsafe", and E32 still fires before the tail candidate region in 4 episodes.
- Keep E32 as evidence that gohome awareness is learnable and should be staged: first detect a conservative tail/cycle-complete candidate phase, then allow a final gohome eligibility trigger inside that phase. This avoids forcing the continuous ACT action head to learn automation-owned gohome behavior.

### E33 Two-Stage Gohome Gate Probe

Reason:
- E32 showed that eligibility probability alone is almost right but fires a few frames before `tail_idle_mask` in 4 episodes.
- The safety constraint is asymmetric: a late request is acceptable, but a request before the cycle-complete/tail candidate phase is unsafe.
- E33 tests whether a learned tail/cycle-complete candidate gate can suppress those pre-tail triggers while preserving gohome eligibility recall.

Implementation:
- Added `gated_active_mask` and `gohome_event_metrics_from_active_mask` to `testbed/testbed/policies/gohome_eligibility.py`.
- Added `scripts/e33_two_stage_gohome_gate_probe.py`.
- Stage 1 trains an episode-heldout MLP on `handoff/tail_idle_mask` using the same E16 intent probabilities plus qpos/qvel features as E32.
- Stage 2 reuses E32 episode-heldout `eligibility_prob` artifacts and only permits eligibility activation when the candidate gate is active.
- The script reports both `oracle_tail && eligibility` and `learned_tail && eligibility` so we can distinguish structural upper bound from learnability.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e33_two_stage_gohome_gate_probe`.
- Candidate fold summary: `candidate_fold_summary.json`.
- Threshold scan: `threshold_scan.csv`.
- Selected events: `learned_tail_t0.97_tc10_e0.80_ec3_events.csv`.
- Final candidate model: `tail_candidate_model.pt` plus `tail_candidate_model_metadata.json`.

E33 results:
- Tail-candidate fold classifier: recall 96.6-100.0%, precision 71.6-81.1%, false-positive rate 0.70-1.11%, accuracy 98.92-99.32%.
- Oracle tail plus eligibility removes all pre-tail false positives at every scanned eligibility setting, confirming the two-stage structure is sufficient if the candidate phase is correct.
- Selected learned gate: `learned_tail_t0.97_tc10_e0.80_ec3`.
- Event recall: 23 / 24 episodes, 95.8%; missed episode: `episode_83`.
- Pre-tail false positives: 0 / 24 episodes, 0 frames.
- Remaining early frames: 6 total, all inside dwell/tail candidate region, each one frame before `eligible_start`.
- Mean detection delay after `eligible_start`: 2.87 steps; median delay 3.0 steps.
- Mean margin before `t_go`: 12.30 steps.
- Candidate-frame metrics at selected threshold: recall 92.9%, precision 90.3%, false-positive rate 0.31%, accuracy 99.48%.

Decision:
- E33 is the best gohome-awareness probe so far and is the first offline gate that satisfies the user's "do not request early" safety direction while preserving E32's event recall.
- This still does not replace the action policy candidate E22b: E33 is a gohome-request awareness probe on cached E16 intent probabilities. The next integration step should package it as an auxiliary runtime gate candidate and test replay timing alongside E22b actions, not retrain ACT on automation-owned gohome motion.

### E34 E22b Action + E33 Gohome Combined Candidate

Reason:
- E22b is the best action-control candidate across action closeness, startup/main should-move intent, and tail stop stability.
- E33 is the best conservative gohome-request probe, but it does not produce actions.
- E34 binds the two into one offline candidate report so future runtime packaging evaluates the full behavior contract instead of optimizing action and gohome separately.

Implementation:
- Added `scripts/e34_combine_policy_gohome_candidate.py`.
- The script reads E22b action `gate_summary.json`, E22b startup/tail deadzone gate CSVs, and E33 `gate_summary.json`.
- It writes a compact candidate summary and artifact manifest without copying large checkpoints or replay arrays.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e34_e22b_action_e33_gohome_combined_candidate`.
- Summary JSON: `combined_candidate_summary.json`.
- Summary CSV: `combined_candidate_summary.csv`.
- Artifact manifest: `candidate_artifacts.json`.
- Action source: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22b_phase_gate_soft_scale_probe/simple_0.15_s0.50`.
- Action deadzone gates: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22b_phase_gate_soft_scale_probe_gate_runtime_scaled`.
- Gohome source: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e33_two_stage_gohome_gate_probe`.

E34 results:
- Action replay: MAE 0.0414, RMSE 0.0861.
- Startup first-effective 40: policy effective 65.1%, same-axis/same-direction 78.6%, extra/wrong 5.7%.
- Main-motion longest expert-effective segment: policy effective 94.8%, same-axis/same-direction 94.5%, extra/wrong 1.2%.
- Tail stability: 0 / 504 effective frames; tail effective rate 0.0%; mean tail p95 max abs 0.0622; max policy max abs 0.2901.
- Gohome request: 23 / 24 event recall, 0 pre-tail false-positive episodes, 0 pre-tail active frames, 6 dwell-region early frames.
- Gohome timing: mean detection delay 2.87 steps after `eligible_start`; mean margin 12.30 steps before `t_go`.

Decision:
- E34 is the best integrated offline candidate so far. It passes the action tail-stop gate and the gohome no-pre-tail-request gate together while preserving E22b's action metrics.
- Do not mark it field-ready yet. The next step should be a runtime/replay smoke package that runs E22b action gating plus E33 gohome request gating in one inference path and measures latency/input availability. Visual-domain robustness also remains a separate unresolved risk.

### E35 Final Gate Runtime Smoke

Reason:
- E34 used materialized OOF gate outputs. Runtime will load final saved MLP artifacts, so we need to check whether the selected thresholds still hold under final-model probabilities.
- This smoke is deliberately limited to the lightweight gate stack: phase gate, tail candidate gate, and gohome eligibility gate. It does not claim full ACT image inference latency.

Implementation:
- Added `scripts/e35_runtime_gate_smoke.py`.
- Added focused tests in `testbed/tests/test_e35_runtime_gate_smoke.py`.
- The script loads:
  - E22b final `phase_gate_model.pt`.
  - E33 final `tail_candidate_model.pt`.
  - E32 final `gohome_eligibility_model.pt`.
- It rebuilds feature tensors from cached E16 intent probabilities plus qpos/qvel, materializes final phase-gated actions, recomputes final two-stage gohome events, and measures CPU latency for the three gate MLPs in a per-step loop.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e35_e34_runtime_gate_smoke`.
- Summary: `runtime_gate_smoke_summary.json`.
- Final phase action replay: `final_phase_action_replay`.
- Final phase deadzone gates: `final_phase_gate_runtime_scaled`.
- Final gohome events: `final_gohome_events.csv`.
- Final gate probabilities: `final_phase_probs`, `final_tail_candidate_probs`, `final_gohome_eligibility_probs`.

E35 results:
- Final phase-gated action replay: MAE 0.0414, RMSE 0.0859, policy p95 abs 0.8269, policy max abs 0.9137.
- Startup first-effective 40 after final phase model: policy effective 64.9%, same-axis/same-direction 78.4%, extra/wrong 5.6%.
- Tail stability after final phase model: 0 / 504 effective frames, tail effective rate 0.0%, mean tail p95 max abs 0.0622, max policy max abs 0.2901.
- Final gohome two-stage gate: 23 / 24 event recall, 0 pre-tail false-positive episodes, 0 pre-tail active frames.
- Remaining early gohome frames: 8 total, all dwell/tail-candidate frames; no unsafe pre-tail request.
- Missed gohome event: `episode_83`, same as E33.
- Gate latency on CPU for phase + tail candidate + gohome eligibility MLPs together: mean 0.027 ms, p50 0.026 ms, p95 0.035 ms, max 0.109 ms over 2,000 sampled steps.

Decision:
- E35 passes as a lightweight runtime gate smoke. The final saved gate models preserve the E34 offline safety properties; the OOF-to-final-model shift did not reintroduce action tail motion or unsafe pre-tail gohome requests.
- The next blocker is not the auxiliary gate latency. It is packaging and validating the full ACT image inference path plus these three gates in one runtime/replay harness, and separately checking visual-domain robustness.

### E36 Candidate Package Manifest

Reason:
- E35 proves the final lightweight gate artifacts preserve the offline safety gates, but deployment and future replay still need a hard artifact contract.
- The package should prevent mixing the wrong ACT checkpoint, dataset stats, phase gate, gohome gates, or evidence files when moving from offline analysis toward runtime smoke.

Implementation:
- Added `scripts/e36_build_policy_gate_package_manifest.py`.
- Added focused tests in `testbed/tests/test_e36_package_manifest.py`.
- The script writes a compact manifest with SHA-256, file size, selected gate names, compact metric evidence, and a short field smoke checklist.
- The verifier recomputes file existence, size, and SHA-256 from disk, so stale or swapped artifacts fail before any runtime use.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e36_e34_runtime_gate_candidate`.
- Manifest: `candidate_package_manifest.json`.
- Verification report: `candidate_package_verify.json`.
- Field checklist: `field_smoke_checklist.md`.

E36 results:
- Package verification: pass.
- Required artifacts checked: 25 / 25.
- Selected phase gate: `simple_0.15_s0.50`.
- Selected gohome gate: `learned_tail_t0.97_tc10_e0.80_ec3`.
- Manifest evidence carries E35 action MAE 0.0414, RMSE 0.0859, gohome event recall 95.8%, gohome pre-tail false positives 0 / 24 episodes, and gate CPU p95 latency 0.035 ms.

Decision:
- E36 passes the packaging/verifier gate. It reduces version-mismatch and handoff risk for the current best offline candidate.
- E36 is not a new policy and does not prove full ACT image inference latency, runtime wiring, or visual-domain robustness. Those remain the next blockers.

### E37 Full ACT Gate Smoke

Reason:
- E35 used cached E16 actions and cached E16 intent probabilities. That checks the lightweight gate models, but it does not prove that the packaged ACT checkpoint can run the image path and produce action plus intent from the same frame stream.
- E37 tests the next deployment-adjacent contract: verify the E36 manifest, load the packaged ACT bundle and gate models, decode real HDF5 images, run ACT, take query-0 intent probabilities from the same forward pass, and apply the final phase/gohome gates.

Implementation:
- Added `scripts/e37_full_act_gate_smoke.py`.
- Added focused tests in `testbed/tests/test_e37_full_act_gate_smoke.py`.
- The script writes raw ACT replay, phase-gated replay, per-episode intent/phase/gohome probabilities, gohome events, latency summaries, and a compact JSON summary.
- Selected episodes for the smoke: `episode_73` as a strict-pass normal sample, and `episode_79` because E32 had pre-tail risk there before E33's two-stage candidate gate.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e37_full_act_gate_smoke_ep73_ep79`.
- Summary: `full_act_gate_smoke_summary.json`.
- Gohome events: `full_act_gohome_events.csv`.
- Raw ACT replay: `raw_action_replay`.
- Phase-gated replay: `phase_gated_action_replay`.
- Gate probabilities: `episode_gate_probs`.
- Startup/tail deadzone gate: `phase_gated_deadzone_runtime_scaled`.

E37 results:
- Package manifest verification: pass.
- Full ACT raw replay matches the existing cached E16 replay: max absolute action difference is about 1.47e-5 on `episode_73` and 5.92e-6 on `episode_79`.
- Raw sampled replay: MAE 0.0405, RMSE 0.0994.
- Phase-gated sampled replay: MAE 0.0354, RMSE 0.0781.
- Startup first-effective 40 after phase gate over the two episodes: policy effective 77.5%, same-axis/same-direction 96.7%, extra/wrong 0.0%.
- Tail stability after phase gate: 0 / 46 effective frames, tail effective rate 0.0%, max policy max abs 0.0984.
- Gohome events: 2 / 2 event recall, 0 pre-tail false-positive episodes, 0 pre-tail active frames.
- `episode_79` still has 2 early frames before `eligible_start`, but both are dwell/tail-candidate frames, not true pre-tail frames.
- Full ACT per-step latency in this offline HDF5 decode loop: p95 20.3 ms. This includes Python/HDF5 image decode and is a smoke number, not a production optimized runtime benchmark.

Decision:
- E37 passes as a sampled full-ACT package smoke. It confirms the packaged checkpoint, stats, config, and gate models can run together from real HDF5 images without relying on cached actions/probabilities.
- This does not prove full-dataset behavior or visual-domain robustness. The next step should broaden E37 coverage, especially across visual domains and known difficult episodes, before treating the candidate as field-test ready.

### E38 Full ACT All-Train-Ready Gate Smoke

Reason:
- E37 proved the package mechanics on two episodes. E38 extends the same full ACT image-inference path to all 24 train-ready episodes.
- This checks whether the E35/E36 candidate still passes the three user metrics when actions and intent probabilities are produced live from HDF5 images rather than read from cached replay artifacts.

Implementation:
- Reused `scripts/e37_full_act_gate_smoke.py` with `--max-episodes 24`.
- Generated a compatible selected manifest in the E38 output dir for downstream startup/tail deadzone gate tooling.
- Ran `scripts/deadzone_startup_tail_eval.py` on the E38 phase-gated replay.
- Joined E38 episode metrics with E27 four-view and eye2 visual-domain labels for domain-level diagnostics.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e38_full_act_gate_smoke_all_train_ready`.
- Summary: `full_act_gate_smoke_summary.json`.
- Gohome events: `full_act_gohome_events.csv`.
- Raw ACT replay: `raw_action_replay`.
- Phase-gated replay: `phase_gated_action_replay`.
- Startup/tail deadzone gate: `phase_gated_deadzone_runtime_scaled`.
- Visual-domain summaries: `visual_domain_summaries/domain_summary_four_k6.csv` and `visual_domain_summaries/domain_summary_eye2_k6.csv`.

E38 results:
- Package manifest verification: pass.
- Full ACT raw replay matches the cached E16 replay closely: worst episode `episode_74`, max absolute action difference 6.53e-4, mean absolute difference 2.03e-5; mean per-episode action difference is 9.89e-6.
- Raw full-ACT replay: MAE 0.0448, RMSE 0.1000.
- Phase-gated full-ACT replay: MAE 0.0414, RMSE 0.0859.
- Startup first-effective 40 after phase gate: policy effective 64.9%, same-axis/same-direction 78.4%, extra/wrong 5.6%; 21 / 24 episodes have startup policy effective >= 50%.
- Tail stability after phase gate: 0 / 504 effective frames, tail effective rate 0.0%, mean tail p95 max abs 0.0622, max policy max abs 0.2901.
- Gohome two-stage gate: 23 / 24 event recall, 0 pre-tail false-positive episodes, 0 pre-tail active frames. Missed event remains `episode_83`.
- Full ACT p95 latency in this offline HDF5 decode loop: 20.3 ms. This includes Python/HDF5 image decode and is not a production optimized runtime benchmark.

Domain diagnostics:
- Four-view dominant domains:
  - `texture_domain_3`: 12 episodes, MAE 0.0422, startup effective 57.9%, same-dir 74.1%, extra/wrong 1.7%, pre-tail FP 0.
  - `texture_domain_4`: 7 episodes, MAE 0.0401, startup effective 72.1%, same-dir 82.7%, extra/wrong 5.5%, pre-tail FP 0.
  - `texture_domain_5`: 5 episodes, MAE 0.0418, startup effective 71.5%, same-dir 82.6%, extra/wrong 15.4%, pre-tail FP 0.
- Eye2 dominant domains:
  - `texture_domain_2`: 17 episodes, MAE 0.0409, startup effective 64.9%, same-dir 80.6%, extra/wrong 3.4%, pre-tail FP 0.
  - `texture_domain_4`: 6 episodes, MAE 0.0439, startup effective 62.5%, same-dir 71.8%, extra/wrong 12.8%, pre-tail FP 0.
  - `texture_domain_5`: 1 episode, MAE 0.0373, startup effective 80.0%, same-dir 80.0%, extra/wrong 0.0%, pre-tail FP 0.

Weak spots:
- Worst startup effective: `episode_74` has 0.0% startup policy effective and the highest MAE, so the phase gate may be suppressing a real startup segment there.
- Worst startup extra/wrong: `episode_100` has 54.5% extra/wrong despite 100.0% same-dir in expert-effective startup frames; `episode_76` has 38.2% extra/wrong.
- Visual-domain signal remains plausible but not conclusive: four-view `texture_domain_3` is weaker on startup effective, while eye2 `texture_domain_4` is weaker on MAE and extra/wrong.

Decision:
- E38 passes the full-dataset full-ACT mechanics and safety gates. It materially reduces the risk that E35 was an artifact of cached actions/probabilities.
- The candidate is still not field-ready from this evidence alone. The next improvement should target startup recall and extra/wrong intent on the identified weak episodes/domains, while preserving the tail and gohome safety gates.

### E39 Startup Failure Diagnostics

Reason:
- E38 identifies the remaining weak spots but not the source. A new training or gate change should not start until we know whether failures come from raw ACT action magnitude, phase-gate suppression, or the startup-window metric itself.
- The diagnostic compares expert action, raw ACT action, phase-gated action, and phase probability inside each episode's first expert-effective startup window.

Implementation:
- Added `scripts/e39_startup_failure_diagnostics.py`.
- Added focused tests in `testbed/tests/test_e39_startup_failure_diagnostics.py`.
- The script reads E38 raw/gated replays, E38 per-episode gate probability files, runtime-scaled deadzone thresholds, and E27 visual-domain labels.
- It writes per-episode startup diagnostics plus a compact summary of worst startup effective and worst extra/wrong episodes.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e39_startup_failure_diagnostics`.
- Per-episode table: `startup_diagnostics.csv`.
- Summary: `startup_diagnostic_summary.json`.

E39 results:
- Mean gated startup metrics match E38: 64.9% policy effective, 78.4% same-dir, 5.6% extra/wrong.
- Total raw-to-gated lost same-dir frames across all startup windows: 15.
- Total expert-effective but phase-inactive frames: 35.
- `episode_74` is not a phase-gate problem. Phase is active 87.5% of the startup window, but raw ACT has 0 effective frames. Expert bucket+ is effective for 34 frames; raw bucket+ max is 0.5438 while the runtime deadzone threshold is 0.544.
- `episode_100` is not phase-gate-created extra/wrong. Raw and gated both have bucket+ effective for 33 frames while expert bucket+ is effective for only 15 frames. This is over-duration near the same axis/direction, not a new wrong axis from the gate.
- `episode_76` similarly keeps boom- effective for 34 frames while expert boom- is effective for 27 frames. Phase is active 100% of the window.
- `episode_90` bucket+ over-duration is partially reduced by the phase gate, from 29 raw frames to 25 gated frames, while expert is 20 frames.
- `episode_98` shows a true under-shoot: expert boom- is effective for 14 frames, while raw/gated boom- never crosses the 0.357 threshold.

Decision:
- Do not spend the next slice on phase-gate-only threshold sweeps. The worst startup misses and extra/wrong episodes are already visible in raw ACT action.
- The next improvement should target action amplitude and duration around runtime-scaled deadzone thresholds, likely with a startup/window-conditioned objective or post-hoc action calibration that can be evaluated before retraining. Any such candidate must preserve E38's tail 0 / 504 and gohome pre-tail 0 / 24 gates.

### E40 Deadzone Snap Calibration Probe

Reason:
- E39 showed two different startup failure modes: under-threshold intended motion (`episode_74`, `episode_98`) and over-duration same-direction motion (`episode_100`, `episode_90`, `episode_76`).
- Before retraining, E40 tests whether a minimal post-hoc calibration can recover under-threshold intended motion without violating tail stop stability.
- This is a diagnostic probe, not a runtime recommendation: it snaps only phase-active actions that are already within a small margin below the runtime-scaled directional deadzone.

Implementation:
- Added `scripts/e40_deadzone_snap_probe.py`.
- Added focused tests in `testbed/tests/test_e40_deadzone_snap_probe.py`.
- The snap rule is causal and local to the current action vector: if the phase gate is active and an axis value is in `[threshold - margin, threshold)`, set it to `threshold + epsilon`; similarly for negative directions.
- The probe materializes one replay per margin, then reuses the standard deadzone startup/tail gate evaluator.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e40_deadzone_snap_probe`.
- Scan table: `snap_probe_scan.csv`.
- Gate summary: `snap_probe_gate_summary.csv`.
- Manifest: `snap_probe_manifest.json`.
- Materialized replay dirs: `snap_m0010`, `snap_m0030`, `snap_m0050`, `snap_m0100`, `snap_m0200`.

E40 results:

| Candidate | Snapped Frames | Replay MAE | Replay RMSE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Tail Effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E38 baseline | 0 | 0.04144 | 0.08591 | 64.9% | 78.4% | 5.6% | 0 / 504 |
| snap margin 0.001 | 22 | 0.04144 | 0.08591 | 65.1% | 78.6% | 5.6% | 0 / 504 |
| snap margin 0.003 | 65 | 0.04144 | 0.08591 | 66.0% | 79.6% | 5.7% | 0 / 504 |
| snap margin 0.005 | 111 | 0.04144 | 0.08591 | 66.7% | 80.1% | 6.0% | 0 / 504 |
| snap margin 0.010 | 227 | 0.04144 | 0.08591 | 68.4% | 82.0% | 6.3% | 0 / 504 |
| snap margin 0.020 | 444 | 0.04144 | 0.08591 | 71.4% | 85.2% | 6.8% | 0 / 504 |

Episode-level interpretation:
- `episode_74` is the clearest under-threshold recovery case. Startup effective improves from 0.0% at baseline to 45.0% at margin 0.020, with no extra/wrong increase, because the raw bucket+ action was just below the bucket+ threshold.
- `episode_100` remains an over-duration problem. It already has 100.0% same-direction coverage at baseline and high extra/wrong; snapping does not fix the duration mismatch and slightly increases extra/wrong at larger margins.
- `episode_98` confirms the risk of a global snap. Larger margins improve some under-threshold motion but also increase extra/wrong in an episode where the failing axis is boom-.

Decision:
- Do not deploy a global snap rule as-is. It is useful evidence that deadzone-boundary calibration can recover startup motion, but the unconstrained margin tradeoff is not clean enough.
- The next candidate should be direction- and window-targeted: only boost the axis/direction supported by expert-like intent probability or a startup phase label, and add duration control so same-direction over-extension does not become more common.
- E40 shifts the next implementation direction from "more phase gating" to "intended-direction amplitude/duration calibration or training loss" while preserving E38's tail 0 / 504 and gohome pre-tail 0 / 24 as hard gates.

### E41 Intent-Targeted Deadzone Snap Probe

Reason:
- E40 showed near-deadzone calibration can recover some startup movement, but a global snap also increases extra/wrong as margin grows.
- E41 tests the narrowest available direction filter without retraining: use the same-frame ACT intent probabilities produced during E38 full-ACT replay, and only snap an axis/direction when that matching intent probability crosses a threshold.
- This keeps the probe tied to the existing ACT visual/state representation rather than adding a hand-coded axis rule.

Implementation:
- Added `scripts/e41_intent_targeted_snap_probe.py`.
- Added focused tests in `testbed/tests/test_e41_intent_targeted_snap_probe.py`.
- The snap rule requires all three conditions: phase gate active, policy action within the near-deadzone margin, and matching axis/direction intent probability above the configured threshold.
- Scanned margins 0.003, 0.005, 0.010, and 0.020 with intent thresholds 0.50, 0.70, 0.80, and 0.90.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e41_intent_targeted_snap_probe`.
- Scan table: `intent_targeted_snap_scan.csv`.
- Gate summary: `intent_targeted_snap_gate_summary.csv`.
- Manifest: `intent_targeted_snap_manifest.json`.

E41 selected results:

| Candidate | Snapped Frames | Replay MAE | Replay RMSE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Tail Effective | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| E38 baseline | 0 | 0.04144 | 0.08591 | 64.9% | 78.4% | 5.6% | 0 / 504 | Reference |
| E40 global m0.020 | 444 | 0.04144 | 0.08591 | 71.4% | 85.2% | 6.8% | 0 / 504 | Stronger startup, too much extra/wrong |
| E41 m0.020 i0.50 | 349 | 0.04144 | 0.08591 | 69.8% | 83.9% | 6.1% | 0 / 504 | Useful but crosses 6.0% extra/wrong line |
| E41 m0.020 i0.70 | 279 | 0.04144 | 0.08591 | 68.5% | 82.6% | 5.9% | 0 / 504 | Best current calibration tradeoff |
| E41 m0.010 i0.50 | 181 | 0.04144 | 0.08591 | 67.7% | 81.5% | 5.9% | 0 / 504 | Smaller, safer alternative |
| E41 m0.005 i0.70 | 69 | 0.04144 | 0.08591 | 65.9% | 79.5% | 5.8% | 0 / 504 | Very conservative, modest gain |

Episode-level interpretation:
- `episode_74` confirms the threshold issue. E41 m0.020 i0.50 recovers startup effective to 45.0% with no extra/wrong, matching the E40 global recovery; E41 m0.020 i0.70 recovers only 25.0%, showing that the intent head is less confident in part of the needed bucket+ segment.
- `episode_100` is still an over-duration case. E41 m0.020 i0.70 keeps it at 82.5% effective / 100.0% same-dir / 54.5% extra, essentially the E38 baseline pattern, so amplitude snap is not the right fix for that episode.
- `episode_98` does not improve under the selected E41 gates and keeps its baseline extra/wrong. This suggests the boom- under-shoot there may need a training-side or lower-threshold intent treatment rather than a high-confidence snap.

Decision:
- E41 is a better post-hoc calibration primitive than E40 global snap, because it keeps the tail gate clean and improves startup while controlling extra/wrong.
- It is still not a final deployment rule. The remaining action problem has two parts: amplitude just below deadzone on some episodes and over-duration on others. A deployable gate or training loss needs both intended-direction amplitude promotion and duration/stop control.
- Next options should compare E41-style calibration against a training-time objective that promotes only intended-direction near-threshold actions and penalizes persistence after expert intent ends.

### E42 Startup Intent-Overlap Diagnostic

Reason:
- E41 still leaves over-duration episodes such as `episode_100` and `episode_76`.
- Before adding another threshold or calibration rule, we need to know whether same-frame ACT intent probabilities actually separate correct same-direction startup motion from extra/wrong startup motion.
- If extra frames have low intent, a stricter intent-targeted release gate is plausible. If extra frames also have high intent, the next fix must include temporal release or duration control.

Implementation:
- Added `scripts/e42_startup_intent_overlap_diagnostic.py`.
- Added focused tests in `testbed/tests/test_e42_startup_intent_overlap_diagnostic.py`.
- The script compares E38 phase-gated policy actions against expert actions in each episode's first expert-effective startup window.
- For every axis/direction event it separates three sets: same-direction, policy extra/wrong, and expert missing. It then reports intent probability statistics and high-intent fractions for the extra/wrong set.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e42_startup_intent_overlap_diagnostic`.
- Per-episode table: `startup_intent_overlap_by_episode.csv`.
- Summary: `startup_intent_overlap_summary.json`.

E42 results:
- Across 24 startup windows: 583 same-direction events, 40 extra/wrong events, and 169 missing expert-direction events.
- Mean per-episode same-direction intent: 0.837.
- Mean per-episode extra/wrong intent: 0.721.
- Mean per-episode missing-direction intent: 0.752.
- High-intent extra/wrong fractions: 90.0% >= 0.50, 60.4% >= 0.70, 28.9% >= 0.80, 20.0% >= 0.90.
- Worst extra/wrong episodes:
  - `episode_100`: 18 extra bucket+ events; extra intent mean 0.735; 72.2% >= 0.70 and 44.4% >= 0.80.
  - `episode_76`: 13 extra boom- events; extra intent mean 0.949; 100.0% >= 0.80.
  - `episode_90`: 5 extra bucket+ events; extra intent mean 0.718; 80.0% >= 0.70.
- Worst missing episodes:
  - `episode_74`: 34 missing bucket+ events; missing intent mean 0.614.
  - `episode_83`: 18 missing events; missing intent mean 0.719.
  - `episode_98`: 15 missing events; missing intent mean 0.809.

Decision:
- Do not pursue a threshold-only release gate on the current ACT intent head. The same-frame intent probability remains high for many over-duration extra frames.
- Keep E41 as the current best post-hoc amplitude calibration probe, but treat it as half of the solution.
- The next training-side candidate should add temporal release semantics: promote intended-direction action near deadzone when expert intent is active, and penalize persistence after the expert-direction event ends. This is different from generic idle suppression because it targets the same axis/direction continuation that creates startup extra/wrong.

### E43 Direction Release Gate Probe

Reason:
- E42 showed that same-frame intent probabilities alone do not reliably signal when to release a direction.
- E43 tests whether a small learned 8-way direction gate over the same features as E22 (`intent_prob + qpos + qvel`) can predict expert-effective directions and suppress policy directions after expert intent ends.
- This is intentionally a release-only probe: it cannot increase action amplitude.

Implementation:
- Added `direction_effective_labels` and `apply_direction_gate_to_actions` in `testbed/testbed/policies/phase_gate.py`.
- Added `scripts/e43_direction_gate_probe.py`.
- Added focused tests in `testbed/tests/test_phase_gate.py` and `testbed/tests/test_e43_direction_gate_probe.py`.
- The probe trains episode-heldout 8-way MLP direction probabilities, materializes threshold/scale replays, and saves a final direction gate bundle for later runtime smoke if the probe becomes useful.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e43_direction_gate_probe`.
- Scan table: `direction_gate_scan.csv`.
- Gate summary: `direction_gate_summary.csv`.
- OOF direction probabilities: `direction_probs/*.npz`.
- Final model: `direction_gate_model.pt` and `direction_gate_model_metadata.json`.

E43 selected results:

| Candidate | Replay MAE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Tail Effective | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E38 baseline | 0.0414 | 64.9% | 78.4% | 5.6% | 0 / 504 | Reference |
| dir t0.70 s0.50 | 0.0448 | 64.7% | 78.4% | 5.3% | 0 / 504 | Extra improves slightly, MAE/startup not better |
| dir t0.80 s0.50 | 0.0466 | 64.0% | 77.8% | 5.0% | 0 / 504 | More release, too much action-closeness loss |
| dir t0.90 s0.50 | 0.0499 | 55.7% | 67.0% | 2.4% | 0 / 504 | Reject; suppresses true startup |

Decision:
- Reject E43 as a standalone action candidate. It can reduce extra/wrong only by suppressing action, so it does not solve missing startup motion and worsens action closeness.
- Keep the direction-gate machinery as a diagnostic and possible runtime component only if paired with amplitude promotion and a better duration objective.

### E44 E41 Snap Plus E43 Direction Release Probe

Reason:
- E41 improves missing startup amplitude but leaves over-duration cases.
- E43 reduces some extra/wrong but cannot recover missing startup motion.
- E44 tests whether the two cheap post-hoc mechanisms compose into a better tradeoff before spending a new training run.

Implementation:
- Reused `scripts/e43_direction_gate_probe.py` with E41 `snap_m0200_i70` as the source replay.
- Scanned direction thresholds 0.50, 0.70, 0.80, 0.90 and inactive scales 0.25, 0.50.
- Reran standard startup/tail gates and wrote `combined_direction_release_summary.csv`.

Artifacts:
- Output root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e44_e41snap_e43_direction_release_probe`.
- Combined summary: `combined_direction_release_summary.csv`.
- Candidate replay dirs: `dir_t50_s25`, `dir_t50_s50`, `dir_t70_s25`, `dir_t70_s50`, `dir_t80_s25`, `dir_t80_s50`, `dir_t90_s25`, `dir_t90_s50`.

E44 selected results:

| Candidate | Replay MAE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Tail Effective | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| E38 baseline | 0.0414 | 64.9% | 78.4% | 5.6% | 0 / 504 | Reference |
| E41 `m0200_i70` | 0.0414 | 68.5% | 82.6% | 5.9% | 0 / 504 | Best amplitude-only probe |
| E44 `dir_t70_s50` | 0.0448 | 68.2% | 82.5% | 5.6% | 0 / 504 | Best combined release tradeoff, but MAE regresses |
| E44 `dir_t80_s50` | 0.0466 | 67.2% | 81.4% | 5.2% | 0 / 504 | More release, too much MAE/startup loss |

Episode-level interpretation:
- `episode_90` and `episode_98` improve under stricter release gates, confirming there are some low/medium-intent extra frames that direction release can remove.
- `episode_100` remains high extra/wrong under `dir_t70_s50`: 82.5% startup effective, 100.0% same-dir, 54.5% extra/wrong. This matches E42: many extra bucket+ frames are still high-intent.
- `episode_76` also remains high extra/wrong because its extra boom- frames have very high direction probability.

Decision:
- E44 is not a new best candidate because the release improvement is bought with action-closeness loss and does not fix the worst high-intent over-duration episodes.
- The next meaningful training experiment should explicitly model temporal release / persistence. A practical objective is: promote expert-active axis/direction near deadzone, suppress same-axis/direction persistence after the expert direction turns off, and keep tail/gohome gates as hard offline checks.

### E45 Temporal Release Training

Reason:
- E42 shows same-frame intent probability is not a reliable release signal: extra frames often remain high-intent.
- E43/E44 show post-hoc direction release can reduce some extra/wrong but costs MAE and does not solve high-intent over-duration.
- E45 moves the release idea into training: after an expert axis/direction falls below the runtime deadzone, penalize the policy only if it continues crossing the same directional deadzone during a short release window.

Implementation:
- Added `temporal_release_loss` to `testbed/testbed/policies/act/adapter.py`.
- Added `temporal_release_loss` passthrough in `testbed/testbed/runtime/_train.py`.
- The loss is default-off and uses the same runtime-scaled deadzone threshold JSON as existing deadzone/intent losses.
- The release mask is derived dynamically from the expert action chunk; no dataset schema or handoff-mask path is required.
- Added focused tests in `testbed/tests/test_act_deadzone_loss.py` for positive and negative direction release.

Smoke:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e45_temporal_release_eye2_smoke.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e45_temporal_release_eye2_smoke`.
- Result: 1 epoch completed and emitted `temporal_release_pos_loss`, `temporal_release_neg_loss`, and `temporal_release_loss` in train/val logs.
- Initial random smoke mostly has zero release loss because it often stays below the runtime deadzone in release windows; this is acceptable for this objective. In the full E45 training run, non-zero release loss appears in some train/val batches once predictions cross release windows.

Full training:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e45_temporal_release_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e45_temporal_release_eye2`.
- Reused split: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_eye2/train_val_split.yaml`.
- Loss config: E16 weighted intent head plus `temporal_release_loss.enabled=true`, `weight=0.05`, `release_window_steps=4`.
- Logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e45_temporal_release_eye2_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e45_temporal_release_eye2_train.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e45_temporal_release_eye2_train.pid`
- Launched at 2026-07-09 in background with PID 934518.
- Early log check: process alive, metadata/config/stats written, and the log shows non-zero temporal release terms in some batches, e.g. `temporal_release_neg_loss:0.0566` at epoch 30 validation.
- Progress checks: at about epoch 370 and later 1287 / 2000, `run_metadata.status=started`; temporal release terms were non-zero in some train/val batches.
- Completed at 2026-07-09 14:44 CST. Training log reports best epoch 1999 and best val loss 0.082703.
- Postprocess watcher:
  - PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e45_postprocess.pid`
  - Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e45_postprocess.log`
  - Command: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e45_postprocess_command.sh`
  - Behavior: wait for training PID 934518 to exit, require `run_metadata.status=completed`, replay `policy_best.ckpt` over all train-ready episodes, run `deadzone_startup_tail_eval.py`, and write `e45_gate_compact_summary.json`.

Evaluation:
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e45_temporal_release_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e45_temporal_release_eye2_gate_runtime_scaled`.
- All 24 train-ready episodes were replayed from `policy_best.ckpt`.
- Compact summary: MAE 0.04477, RMSE 0.08866, startup effective 58.85%, startup same-dir 65.37%, startup extra/wrong 11.15%, tail 6 / 504 effective frames.
- Main visible regressions:
  - `episode_74` startup effective is only 7.5% and same-dir 8.8%.
  - `episode_100` startup extra/wrong rises to 90.9%.
  - `episode_83` has all 6 tail frames above the boom deadzone, accounting for the full 6 / 504 tail failure.

Decision:
- Reject E45 as a candidate. It fails all three target metrics compared with E38/E41/E44: action closeness is worse, startup movement intent is weaker, and tail stop stability is no longer clean.
- Do not scale this exact temporal-release objective to four views. The evidence suggests this release-only training penalty is not preserving the should-move side of the deadzone-aware contract.

### E46 Weak Temporal Release Training

Reason:
- E45 failed clearly, but it changed two release-loss knobs at once relative to E16: non-zero temporal release weight and a 4-step release window.
- E46 is a narrow sensitivity check, not a new architecture: keep E16/E45 data, split, views, intent head, and ACT shape unchanged, then reduce only the release penalty strength and window.
- If E46 still weakens startup or creates tail crossings, the temporal-release-loss training direction should be deprioritized in favor of data/label or runtime-gate approaches.

Planned config:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e46_temporal_release_weak_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e46_temporal_release_weak_eye2`.
- Loss config: E16 weighted intent head plus `temporal_release_loss.enabled=true`, `weight=0.01`, `release_window_steps=2`.

Launch:
- Config validation: YAML parsed; dataset, manifest, split, and threshold JSON paths exist; ckpt dir was absent before launch.
- Compile check: `adapter.py` and `_train.py` compiled with current `PYTHONPATH`.
- Training PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e46_temporal_release_weak_eye2_train.pid` (`948696` at launch).
- Training log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e46_temporal_release_weak_eye2_train.log`.
- Postprocess watcher PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e46_postprocess.pid` (`948764` at launch).
- Postprocess watcher log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e46_postprocess.log`.
- Early log check: process alive, stats/config/metadata written, and weak `temporal_release_loss` terms appear in train/val logs.
- Completed at 2026-07-09 15:07 CST. Training log reports best epoch 1580 and best val loss 0.110259.

Acceptance gate:
- E46 must not regress below E38/E41 on tail stability; any tail effective crossing rejects it as a candidate.
- E46 must recover startup effective/same-dir close to E16/E38, or improve over E45 without increasing extra/wrong.
- If E46 only improves MAE while startup remains below E38 or extra/wrong remains elevated, treat it as a negative sensitivity result.

Evaluation:
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e46_temporal_release_weak_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e46_temporal_release_weak_eye2_gate_runtime_scaled`.
- All 24 train-ready episodes were replayed from `policy_best.ckpt`.
- Compact summary: MAE 0.04255, RMSE 0.08914, startup effective 48.96%, startup same-dir 52.02%, startup extra/wrong 11.75%, tail 0 / 504 effective frames.
- Main visible regressions:
  - `episode_74` has 0 / 40 policy effective startup frames despite 34 expert effective frames.
  - `episode_100` startup extra/wrong is 100.0%.
  - `episode_90` startup extra/wrong is 83.3%.

Decision:
- Reject E46. Weakening the release loss recovers tail stability but further suppresses startup should-move behavior.
- E45 and E46 together are a negative sensitivity result for training-time `temporal_release_loss`; do not continue by sweeping release weight/window. The next useful direction should avoid release-only action loss and instead use either explicit amplitude calibration with better release evidence, or a separate candidate/gate model with temporal features.

### E47 Temporal-Context Direction Gate Probe

Reason:
- E41 fixed part of the near-deadzone amplitude problem by snapping only intent-supported directions, but same-frame intent could not distinguish valid same-direction motion from over-duration extra motion.
- E42/E43 showed that a same-frame direction gate is not enough: extra/wrong frames often still have high intent probability, and suppress-only gates hurt MAE.
- E47 keeps the E41 amplitude primitive and tests whether temporal context over intent/qpos/qvel can provide a better release signal without retraining ACT.

Implementation:
- Added `scripts/e47_temporal_direction_gate_probe.py`.
- The probe reuses the E43 episode-heldout 8-way direction-gate machinery, but expands each frame with temporal context offsets over base features: intent probabilities, qpos, and qvel.
- Default offsets are `[-10, -5, -2, -1, 0, 1, 2, 5, 10]`.
- Added `testbed/tests/test_e47_temporal_direction_gate_probe.py` to lock the edge-padding semantics used when context offsets fall before the first or after the last frame.
- Source replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e41_intent_targeted_snap_probe/snap_m0200_i70`.
- Intent probabilities: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e20_e16_intent_head_gate_scan/intent_probs`.

Artifacts:
- Initial scan: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e47_temporal_direction_gate_probe`.
- Scale-0.75 scan: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e47b_temporal_direction_gate_probe_s75`.
- Main-window comparison: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e47b_t50_s75_vs_refs_deadzone_window_eval`.
- Non-causal combined candidate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e48_e47b_action_e33_gohome_combined_candidate`.

Results:

| Candidate | Replay MAE | Replay RMSE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Main Effective | Main Same Dir | Main Extra/Wrong | Tail Effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E38 phase-gated baseline | 0.0414 | 0.0859 | 64.9% | 78.4% | 5.6% | 94.8% | 94.5% | 1.2% | 0 / 504 |
| E41 `snap_m0200_i70` | 0.0414 | 0.0859 | 68.5% | 82.6% | 5.9% | 95.3% | 95.2% | 1.1% | 0 / 504 |
| E44 `dir_t70_s50` | 0.0448 | n/a | 68.2% | 82.5% | 5.6% | n/a | n/a | n/a | 0 / 504 |
| E47 `tdir_t50_s50` | 0.0440 | n/a | 67.3% | 81.4% | 5.3% | n/a | n/a | n/a | 0 / 504 |
| E47b `tdir_t50_s75` | 0.0406 | 0.0859 | 67.3% | 81.4% | 5.3% | 95.3% | 95.2% | 1.1% | 0 / 504 |

Decision:
- `E47b tdir_t50_s75` is the best non-causal action-only offline tradeoff so far. It improves action MAE versus E38/E41, improves startup effective/same-dir versus E38, lowers startup extra/wrong versus E38/E41/E44, preserves main-motion coverage, and keeps tail crossings at 0 / 504.
- E48 confirms that combining E47b with E33 keeps gohome recall at 23 / 24 and pre-tail false positives at 0, but this is still an offline upper-bound diagnostic.
- E47b/E48 are not runtime candidates because the default context offsets include future frames (`+1/+2/+5/+10`). A real-time candidate must use causal offsets only.

### E49/E50 Causal Temporal-Context Direction Gate

Reason:
- E47b looked better than E38/E41 on several offline gates, but future context would not be available in online inference.
- E49 repeats the same temporal direction-gate structure with only historical/current offsets: `[-10, -5, -2, -1, 0]`.
- E50 combines the best causal E49 action replay with the already-selected E33 two-stage gohome gate.

Artifacts:
- Causal scan: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e49_causal_temporal_direction_gate_probe_s75`.
- Causal startup/tail gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e49_causal_temporal_direction_gate_probe_s75_gate_runtime_scaled`.
- Causal reference window comparison: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e49_causal_t50_s75_vs_refs_deadzone_window_eval`.
- Causal combined candidate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e50_e49_causal_action_e33_gohome_combined_candidate`.

Results:

| Candidate | Causal | Replay MAE | Replay RMSE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Main Effective | Main Same Dir | Main Extra/Wrong | Full Extra/Wrong | Tail Effective | Gohome Recall | Gohome Pre-tail FP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E38 phase-gated baseline | yes | 0.0414 | 0.0859 | 64.9% | 78.4% | 5.6% | 94.8% | 94.5% | 1.2% | 7.4% | 0 / 504 | 95.8% | 0 |
| E41 `snap_m0200_i70` | yes | 0.0414 | 0.0859 | 68.5% | 82.6% | 5.9% | 95.3% | 95.2% | 1.1% | 7.8% | 0 / 504 | n/a | n/a |
| E47b/E48 `tdir_t50_s75` | no | 0.0406 | 0.0859 | 67.3% | 81.4% | 5.3% | 95.3% | 95.2% | 1.1% | 5.6% | 0 / 504 | 95.8% | 0 |
| E49/E50 `tdir_t50_s75` | yes | 0.0408 | 0.0865 | 67.9% | 81.9% | 5.8% | 95.3% | 95.2% | 1.1% | 7.7% | 0 / 504 | 95.8% | 0 |

Decision:
- E50 is the best real-time-causal integrated offline candidate so far. It improves action closeness and startup intent versus E38, keeps main-motion coverage close to E41, keeps tail stable, and preserves E33's conservative gohome timing.
- It is not a live-control candidate yet. The next verification gate is package/runtime smoke with the causal temporal gate model, plus a focused review of full-window extra/wrong because E49's full-window extra is slightly worse than E38 and close to E41.

### E51 Full-ACT Causal Temporal Gate Smoke

Reason:
- E49/E50 are based on cached actions and episode-heldout direction probabilities. Runtime must use the final saved causal temporal gate model and ACT image inference outputs.
- E51 verifies the full chain: HDF5 images -> ACT action and query-0 intent -> final phase gate -> E41 intent-targeted snap -> final causal temporal direction gate -> E33 gohome request gates.
- This also checks that the E49 model artifact is causal at load time. A loader bug was found and fixed: older E49 `.pt` payloads inherited stale 16-dim E43 `feature_names`, so E51 now falls back to `temporal_direction_gate_model_metadata.json` when the payload feature names do not match the model feature dimension. E47 was also fixed to write temporal `feature_names` into future `.pt` payloads.

Artifacts:
- Script: `scripts/e51_full_act_temporal_gate_smoke.py`.
- Tests: `testbed/tests/test_e51_full_act_temporal_gate_smoke.py`.
- Sampled smoke: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_ep73_ep79`.
- Sampled startup/tail gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_ep73_ep79_deadzone_runtime_scaled`.
- Full all-train-ready smoke: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready`.
- Full startup/tail gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_full_act_causal_temporal_gate_smoke_all_train_ready_deadzone_runtime_scaled`.
- Full window comparison: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e51_fullact_vs_e38_e49_deadzone_window_eval`.

Results:

| Candidate | Scope | Replay MAE | Replay RMSE | Startup Effective | Startup Same Dir | Startup Extra/Wrong | Main Effective | Main Same Dir | Main Extra/Wrong | Full Extra/Wrong | Tail Effective | Gohome Recall | Gohome Pre-tail FP | ACT P95 | Gate P95 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E38 phase-gated baseline | full ACT, 24 eps | 0.0414 | 0.0859 | 64.9% | 78.4% | 5.6% | 94.8% | 94.5% | 1.2% | 7.4% | 0 / 504 | 95.8% | 0 | 20.3 ms | n/a |
| E49 `tdir_t50_s75` | cached OOF, 24 eps | 0.0408 | 0.0865 | 67.9% | 81.9% | 5.8% | 95.3% | 95.2% | 1.1% | 7.7% | 0 / 504 | n/a | n/a | n/a | n/a |
| E51 final temporal gate | full ACT, 24 eps | 0.0408 | 0.0865 | 68.4% | 82.6% | 5.7% | 95.3% | 95.2% | 1.1% | 7.5% | 0 / 504 | 95.8% | 0 | 20.3 ms | 0.0075 ms |

Decision:
- E51 is the best runtime-smoke candidate so far. It improves action closeness and startup same-direction intent versus E38, preserves tail stability and gohome pre-tail safety, and keeps main-motion coverage at the E49/E41 level.
- E51 supersedes E50 for decision-making because it uses the final saved causal temporal model and the full ACT image path. E50 remains useful as the cached OOF reference.
- E52 closes the package traceability gap: 23 / 23 declared artifacts verify by SHA-256 and the field smoke checklist includes the causal temporal direction model and E51 evidence.
- Residual behavior risk: E51 full-window extra/wrong is 7.45%, slightly above E38's 7.36% and below E49's 7.69%. This should be reviewed by episode before live motion, but it does not violate the current startup/tail/gohome gates.

### E29 E28 Phase-Gate Probe

Reason:
- E28 had good action closeness but weak startup and non-zero tail crossings.
- E29 tests the cheapest rescue path: reuse E28 action replay and E28 auxiliary intent probabilities with the same phase-gate tooling that produced E22b.

Artifacts:
- Intent probabilities: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e29_e28_four_camera_intent_probs/intent_probs`.
- Auto gate probe: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e29_e28_phase_gate_soft_scale_probe`.
- Auto materialized replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e29_e28_phase_gate_soft_scale_probe/hyst_o0.25_c0.10_s0.75`.
- Auto gate eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e29_e28_phase_gate_soft_scale_probe_gate_runtime_scaled`.
- E22b-style materialized replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e29b_e28_phase_gate_simple015_s050_probe/simple_0.15_s0.50`.
- E22b-style gate eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e29b_e28_phase_gate_simple015_s050_probe_gate_runtime_scaled`.

Results:

| Candidate | Replay MAE | Replay RMSE | Startup Policy Effective | Startup Same Dir | Startup Extra/Wrong | Main Policy Effective | Main Same Dir | Main Extra/Wrong | Start40 Policy Effective | Tail Effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E29 auto `hyst_o0.25_c0.10_s0.75` | 0.0404 | 0.0787 | 59.9% | 68.5% | 8.1% | 95.9% | 93.5% | 3.3% | 1.25% | 1 / 504 |
| E29b `simple_0.15_s0.50` | 0.0413 | 0.0809 | 58.9% | 67.3% | 7.9% | 96.3% | 93.9% | 3.3% | 1.25% | 1 / 504 |

Decision:
- Reject as best candidate. Both E29 variants improve action closeness and start40 quietness compared with raw E28, but both remain below E22b on startup same-direction coverage, main extra/wrong, and tail stability.
- E22b remains the current best original-image offline tradeoff.

### E28 Four-View Weighted ACT Intent Head

Reason:
- E27 proved the current eye2 E16-style policy has a real visual-domain gap under a held-out four-view texture-domain split.
- The missing comparison is an E16-style weighted intent head with all four GMSL views. E00/E02/E04 already cover four-view baseline/deadzone variants, but not the weighted auxiliary intent-head route.
- E28 is not a deadzone suppressor. It should be judged by the corrected contract: preserve or improve should-move startup/main coverage, keep extra/wrong low, and keep tail quiet.

Config and launch:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e28_weighted_intent_head_four_camera.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e28_weighted_intent_head_four_camera`.
- Split path: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform/train_val_split.yaml`.
- Verification before launch: YAML parsed, data/manifest/split/deadzone paths exist, checkpoint dir was free, and `adapter.py`, `detr_vae.py`, `_train.py`, and `actions/policy.py` compiled.
- Training launched at 2026-07-09 05:10 Asia/Shanghai with PID 728370. Completed at 2026-07-09 05:38 Asia/Shanghai.
- Training logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e28_weighted_intent_head_four_camera_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e28_weighted_intent_head_four_camera_train.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e28_weighted_intent_head_four_camera_train.pid`
- Training result: best epoch 1999, best val loss 0.08447404392063618.
- Postprocess watcher launched at 2026-07-09 05:11 Asia/Shanghai with PID 728925. Completed at 2026-07-09 05:45 Asia/Shanghai.
- Postprocess logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e28_postprocess.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e28_postprocess.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e28_postprocess.pid`
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e28_weighted_intent_head_four_camera_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e28_weighted_intent_head_four_camera_gate_runtime_scaled`.

E28 results:
- Replay: MAE 0.0433, RMSE 0.0869, policy p95 abs 0.8589, policy max abs 0.9595.
- Startup gate: policy effective 62.0%, same-axis/same-direction 69.2%, extra/wrong 10.3%.
- Main-motion gate: policy effective 96.8%, same-axis/same-direction 94.3%, extra/wrong 3.3%.
- Start40 gate: policy effective 5.9%, extra/wrong 4.2%.
- Tail gate: 6 / 504 effective frames, tail effective rate 1.19%, mean tail p95 max abs 0.1792, max policy max abs 0.4464.
- Decision: reject as current best candidate. E28 improves action closeness versus E16 and is close to E22b on MAE/RMSE, but it violates the stop-stability gate and does not recover startup should-move coverage. Extra views alone are not the next primary route.

### E27 Visual Domain Clustering and Held-Out Training

Visual-domain artifacts:
- Four-view clustering: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/visual_domain_clusters_four_k6`.
- Eye2 clustering: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/visual_domain_clusters_eye2_k6`.
- Each clustering used 24 train-ready episodes, 24 task-segment frames per episode, and produced 576 sampled frames. Outputs include `cluster_summary.json`, `sample_domains.csv`, `episode_domains.csv`, and `contact_sheets/texture_domain_*.jpg`.
- Four-view dominant episode domains:
  - `texture_domain_3`: 12 episodes (`episode_73`, `episode_74`, `episode_79`, `episode_83`, `episode_84`, `episode_85`, `episode_86`, `episode_87`, `episode_90`, `episode_92`, `episode_94`, `episode_104`).
  - `texture_domain_4`: 7 episodes (`episode_75`, `episode_76`, `episode_78`, `episode_80`, `episode_91`, `episode_93`, `episode_97`).
  - `texture_domain_5`: 5 episodes (`episode_82`, `episode_98`, `episode_99`, `episode_100`, `episode_102`).
- Eye2 dominant episode domains:
  - `texture_domain_2`: 17 episodes.
  - `texture_domain_4`: 6 episodes (`episode_73`, `episode_74`, `episode_82`, `episode_98`, `episode_99`, `episode_100`).
  - `texture_domain_5`: 1 episode (`episode_102`).
- Interpretation: frame clusters visibly track texture, bucket proximity, shadows, and stage/pose. The contact sheets are useful for manual labels, but the clusters are still partly phase-confounded; use episode domain proportions as evidence, not as final semantic class names.

Held-out split:
- Split file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/splits/visual_domain_four_k6_domain5_heldout.yaml`.
- Train episodes: 73, 74, 75, 76, 78, 79, 80, 83, 84, 85, 86, 87, 90, 91, 92, 93, 94, 97, 104.
- Val / held-out episodes: 82, 98, 99, 100, 102.
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e27_domain5_heldout_weighted_intent_head_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e27_domain5_heldout_weighted_intent_head_eye2`.
- Logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e27_domain5_heldout_weighted_intent_head_eye2_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e27_domain5_heldout_weighted_intent_head_eye2_train.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e27_domain5_heldout_weighted_intent_head_eye2_train.pid`
- Training status: completed at 2026-07-09 04:52 Asia/Shanghai. Best epoch 1750, best val loss 0.12222892977297306.
- Eval manifests:
  - Held-out only: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/manifests/e27_domain5_heldout_manifest.json`.
  - Train-domain only: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/manifests/e27_domain5_train_domain_manifest.json`.
- Postprocess automation:
  - PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e27_domain5_postprocess.pid`.
  - Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e27_domain5_postprocess.log`.
  - Master log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e27_domain5_postprocess.master.log`.
  - Started at 2026-07-09 04:36 Asia/Shanghai with PID 708084. Completed at 2026-07-09 05:04 Asia/Shanghai.
  - Replay outputs:
    - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e27_domain5_heldout_eye2_all_train_ready_best`
    - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e27_domain5_heldout_eye2_heldout_best`
    - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e27_domain5_heldout_eye2_train_domain_best`
  - Gate outputs append `_gate_runtime_scaled` to each replay output path.

E27 results:

| Slice | Episodes | Replay MAE | Replay RMSE | Startup Policy Effective | Startup Same Dir | Startup Extra/Wrong | Main Policy Effective | Main Same Dir | Main Extra/Wrong | Start40 Policy Effective | Tail Effective |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all train-ready | 24 | 0.0488 | 0.1025 | 50.6% | 54.0% | 14.3% | 96.8% | 96.2% | 1.5% | 0.0% | 0 / 504 |
| train-domain only | 19 | 0.0442 | 0.0908 | 47.1% | 51.1% | 8.8% | 97.1% | 97.2% | 0.4% | 0.0% | 0 / 403 |
| held-out domain 5 | 5 | 0.0663 | 0.1377 | 64.5% | 65.8% | 35.2% | 95.6% | 92.3% | 6.0% | 0.0% | 0 / 101 |

Interpretation:
- The held-out split is meaningful: held-out MAE/RMSE are much worse than train-domain, and held-out startup/main extra-or-wrong crossings are substantially higher.
- E27 is not a better policy candidate. All train-ready startup effective/same-dir falls well below E16 and E22b, even though start40 and tail are quiet.
- The failure mode matches the corrected deadzone-aware contract: a model cannot pass by being quiet in start/tail if it loses should-move startup coverage or produces held-out extra/wrong motion.
- Next direction: keep E22b as the current original-image action-control candidate, and treat visual-domain robustness as a separate representation/domain problem. If training another deadzone-aware model, use a window-conditioned should-move objective rather than a global suppression-heavy action loss.

### E14a Result

- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_transition_deadzone_eye2.yaml`
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_transition_deadzone_eye2`
- Logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/transition_deadzone_eye2_train.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/transition_deadzone_eye2_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/transition_deadzone_eye2_train.pid`
- Launch method: `setsid` wrapper, because the previous background launch was cleaned up by the command process group.
- Training completed at 2026-07-08 22:58 Asia/Shanghai.
- Best epoch: 1960.
- Best val loss: 0.10065057640895247.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/transition_deadzone_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/transition_deadzone_eye2_gate_runtime_scaled`.
- Replay metrics: MAE 0.0447, RMSE 0.0903, expert p95 0.7826, policy p95 0.8429, policy max 0.9690.
- Startup gate: policy effective 58.0%, same-axis/same-direction 61.0%, extra/wrong 15.4%.
- Main-motion gate: policy effective 95.1%, same-axis/same-direction 94.9%, extra/wrong 1.2%.
- Tail gate: 0 / 504 effective frames; mean tail p95 max abs 0.1087, max policy max abs 0.4403.
- Decision: reject and skip E14b. This candidate is quieter, but it does not satisfy the movement-first deadzone-aware criterion.

### E15a Low-Dimensional Intent Probe

- Artifact: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e15_intent_probe`.
- Labels: per-frame `should_move = expert action crosses any runtime-scaled directional deadzone`; `should_stop = not should_move`.
- Split: four episode-held-out folds over the 24 train-ready episodes.
- Probe type: linear binary classifier trained with balanced BCE, evaluated at thresholds 0.3/0.4/0.5/0.6/0.7.
- qpos-only at threshold 0.3: move recall 99.7%, startup move recall 100.0%, but stop false-positive 99.0% and startup stop false-positive 75.0%.
- qpos-only at threshold 0.5: stop false-positive improves to 36.4%, but move recall drops to 51.8% and startup move recall to 20.3%.
- qpos+qvel at threshold 0.5: move recall 45.6%, startup move recall 3.2%, stop false-positive 35.1%.
- qpos+qvel at threshold 0.6: stop false-positive 10.5%, but startup move recall 0.0%.
- Decision: reject low-dimensional standalone intent gate. It reproduces the same tradeoff as the action-output deadzone losses: either it predicts move almost everywhere, or it suppresses startup. E15b must include image evidence or shared ACT visual/state representations.

### E15b ACT Intent Head

Implementation boundary:
- `testbed/testbed/policies/act/detr/models/detr_vae.py`: owns the optional intent head structure only.
- `testbed/testbed/policies/act/adapter.py`: owns intent-label derivation from expert actions and the auxiliary BCE loss.
- `testbed/testbed/runtime/_train.py` and `testbed/testbed/actions/policy.py`: only pass `intent_loss` config so training and replay rebuild the same optional head.
- No dataset schema change and no runtime action gate in this slice.

Implemented semantics:
- Default off: old configs get `intent_dim=0`, no extra model parameters.
- Enabled config creates an 8-logit per-query head in axis-major/direction-minor order: `swing_pos`, `swing_neg`, `boom_pos`, `boom_neg`, `stick_pos`, `stick_neg`, `bucket_pos`, `bucket_neg`.
- Targets are derived online from the same runtime-scaled deadzone table as the gates.
- Loss is auxiliary only: `loss = action_l1 + kl * kl_weight + deadzone_loss + intent_loss`. E15b does not apply deadzone promotion/suppression to the action head directly.

Smoke verification:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e15_intent_head_eye2_smoke.yaml`.
- Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e15_intent_head_eye2_smoke`.
- Result: 1 epoch completed; val printed `intent_axis_dir_loss:0.7663`, `intent_loss:0.0383`; run metadata completed with best epoch 0.
- Replay smoke: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e15_intent_head_eye2_smoke_episode73_20steps` loaded the intent-head checkpoint successfully.
- Backward compatibility smoke: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/compat_eye2_baseline_after_intent_head_5steps` loaded the old E01 eye2 checkpoint with `intent_dim=0`.

Active training:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_intent_head_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_intent_head_eye2`.
- Reused split: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_eye2/train_val_split.yaml`.
- Logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/intent_head_eye2_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/intent_head_eye2_train.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/intent_head_eye2_train.pid`
- Launch method: `setsid` wrapper.
- Training completed at 2026-07-08 23:36 Asia/Shanghai.
- Best epoch: 1915.
- Best val loss: 0.11985711753368378.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/intent_head_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/intent_head_eye2_gate_runtime_scaled`.
- Intent-head eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/intent_head_eye2_head_eval`.
- Replay metrics: MAE 0.0420, RMSE 0.0892, expert p95 0.7826, policy p95 0.8537, policy max 0.9604.
- Startup gate: policy effective 39.4%, same-axis/same-direction 40.1%, extra/wrong 13.2%.
- Main-motion gate: policy effective 95.1%, same-axis/same-direction 94.8%, extra/wrong 1.3%.
- Tail gate: 0 / 504 effective frames; mean tail p95 max abs 0.1691, max policy max abs 0.4395.
- Intent-head eval: thresholds 0.3/0.4/0.5/0.6/0.7 all produce axis-dir recall 0.0%, any-move recall 0.0%, and startup any-move recall 0.0%.
- Decision: reject as a deployment candidate and as an unweighted intent-head route. It improves action closeness and keeps tail stable, but it worsens startup and the head learns the trivial no-move classifier.

### E16 Weighted ACT Intent Head

Reason:
- E15b's intent head saw 10,401 positive axis-direction labels out of 132,232 total labels, about 7.9% positive.
- The unweighted BCE head collapsed to all-negative predictions.
- E16 keeps the same optional head but sets `intent_loss.positive_weight: 8.0`, a conservative correction below the global negative/positive ratio of about 11.7.

Verification before launch:
- Added a focused unit test proving `intent_loss.positive_weight` increases the penalty for missed positive direction labels.
- `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_act_deadzone_loss.py testbed/tests/test_policy_action_source.py -q` passed: 22 tests.
- `py_compile` passed for `adapter.py`, `detr_vae.py`, `_train.py`, and `actions/policy.py`.
- Smoke config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e16_weighted_intent_head_eye2_smoke.yaml`.
- Smoke bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e16_weighted_intent_head_eye2_smoke`.
- Smoke result: 1 epoch completed; val printed `intent_axis_dir_loss:1.5783`, `intent_loss:0.0789`; run metadata completed with best epoch 0.

Training result:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_weighted_intent_head_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_weighted_intent_head_eye2`.
- Reused split: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_eye2/train_val_split.yaml`.
- Logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/weighted_intent_head_eye2_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/weighted_intent_head_eye2_train.master.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/weighted_intent_head_eye2_train.pid`
- Launch method: `setsid` wrapper.
- Training completed at 2026-07-09 00:09 Asia/Shanghai.
- Best epoch: 1805.
- Best val loss: 0.08149551041424274.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/weighted_intent_head_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/weighted_intent_head_eye2_gate_runtime_scaled`.
- Intent-head eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/weighted_intent_head_eye2_head_eval`.
- Replay metrics: MAE 0.0448, RMSE 0.1000, expert p95 0.7826, policy p95 0.8282, policy max 0.9137.
- Startup gate: policy effective 67.3%, same-axis/same-direction 80.4%, extra/wrong 6.9%.
- Main-motion gate: policy effective 95.5%, same-axis/same-direction 95.2%, extra/wrong 1.2%.
- Tail gate: 0 / 504 effective frames; mean tail p95 max abs 0.1153, max policy max abs 0.3540.
- Intent-head eval: positive labels are 10,401 / 132,232 axis-direction labels (7.87%). At threshold 0.7, axis-direction recall 91.9%, precision 64.2%, FPR 4.37%; any-move recall 94.8%, precision 67.6%, FPR 66.4%; startup any-move recall 88.0%, precision 82.6%, FPR 66.8%.
- Decision: keep as evidence, not as deployment candidate. E16 fixes the all-negative auxiliary-head collapse and improves startup same-direction quality over E01, but it still trails E01 on raw startup effective rate and the auxiliary any-move false-positive rate is too high for runtime gating.

Follow-up E16 vs E01 diagnosis:
- Artifact: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e16_vs_e01_diagnostics`.
- Per-episode comparison: `e16_minus_e01_by_episode.csv`; per-window axis table: `window_axis_by_episode.csv`.
- Plots:
  - `e16_vs_e01_startup_effective_delta.png`: per-episode startup effective delta, sorted by E16 minus E01.
  - `e16_start40_bucket_pos_by_episode.png`: E16 early bucket+ crossings by episode.
  - `e16_start40_bucket_vs_first_effective.png`: early bucket+ rate versus first expert-effective step.
  - `e16_top_risk_episodes.png`: top risk episodes combining early bucket+ and startup regression.
- E16 startup policy-effective is 4.6 percentage points lower than E01 on average, but same-direction coverage is 3.8 points higher and startup extra/wrong is 9.4 points lower. This confirms E16 improved direction quality, not raw startup coverage.
- E16 is worse than E01 on startup policy-effective in 12 / 24 episodes and better in 11 / 24 episodes.
- E16 start40 policy-effective is 17.2 points higher than E01 on average, almost entirely from bucket positive crossings. E16 mean start40 bucket+ rate is 33.4% vs E01 16.2%; 8 / 24 episodes have E16 start40 bucket+ above 50%.
- Worst startup regressions include episode_74 (-47.5 points), episode_83 (-32.5), episode_102 (-20.0), episode_100 (-17.5), episode_79 (-15.0), and episode_85 (-15.0).
- Largest early bucket+ cases include episode_86 (100.0%), episode_91 (100.0%), episode_85 (97.5%), episode_99 (92.5%), episode_100 (90.0%), episode_87 (90.0%), and episode_98 (90.0%).
- The `first_effective_step` scatter shows early bucket+ is not only a short pre-roll artifact: several episodes with first expert-effective step from roughly 50 to 130 still show high bucket+ in the first 40 steps.
- E17 implication: do not train a generic "move more" objective. The next loss must suppress pre-intent bucket+ in should-stop windows while preserving or increasing same-axis/same-direction crossings after the expert first becomes effective.

### E17 Balanced Deadzone Action Loss

Reason:
- E16 improved startup direction quality but not startup coverage, and introduced / preserved high early bucket+ in several episodes.
- Existing deadzone idle/wrong losses used crossing-only denominators. That means one wrong crossing and five wrong crossings with the same magnitude can produce the same loss, so frequency of early bucket+ is under-penalized.
- E17 keeps E16's weighted auxiliary intent head, but makes deadzone action loss frequency-sensitive for idle and wrong/extra crossings.
- E17 is still a should-move / should-stop matching experiment, not an output-shrinking experiment: same-direction promotion must preserve or improve above-deadzone action in expert-effective windows, while idle/wrong terms suppress crossings only where motion intent is absent or mismatched.

Implementation boundary:
- `testbed/testbed/policies/act/adapter.py` remains the owner because this is training-time ACT loss semantics.
- No model structure or dataset schema change.
- Old configs keep the legacy crossing-only behavior by default.

Implemented config keys:
```yaml
deadzone_loss:
  idle_denominator: all_idle_axes
  wrong_denominator: all_wrong_candidate_axes
```

Verification:
- Added focused tests proving frequent idle crossings and frequent wrong crossings are penalized more than sparse crossings under the new denominators.
- `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_act_deadzone_loss.py testbed/tests/test_policy_action_source.py -q` passed: 24 tests.
- `python -m py_compile testbed/testbed/policies/act/adapter.py testbed/testbed/runtime/_train.py testbed/testbed/actions/policy.py` passed.

Smoke:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e17_balanced_deadzone_eye2_smoke.yaml`.
- Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e17_balanced_deadzone_eye2_smoke`.
- Smoke training completed 1 epoch; val printed `deadzone_same_dir_loss`, `deadzone_idle_loss`, `deadzone_wrong_loss`, `deadzone_loss`, `intent_axis_dir_loss`, and `intent_loss`.
- Smoke replay loaded the bundle successfully for episode_73 first 20 steps: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e17_balanced_deadzone_eye2_smoke_episode73_20steps`.

Training result:
- Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_e17_balanced_deadzone_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_e17_balanced_deadzone_eye2`.
- Reused split: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_eye2/train_val_split.yaml`.
- Logs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e17_balanced_deadzone_eye2_train.log`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e17_balanced_deadzone_eye2_train.pid`
- Launch method: `setsid` wrapper.
- Completed at 2026-07-09 00:57 Asia/Shanghai.
- Best epoch: 1805.
- Best val loss: 0.0942519810050726.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e17_balanced_deadzone_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e17_balanced_deadzone_eye2_gate_runtime_scaled`.
- Replay metrics: MAE 0.0394, RMSE 0.0850, expert p95 0.7826, policy p95 0.8171, policy max 0.9104.
- Startup gate: policy effective 48.1%, same-axis/same-direction 52.6%, extra/wrong 9.3%.
- Main-motion gate: policy effective 94.4%, same-axis/same-direction 94.3%, extra/wrong 1.1%.
- Tail gate: 0 / 504 effective frames; mean tail p95 max abs 0.0855, max policy max abs 0.3387.
- Decision: reject as an action-policy candidate. E17 has the best replay MAE and quietest tail so far, but it fails the should-move side of deadzone awareness by suppressing startup crossings far below E01 and E16.

### E18a Recall-Priority Deadzone Action Loss

Reason:
- User clarified that deadzone-aware learning should match when the model should move and when it should not move. E17 still behaved like a suppression-heavy objective: MAE and tail improved, but startup should-move recall collapsed.
- E18a keeps the useful E16 weighted intent head and changes the action loss balance toward same-direction promotion while retaining light idle/wrong penalties.

Config:
- `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_e18a_recall_priority_deadzone_eye2.yaml`.
- Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_e18a_recall_priority_deadzone_eye2`.
- Reused split: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_eye2/train_val_split.yaml`.
- Deadzone weights: `same_dir_promote_weight: 0.12`, `idle_suppression_weight: 0.02`, `wrong_effective_weight: 0.03`, with `idle_denominator: all_idle_axes` and `wrong_denominator: all_wrong_candidate_axes`.
- Intent head: `intent_loss.enabled: true`, `weight: 0.05`, `positive_weight: 8.0`.

Training result:
- Wrapper: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e18a_recall_priority_deadzone_eye2_train_wrapper.sh`.
- Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e18a_recall_priority_deadzone_eye2_train.log`.
- PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e18a_recall_priority_deadzone_eye2_train.pid`.
- Launched at 2026-07-09 01:07 Asia/Shanghai with PID 586192.
- Completed at 2026-07-09 01:31 Asia/Shanghai.
- Best epoch: 1730.
- Best val loss: 0.1046154536306858.
- Replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e18a_recall_priority_deadzone_eye2_all_train_ready_best`.
- Gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e18a_recall_priority_deadzone_eye2_gate_runtime_scaled`.
- Replay metrics: MAE 0.0435, RMSE 0.0936, expert p95 0.7826, policy p95 0.8499, policy max 0.9590.
- Startup gate: policy effective 59.8%, same-axis/same-direction 64.9%, extra/wrong 12.4%.
- Main-motion gate: policy effective 92.7%, same-axis/same-direction 91.9%, extra/wrong 1.6%.
- Tail gate: 0 / 504 effective frames; mean tail p95 max abs 0.1879, max policy max abs 0.5970.
- Decision: reject as final. E18a confirms recall-priority action loss is less damaging than E17, but it still fails the should-move criterion relative to E16/E01 and weakens main-motion coverage. Current best action-intent candidate remains E16, with E01 as the raw startup baseline.

### E19 E16 Solo Bucket+ Suppression Gate

Reason:
- E16 has the best startup same-direction quality so far but high early bucket+ crossings in the first 40 steps.
- Before launching another training run, test whether a simple runtime-style action filter could remove this false motion while preserving startup/main coverage.

Post-hoc rule:
- Source replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/weighted_intent_head_eye2_all_train_ready_best`.
- Output replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e19_e16_solo_bucket_pos_suppressed`.
- Gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e19_e16_solo_bucket_pos_suppressed_gate_runtime_scaled`.
- Rule: if the E16 policy has only bucket-positive crossing an effective directional deadzone and no other effective axis/direction, set the bucket action to 0 for that frame.
- Suppressed frames: 2,013 total solo bucket+ frames across all 24 episodes.

Result:
- Start40 gate: policy effective drops from 33.4% to 0.0%; bucket+ early crossings are fully removed.
- Startup gate: policy effective drops from 67.3% to 31.5%, same-direction from 80.4% to 32.9%, extra/wrong from 6.9% to 3.0%.
- Main-motion gate: policy effective stays close, 95.5% to 95.1%, same-direction 95.2% to 94.8%.
- Tail remains 0 / 504 effective frames.
- Decision: reject this rule. It removes the visible early bucket+ problem, but it also suppresses real startup bucket motion. The next gate must use visual-state phase/intent, not a bucket-axis-only rule.

### E20 E16 Auxiliary Intent-Head Gate

Reason:
- E16 already has an auxiliary 8-direction intent head trained from visual/state ACT features.
- Before training a separate phase gate, test whether this existing head can gate E16 actions post-hoc.

Probe:
- Intent-prob cache and threshold scan: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e20_e16_intent_head_gate_scan`.
- Source action replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/weighted_intent_head_eye2_all_train_ready_best`.
- Cached probabilities: query-0 sigmoid of E16's 8-direction intent logits for every train-ready frame.
- Scanned gates:
  - `any_move`: zero the whole E16 action when max intent probability is below threshold.
  - `direction`: zero only action axes whose current sign's intent probability is below threshold.
- Thresholds: 0.2 through 0.9.

Scan result:
- Low thresholds 0.2-0.6 preserve startup but leave start40 bucket+ effectively unchanged at 33.4%.
- Higher thresholds reduce startup/main together with false motion. At 0.8, startup drops to 48.8%; at 0.9, startup drops to 30.3%.
- Representative materialized eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e20_e16_intent_head_gate_scan/direction_thr_0.7`.
- Gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e20_e16_intent_head_direction_thr07_gate_runtime_scaled`.
- Direction threshold 0.7:
  - Startup: policy effective 60.4%, same-dir 73.3%, extra/wrong 4.7%.
  - Start40: policy effective 33.0%, extra/wrong 30.5%, bucket+ 33.0%, essentially unchanged from E16.
  - Main-motion: policy effective 92.1%, same-dir 91.5%, extra/wrong 1.5%.
  - Tail: 0 / 504 effective frames; mean tail p95 max abs 0.0112, max 0.2719.
- Decision: reject as a runtime action gate. E20 improves tail magnitude and slightly reduces startup extra/wrong, but it fails the actual problem: early bucket+ remains while startup/main coverage degrade.

### E21a E16 Intent-Probability + State Phase Gate Probe

Reason:
- User clarified that deadzone-aware behavior should match both sides of intent: should-move frames need enough same-direction output to cross the physical deadzone, and should-stop frames need outputs inside the deadzone.
- E17/E18a showed that action-loss reweighting alone can become suppression-heavy and hurt startup. E21a tests a separate should-move phase gate before launching another training run.

Probe:
- Artifact root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e21a_e16_intentprob_qpos_phase_gate_probe`.
- Source policy replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/weighted_intent_head_eye2_all_train_ready_best`.
- Features: cached E16 query-0 8-direction intent probabilities from E20 plus qpos and qvel.
- Label: per-frame `should_move = expert action crosses any runtime-scaled directional deadzone`.
- Split: four deterministic episode-held-out folds over the 24 train-ready episodes.
- Model: small PyTorch MLP, trained with balanced BCE. This is a diagnostic probe, not a deployable runtime module.

Fold result at threshold 0.5:
- Fold recall range: 88.0% to 90.6%.
- Fold precision range: 92.8% to 94.6%.
- Fold stop false-positive range: 7.3% to 10.7%.

Best operating point from gate scan:
- Threshold 0.2 is the most useful tradeoff in this probe.
- Materialized replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e21a_e16_intentprob_qpos_phase_gate_probe/phase_thr_0.2`.
- Gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e21a_e16_intentprob_qpos_phase_thr02_gate_runtime_scaled`.
- Startup: policy effective 64.0%, same-dir 77.2%, extra/wrong 5.7%, with 21 / 24 episodes at or above 50% startup effective.
- Start40: policy effective 1.6%, same-dir 8.2%, extra/wrong 0.0%, bucket+ 1.6%.
- Main-motion: policy effective 94.2%, same-dir 93.9%, extra/wrong 1.2%.
- Tail: 0 / 504 effective frames; mean tail p95 max abs 0.0091, max policy max abs 0.2901.
- Action closeness regresses versus E16: MAE 0.0492 vs 0.0448 and RMSE 0.1241 vs 0.1000, because the gate zeros some expert-moving frames.

Decision:
- Keep E21a as the best evidence for the next direction, not as a final candidate.
- It is the first gate that materially attacks early bucket+ without the catastrophic startup collapse of E19 or the unchanged start40 behavior of E20.
- Next step E22 should train or calibrate a deployable visual-state should-move gate and evaluate it with the same episode-held-out protocol plus replay/gate metrics. The acceptance criterion is not lower MAE alone; it must preserve startup/main same-direction crossing while suppressing pre-intent and tail effective motion.

### E22a Reproducible Phase-Gate Artifact and Hysteresis Scan

Reason:
- E21a showed that should-move phase is learnable, but it was only an inline probe.
- E22a makes the probe reproducible and closer to deployment: it saves a final tiny MLP model artifact with feature metadata, writes out-of-fold probabilities for evaluation, scans simple and hysteresis gates, and materializes the selected gate as a normal offline replay directory.

Implementation:
- Pure helper module: `testbed/testbed/policies/phase_gate.py`.
- Focused test: `testbed/tests/test_phase_gate.py`.
- Experiment script: `scripts/e22_phase_gate_probe.py`.
- Feature vector: E16 query-0 8-way intent probabilities from E20 plus qpos and qvel.
- Label: `should_move = expert action crosses any runtime-scaled directional deadzone`.
- Split: four deterministic episode-held-out folds over the 24 train-ready episodes.
- Model: 16 -> 32 -> 1 PyTorch MLP with balanced BCE.

Artifacts:
- Root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22a_phase_gate_hysteresis_probe`.
- Out-of-fold probabilities: `phase_probs/*.npz`.
- Fold metrics: `fold_summary.json`.
- Gate scan: `threshold_scan.csv`.
- Final all-data model artifact: `phase_gate_model.pt` and `phase_gate_model_metadata.json`.
- Materialized selected replay: `simple_0.15`.
- Standard gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22a_phase_gate_hysteresis_probe_gate_runtime_scaled`.

Fold result at threshold 0.5:
- Recall range: 88.1% to 90.6%.
- Precision range: 92.4% to 94.9%.
- Stop false-positive range: 6.9% to 10.5%.

Scan result:
- Best selected gate: `simple_0.15`.
- Hysteresis did not beat the low simple threshold under the current score. The closest hysteresis candidate was `hyst_o0.25_c0.10`, with similar start40 but slightly worse MAE and startup.
- E22a `simple_0.15` replay metrics: MAE 0.0458, RMSE 0.1137, policy p95 0.8281, policy max 0.9137.
- For comparison, E16 replay metrics: MAE 0.0448, RMSE 0.1000, policy p95 0.8282, policy max 0.9137.

Standard deadzone gate:
- Start40: E16 policy effective 33.4%, extra/wrong 30.9%, bucket+ 33.4%; E22a policy effective 1.7%, extra/wrong 0.0%, bucket+ 1.7%.
- Startup first-effective 40: E16 policy effective 67.3%, same-dir 80.4%, extra/wrong 6.9%; E22a policy effective 65.1%, same-dir 78.6%, extra/wrong 5.7%.
- Main-motion: E16 policy effective 95.5%, same-dir 95.2%, extra/wrong 1.2%; E22a policy effective 94.8%, same-dir 94.5%, extra/wrong 1.2%.
- Tail: both 0 / 504 effective frames. E22a reduces mean tail p95 max abs from 0.1153 to 0.0091 and max policy max abs from 0.3540 to 0.2901.

Decision:
- Keep E22a as the current best offline gate tradeoff.
- Do not call it final deployment yet: it is still a post-hoc hard zero gate, and RMSE worsens versus E16.
- Next step E22b should preserve the phase separability gain while reducing hard-gate action-closeness loss. Candidate directions are soft attenuation, per-axis gating using phase plus direction confidence, or using the phase gate as a training/calibration loss instead of only zeroing actions after replay.

### E22b Soft Phase-Gate Scale Scan

Reason:
- E22a proved the phase decision is useful, but hard zeroing inactive frames worsened action closeness.
- E22b tests whether inactive-phase actions can be attenuated instead of zeroed. If the attenuated residual remains below directional deadzones, it can improve MAE/RMSE without changing real-machine effective intent.

Implementation:
- Extended `testbed/testbed/policies/phase_gate.py` with `inactive_scale`.
- Extended `scripts/e22_phase_gate_probe.py` with `--inactive-scales`.
- Added a focused test that inactive frames can retain a configurable action fraction while active frames stay unchanged.

Artifacts:
- Root: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22b_phase_gate_soft_scale_probe`.
- Selected replay: `simple_0.15_s0.50`.
- Standard gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e22b_phase_gate_soft_scale_probe_gate_runtime_scaled`.

Result:
- Selected gate: simple threshold 0.15, inactive action scale 0.50.
- Replay metrics:
  - E16: MAE 0.0448, RMSE 0.1000, policy p95 0.8282, policy max 0.9137.
  - E22a hard zero: MAE 0.0458, RMSE 0.1137, policy p95 0.8281, policy max 0.9137.
  - E22b soft scale: MAE 0.0414, RMSE 0.0861, policy p95 0.8281, policy max 0.9137.
- Start40 gate:
  - E16: policy effective 33.4%, extra/wrong 30.9%, bucket+ 33.4%.
  - E22a: policy effective 1.7%, extra/wrong 0.0%, bucket+ 1.7%.
  - E22b: policy effective 1.7%, extra/wrong 0.0%, bucket+ 1.7%.
- Startup first-effective 40:
  - E16: policy effective 67.3%, same-dir 80.4%, extra/wrong 6.9%.
  - E22a: policy effective 65.1%, same-dir 78.6%, extra/wrong 5.7%.
  - E22b: policy effective 65.1%, same-dir 78.6%, extra/wrong 5.7%.
- Main-motion:
  - E16: policy effective 95.5%, same-dir 95.2%, extra/wrong 1.2%.
  - E22a/E22b: policy effective 94.8%, same-dir 94.5%, extra/wrong 1.2%.
- Tail:
  - All three have 0 / 504 effective frames.
  - E22b tail p95 max abs 0.0622, between E22a 0.0091 and E16 0.1153, still below runtime deadzones.

Decision:
- E22b is the current best offline tradeoff across the three user metrics: action closeness, action intent/deadzone crossing, and stop stability.
- It should still not be promoted straight to live control. Next evidence should test whether the phase gate holds under visual-domain or texture-domain shifts and whether the gate can be packaged with real-time inference latency within budget.

### E23a/E23b E16 Low-Pass Visual Stress

Reason:
- E22b is the current best original-image offline candidate, but it depends on E16's image-conditioned action and intent features.
- Before packaging E22b, check whether the base E16 policy keeps movement intent under visual texture/detail shifts.
- This test uses existing `offline_policy_eval` deterministic image transforms only. It does not retrain and it does not yet apply E22b phase gating under transformed images, because that requires transformed-image intent-prob caching.

Artifacts:
- E23a replay, downsample_060: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e23a_e16_weighted_intent_eye2_infer_downsample060_all_train_ready_best`.
- E23a gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e23a_e16_downsample060_gate_runtime_scaled`.
- E23b replay, downsample_080: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e23b_e16_weighted_intent_eye2_infer_downsample080_all_train_ready_best`.
- E23b gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e23b_e16_downsample080_gate_runtime_scaled`.

Replay metrics:
- E16 original: MAE 0.0448, RMSE 0.1000, policy p95 0.8282, policy max 0.9137.
- E23a downsample_060: MAE 0.0543, RMSE 0.1166, policy p95 0.7638, policy max 0.8979.
- E23b downsample_080: MAE 0.0466, RMSE 0.1043, policy p95 0.8005, policy max 0.9047.

Deadzone gates:
- Start40:
  - Original: policy effective 33.4%, extra/wrong 30.9%, bucket+ 33.4%.
  - Downsample_060: policy effective 0.0%, extra/wrong 0.0%, bucket+ 0.0%.
  - Downsample_080: policy effective 4.4%, extra/wrong 4.1%, bucket+ 4.4%.
- Startup first-effective 40:
  - Original: policy effective 67.3%, same-dir 80.4%, extra/wrong 6.9%.
  - Downsample_060: policy effective 18.6%, same-dir 20.5%, extra/wrong 0.4%.
  - Downsample_080: policy effective 48.6%, same-dir 54.1%, extra/wrong 6.8%.
- Main-motion:
  - Original: policy effective 95.5%, same-dir 95.2%, extra/wrong 1.2%.
  - Downsample_060: policy effective 72.9%, same-dir 72.3%, extra/wrong 1.0%.
  - Downsample_080: policy effective 89.5%, same-dir 89.5%, extra/wrong 0.9%.
- Tail:
  - All variants keep 0 / 504 effective frames.
  - Tail p95 max abs rises under low-pass: original 0.1153, downsample_060 0.2080, downsample_080 0.1613, but still below deadzone-effective thresholds.

Decision:
- Treat E23 as a generalization warning, not as a deployment improvement.
- Infer-only low-pass removes early false motion partly by suppressing action amplitude, but it also suppresses true startup/main motion. This is the same failure mode as earlier downsample training/inference tests.
- Do not assume E22b solves texture sensitivity. E22b improves original-image phase gating, but transformed-image robustness remains unproven.
- Next experiment E24 should either:
  - cache E16 action plus intent probabilities under transformed images and apply the same E22b soft phase gate, or
  - train an augmentation/domain-robust candidate and evaluate it with the same start40/startup/main/tail gates.

### E24a/E24b Phase Gate Under Transformed Visual Inputs

Reason:
- E23 showed that infer-only low-pass damages E16 movement intent. E24 checks whether the phase gate can help once its own intent-probability features are also computed from transformed images.
- This is still an offline diagnostic. The E24 phase gate is trained and validated on transformed features to test separability under visual stress; it is not yet proof that the original E22b phase-gate artifact deploys across domains.

Implementation:
- Added `scripts/cache_act_intent_probs.py` to cache ACT auxiliary intent probabilities with the same offline image transforms used by `offline_policy_eval`.
- E24a caches `downsample_080` intent probabilities and applies E22b-style `simple_0.15_s0.50` to the E23b transformed action replay.
- E24b caches `downsample_060` intent probabilities and applies E22b-style `simple_0.15_s0.50` to the E23a transformed action replay.

Artifacts:
- E24a intent cache: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e24a_e16_downsample080_intent_probs`.
- E24a phase replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e24a_e22b_phase_gate_on_downsample080_probe/simple_0.15_s0.50`.
- E24a gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e24a_e22b_phase_gate_on_downsample080_gate_runtime_scaled`.
- E24b intent cache: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e24b_e16_downsample060_intent_probs`.
- E24b phase replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e24b_e22b_phase_gate_on_downsample060_probe/simple_0.15_s0.50`.
- E24b gate: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e24b_e22b_phase_gate_on_downsample060_gate_runtime_scaled`.

Replay metrics:
- E16 original: MAE 0.0448, RMSE 0.1000, policy p95 0.8282.
- E22b original phase gate: MAE 0.0414, RMSE 0.0861, policy p95 0.8281.
- E16 downsample_080: MAE 0.0466, RMSE 0.1043, policy p95 0.8005.
- E24a downsample_080 phase gate: MAE 0.0439, RMSE 0.0941, policy p95 0.7959.
- E16 downsample_060: MAE 0.0543, RMSE 0.1166, policy p95 0.7638.
- E24b downsample_060 phase gate: MAE 0.0516, RMSE 0.1077, policy p95 0.7412.

Deadzone gates:
- Downsample_080:
  - Base E16 transformed startup: policy effective 48.6%, same-dir 54.1%, extra/wrong 6.8%.
  - E24a phase gate startup: policy effective 47.8%, same-dir 53.5%, extra/wrong 6.2%.
  - Base E16 transformed main: policy effective 89.5%, same-dir 89.5%.
  - E24a phase gate main: policy effective 88.9%, same-dir 88.8%.
  - Start40 improves from 4.4% policy effective / 4.1% extra-wrong to 0.3% / 0.0%.
  - Tail remains 0 / 504; tail p95 max abs drops from 0.1613 to 0.0807.
- Downsample_060:
  - Base E16 transformed startup: policy effective 18.6%, same-dir 20.5%, extra/wrong 0.4%.
  - E24b phase gate startup: unchanged at policy effective 18.6%, same-dir 20.5%, extra/wrong 0.4%.
  - Base E16 transformed main: policy effective 72.9%, same-dir 72.3%.
  - E24b phase gate main: policy effective 72.3%, same-dir 71.7%.
  - Start40 stays 0.0%.
  - Tail remains 0 / 504; tail p95 max abs drops from 0.2080 to 0.1095.

Decision:
- E24 confirms the gate is useful for stop/noise cleanup but cannot solve visual-domain movement suppression.
- Do not spend the next slice on more phase-gate threshold tuning unless a new base policy recovers transformed-image should-move coverage.
- Next useful experiment is E25: train or evaluate a visually robust candidate, likely with stochastic image augmentation or visual-domain splits, then apply the same original/start40/startup/main/tail gate matrix.

## Implementation Tasks

### Task 1: Motion-Intent-Aware Deadzone Loss

**Files:**
- Modify: `testbed/testbed/policies/act/adapter.py`
- Modify: `testbed/testbed/runtime/_train.py` only if config keys need pass-through
- Test: `testbed/tests/test_policy_action_source.py` or a new focused ACT loss test if existing test ownership is insufficient

- [ ] **Step 1: Write a failing loss-mask test**

Create a test that constructs synthetic expert/policy action tensors and verifies:
- same-direction promotion applies where expert crosses a directional deadzone.
- same-direction promotion pushes policy over the same directional deadzone plus margin, not merely toward zero MAE.
- idle suppression applies where expert is below all directional deadzones and policy crosses any directional deadzone.
- wrong-direction suppression applies where policy crosses a deadzone in an axis/direction the expert does not request.
- useful startup/main-motion expert-effective frames are never treated as idle suppression frames.

Run:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
python -m pytest testbed/tests -q -k deadzone_loss
```

Expected: FAIL before implementation or before config update.

- [ ] **Step 2: Implement motion-intent-aware deadzone terms**

Use config keys:

```yaml
deadzone_loss:
  enabled: true
  same_dir_promote_weight: 0.10
  idle_suppression_weight: 0.10
  wrong_effective_weight: 0.05
  margin: 0.02
  effective_target: threshold_plus_margin
  apply_idle_suppression_when: expert_ineffective
  threshold_json: /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/deadzone_policy_raw_for_runtime_scale.json
```

Preserve existing disabled behavior when `enabled: false`.

Loss semantics:

```text
expert effective on axis/direction:
  penalize policy if it is below that same directional threshold + margin

expert ineffective on all directions:
  penalize policy if it crosses any directional threshold

policy effective in a direction not requested by expert:
  penalize as wrong/extra motion
```

- [ ] **Step 3: Run focused tests**

```bash
python -m py_compile testbed/testbed/policies/act/adapter.py testbed/testbed/runtime/_train.py
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
python -m pytest testbed/tests -q -k "deadzone_loss or policy_action_source"
```

Expected: PASS.

### Task 2: Train Motion-Intent-Aware Models

**Files / Artifacts:**
- Create configs under `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/`
- Output ckpts under `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/`

- [ ] **Step 1: Create four-view scoped-loss config**

Copy the current four-view config and change only task name, checkpoint dir, and `deadzone_loss` motion-intent keys.

- [ ] **Step 2: Create eye2 scoped-loss config**

Copy the current eye2 config and apply the same motion-intent loss.

- [ ] **Step 3: Launch sequential training**

Use one background shell, four-view first, eye2 second, to avoid GPU contention.

```bash
ROOT=/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104
export PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed
export MPLCONFIGDIR=/tmp/excavator_mpl
python -m testbed.cli.train --config "$ROOT/configs/<four_scoped>.yaml"
python -m testbed.cli.train --config "$ROOT/configs/<eye2_scoped>.yaml"
```

- [ ] **Step 4: Record training result**

Append best epoch/loss and checkpoint paths to this ledger.

### Task 3: Offline Replay and Gate

**Files / Artifacts:**
- Output under `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/`

- [ ] **Step 1: Replay all train-ready episodes**

Run for each new bundle:

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
MPLCONFIGDIR=/tmp/excavator_mpl \
python -m testbed.cli.offline_policy_eval \
  --bundle-dir <BUNDLE_DIR> \
  --ckpt <BUNDLE_DIR>/policy_best.ckpt \
  --dataset-dir /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz \
  --manifest /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz/qc_batch_ref_72_87/train_ready_manifest.json \
  --all-train-ready \
  --output-dir <EVAL_OUTPUT_DIR> \
  --device cuda \
  --progress-every 0
```

- [ ] **Step 2: Run deadzone window gate**

```bash
PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed \
python scripts/deadzone_window_eval.py \
  --deadzone-json /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/deadzone_policy_raw_for_runtime_scale.json \
  --manifest /data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz/qc_batch_ref_72_87/train_ready_manifest.json \
  --output-dir <GATE_OUTPUT_DIR> \
  --eval "model=<EVAL_OUTPUT_DIR>"
```

- [ ] **Step 3: Run startup first-effective gate**

Use the same first expert-effective + 40 step calculation used for E00-E03 and write:
- `startup_first_expert_effective_40_summary.csv`
- `startup_first_expert_effective_40_aggregate.csv`

- [ ] **Step 4: Run tail stability gate**

Use raw `t_stop/t_go` from `/data/.../real_teleop_v1_episodes_72_104` and policy replay actions from the 20Hz eval output. Write:
- `tail_stability_summary.csv`
- `tail_stability_by_episode.csv`

- [ ] **Step 5: Update this ledger**

Fill the Completed Experiment Ledger with the new model rows and write a one-line decision.

### Task 4: Gohome Eligibility Data Contract

**Files:**
- Create: `testbed/testbed/data/handoff_labels.py`
- Modify: `testbed/testbed/data/resample_20hz.py`
- Test: `testbed/tests/test_realworld_v1.py`

- [ ] **Step 1: Add pure label tests**

Verify:
- `t_go` is the first of request / accepted / running.
- `t_stop` is the frame after the last action with `max(abs(action)) > idle_action_threshold`.
- `eligible_start = t_stop + dwell_min_steps`.
- `gohome_eligible_label` never starts before `eligible_start`.
- automation after `t_go` is ignored for gohome loss and action loss.

- [ ] **Step 2: Implement label computation**

Create a pure function that returns:

```python
gohome_eligible_label
gohome_loss_mask
tail_idle_mask
action_loss_mask
owner_automation
t_stop
t_go
eligible_start
```

- [ ] **Step 3: Write labels into handoff HDF5**

Extend `build_handoff_20hz_episode` without removing existing audit labels.

- [ ] **Step 4: Generate dwell sweep census**

Build census for `dwell_min_steps` values `5,10,15,20` and record positive fraction, earliest label relative to stop, and skipped episodes.

### Task 5: Independent Gohome Eligibility Classifier

**Files:**
- Create a new focused training/eval module only after Task 4 defines stable labels.

- [ ] **Step 1: State-only baseline**

Train with `qpos/qvel/action-history` and no images.

- [ ] **Step 2: Eye2 + state**

Train with `video4/video5 + qpos/qvel/action-history`.

- [ ] **Step 3: Four-view + state**

Train with `video4/video5/video6/video7 + qpos/qvel/action-history`.

- [ ] **Step 4: Event-level gate**

For each episode report:
- first eligible trigger step
- trigger delay relative to `t_go`
- early false positives before `eligible_start`
- missed trigger
- false trigger count per episode

## Result Update Rules

After every completed experiment:

1. Add one row to the Completed Experiment Ledger.
2. Link the checkpoint and eval output paths.
3. Record which gate improved and which regressed.
4. If no model improves all three primary metrics, add a short failure analysis under Active Hypotheses.
5. Decide the next experiment from evidence, not from val loss alone.

## Current Next Slice

Update 2026-07-08 18:30 Asia/Shanghai:
- Implemented motion-intent-aware deadzone terms in `ACTAdapter`.
- Added focused tests in `testbed/tests/test_act_deadzone_loss.py`.
- Smoke trained four-view and eye2 configs for 1 epoch; both emitted `deadzone_same_dir_loss`, `deadzone_idle_loss`, and `deadzone_wrong_loss`.
- Started formal sequential E04 training with master PID recorded at `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/motion_intent_deadzone_four_then_eye2_train.pid`.

Update 2026-07-08 18:39 Asia/Shanghai:
- Added pure gohome eligibility label owner `testbed/testbed/data/handoff_labels.py`.
- Extended `build_handoff_20hz_episode` to write `handoff/gohome_eligible_label`, `handoff/gohome_loss_mask`, `handoff/tail_idle_mask`, and eligibility metadata.
- Tests passed:
  - `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_handoff_labels.py testbed/tests/test_realworld_v1.py -q -k 'handoff_labels or handoff_20hz_builder'`
- Dwell sweep census written to `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_eligibility_census`.
- Dwell sweep result:
  - dwell 5 raw steps: 438 eligible 20Hz frames, 24/24 episodes positive, 2.65%.
  - dwell 10 raw steps: 373 eligible 20Hz frames, 24/24 episodes positive, 2.26%.
  - dwell 15 raw steps: 313 eligible 20Hz frames, 23/24 episodes positive, 1.89%.
  - dwell 20 raw steps: 254 eligible 20Hz frames, 23/24 episodes positive, 1.54%.
- Initial recommendation for gohome eligibility training: use dwell 10 raw steps as the first conservative-but-learnable label setting.
- Generated actual dwell10 handoff HDF5 dataset at `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/handoff_eligibility_20hz_dwell10`.
  - 24 episodes.
  - 24,747 total output steps including automation tail.
  - 388 eligible frames.
  - 519 tail idle frames.
  - 16,544 gohome loss-mask frames.
  - 16,025 action loss-mask frames.

Update 2026-07-08 19:02 Asia/Shanghai:
- E04 four-view motion-intent deadzone training was still in progress at this intermediate check; later metadata supersedes the intermediate best value below.
- Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_motion_intent_deadzone_four_camera`.
- Intermediate best epoch: 1255.
- Intermediate best val loss: 0.09551059827208519.
- E04 eye2 training started automatically from the same master process after the four-view run.

Update 2026-07-08 19:36 Asia/Shanghai:
- User clarified the target semantics: deadzone-aware policy learning should match when to move and when not to move. The implemented E04 loss matches this by separating:
  - `deadzone_same_dir_loss`: promote same-axis same-direction crossing when the expert is effective.
  - `deadzone_idle_loss`: suppress effective crossing when the expert is ineffective.
  - `deadzone_wrong_loss`: suppress wrong-axis or wrong-direction effective crossing when the expert is moving.
- E04 four-view final training metadata:
  - Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_motion_intent_deadzone_four_camera`.
  - Best epoch: 1960.
  - Best val loss: 0.11135327070951462.
- E04 eye2 final training metadata:
  - Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_motion_intent_deadzone_eye2`.
  - Best epoch: 1995.
  - Best val loss: 0.07216260582208633.
- Started sequential all-train-ready replay for both E04 checkpoints with master PID recorded at `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/motion_intent_eval_four_then_eye2.pid`.
- Replay outputs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/motion_intent_four_all_train_ready_best`
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/motion_intent_eye2_all_train_ready_best`
- Added reusable gate script `scripts/deadzone_startup_tail_eval.py` for first expert-effective startup and `t_stop -> t_go` tail stability metrics.
- Six-model gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/deadzone_gate_six_models_motion_intent_runtime_scaled`.
- E04 replay and gate results:
  - four-view motion-intent: MAE 0.04136, startup effective 60.6%, startup same-dir 63.9%, startup extra/wrong 16.2%, main effective 95.7%, tail effective 0.0%.
  - eye2 motion-intent: MAE 0.04848, startup effective 54.1%, startup same-dir 55.5%, startup extra/wrong 16.9%, main effective 96.5%, tail effective 0.0%.
- Decision: do not select E04 as the deployment candidate. It improves four-view MAE and improves eye2 startup relative to broad deadzone, but it still underperforms the no-treatment baselines on the startup deadzone-crossing gate.
- Next action-policy experiment E10:
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_promote_biased_deadzone_four_camera.yaml`.
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_no_transform_promote_biased_deadzone_eye2.yaml`.
  - Hypothesis: E04 still over-regularized startup. Increasing same-direction promotion while reducing idle/wrong suppression should recover startup effective motion without breaking tail stability.
- Started E10 sequential training:
  - Master PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/promote_biased_deadzone_four_then_eye2_train.pid`.
  - Master log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/promote_biased_deadzone_four_then_eye2_train.master.log`.
  - Four-view log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/promote_biased_deadzone_four_camera_train.log`.
  - Eye2 log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/promote_biased_deadzone_eye2_train.log`.
  - Initial health check: four-view run reached about epoch 150/2000 and emitted `deadzone_same_dir_loss`, `deadzone_idle_loss`, and `deadzone_wrong_loss`.

Update 2026-07-08 20:55 Asia/Shanghai:
- E10 training completed:
  - Four-view bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_promote_biased_deadzone_four_camera`.
  - Four-view best epoch: 1915; best val loss: 0.12156824627891183.
  - Eye2 bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_no_transform_promote_biased_deadzone_eye2`.
  - Eye2 best epoch: 1415; best val loss: 0.09827480278909206.
- E10 replay outputs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/promote_biased_four_all_train_ready_best`.
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/promote_biased_eye2_all_train_ready_best`.
- Eight-model gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/deadzone_gate_eight_models_promote_biased_runtime_scaled`.
- E10 results:
  - Four-view promote-biased: MAE 0.05339, policy p95 0.9271, startup effective 65.0%, startup same-dir 74.9%, startup extra/wrong 9.7%, main effective 97.6%, tail effective 0.0%.
  - Eye2 promote-biased: MAE 0.05001, policy p95 0.8366, startup effective 59.4%, startup same-dir 59.0%, startup extra/wrong 20.4%, main effective 97.2%, tail effective 0.0%.
- Decision: reject E10 as a deployment candidate. It confirms that stronger global promotion can recover some startup intent, but the MAE/action distribution regression is too large and startup still does not beat the no-deadzone baselines.
- Current Pareto candidates:
  - `eye2_baseline`: strongest startup/main intent balance; MAE worse than four-view but still reasonable.
  - `four_baseline`: good startup and better MAE than eye2 baseline.
  - `four_motion_intent`: best MAE so far, but startup deficit makes it unsafe as the current primary candidate.

Update 2026-07-08 21:12 Asia/Shanghai:
- E08 infer-only visual downsample gate completed for baseline candidates.
- Replay outputs:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/visual_downsample060_four_baseline_all_train_ready_best`.
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/visual_downsample060_eye2_baseline_all_train_ready_best`.
- Gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/visual_downsample060_baseline_gate_runtime_scaled`.
- Results:
  - Four-view baseline with infer-only `downsample_060`: MAE 0.08690, startup effective 64.9%, main effective 47.7%, tail effective 0.0%.
  - Eye2 baseline with infer-only `downsample_060`: MAE 0.06614, startup effective 65.3%, main effective 83.3%, tail effective 0.6%.
- Decision: reject infer-only downsample as a runtime patch. It suppresses action amplitude and damages main-motion intent. Eye2 is more robust than four-view, but still degrades enough that texture suppression must be trained consistently or handled by domain augmentation, not added only at inference.
- Next targeted experiment E12: train eye2 baseline with `train.image_transform: downsample_060`, no deadzone loss, then evaluate with the same transform and compare against eye2 baseline and infer-only downsample.
- E12 config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_downsample060_eye2.yaml`.
- E12 training completed:
  - Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_downsample060_eye2`.
  - Best epoch: 1520.
  - Best val loss: 0.10079501196742058.
- E12 replay outputs:
  - Consistent downsample inference: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/downsample060_eye2_train_downsample060_infer_all_train_ready_best`.
  - Original-image inference: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/downsample060_eye2_train_none_infer_all_train_ready_best`.
- E12 gate output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/downsample060_eye2_train_gate_runtime_scaled`.
- E12 results:
  - Eye2 baseline: MAE 0.04728, startup 71.9%, main 96.6%, tail 0.0%.
  - Eye2 baseline infer-only downsample060: MAE 0.06614, startup 65.3%, main 83.3%, tail 0.6%.
  - Eye2 train+infer downsample060: MAE 0.05228, startup 63.0%, startup same-dir 71.3%, startup extra/wrong 11.1%, main 96.8%, tail 0.0%.
  - Eye2 train downsample060 + original-image infer: MAE 0.05457, startup 34.1%, main 96.7%, tail 0.0%.
- Decision: reject E12 as a deployment candidate. Training-consistent downsample fixes main/tail relative to infer-only downsample, but startup and MAE remain below baseline. Original-image inference for a downsample-trained model is not viable. Next experiment should reduce the transform strength rather than increase deadzone regularization.
- E13 config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_downsample080_eye2.yaml`.
- Verification before E13 training: `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_dataset_image_transform.py testbed/tests/test_offline_policy_eval.py -q -k 'image_transform'` passed.

Update 2026-07-09 01:55 Asia/Shanghai:
- Latest action-policy evidence rejects continuing blind deadzone action-loss reweighting as the default next move.
- E16 remains the best current action-intent source: good startup same-direction quality and low extra/wrong, but early start40 bucket+ is too high.
- E17 and E18a confirm the risk in suppression-heavy deadzone-aware losses: they improve MAE or quietness while damaging should-move startup coverage.
- E20 rejects reuse of the existing E16 auxiliary intent head as a runtime gate because early bucket+ remains until startup/main coverage is suppressed.
- E21a is the current best direction: a learned should-move phase gate using intent probabilities plus qpos/qvel reduces start40 bucket+ to 1.6% and keeps useful startup/main coverage, but it is only an offline probe and worsens MAE after hard zeroing.

Current next slice:
- Implement E22 as a deployable should-move / should-stop phase-gate experiment, not as another global action-loss weight sweep.
- Acceptance gates for E22:
  - should-move: startup/main same-direction crossings should stay near E16/E01, not collapse like E17/E19.
  - should-stop: start40 pre-intent and tail crossings should stay near E21a threshold 0.2 behavior.
  - action closeness: MAE/RMSE regression from hard zeroing must be measured and reduced, for example by using the gate as a loss mask/calibration target rather than only post-hoc zeroing.
- Do not promote E21a to runtime directly. It is evidence that the problem is learnable with phase/intent context, not a final deployment path.

Update 2026-07-09 02:03 Asia/Shanghai:
- E22a completed and supersedes E21a as the current phase-gate evidence.
- Selected gate: `simple_0.15` from a reproducible 4-fold MLP phase-gate artifact.
- Compared with E16, E22a reduces early start40 false motion from 33.4% to 1.7% while keeping startup at 65.1% and main at 94.8%.
- Compared with E21a threshold 0.2, E22a improves MAE/RMSE and startup slightly by using a lower calibrated threshold.
- Hysteresis was tested but did not beat the lower simple threshold in this batch.
- Current next slice: E22b should reduce the remaining hard-zero action-closeness penalty while preserving E22a's start40/tail suppression.

Update 2026-07-09 02:08 Asia/Shanghai:
- E22b soft-scale scan completed.
- Selected gate: `simple_0.15_s0.50`, meaning the phase gate uses threshold 0.15 and keeps 50% of E16 action during inactive steps.
- This preserves E22a deadzone behavior because the attenuated inactive residual stays below the runtime-scaled directional deadzones in the measured start40 and tail windows.
- E22b improves action closeness beyond E16: MAE 0.0414 vs E16 0.0448, RMSE 0.0861 vs E16 0.1000.
- E22b is now the current best offline candidate, but deployment is not yet proven. Next slice should stress it with visual-domain/texture-domain gates or package a runtime-latency smoke for the phase gate.

Update 2026-07-09 02:23 Asia/Shanghai:
- E23a/E23b low-pass visual stress completed for E16 weighted-intent eye2.
- Downsample_060 is too strong: startup effective drops to 18.6%, main drops to 72.9%.
- Downsample_080 is still concerning: MAE remains close to original at 0.0466, but startup effective drops to 48.6% and startup same-dir to 54.1%.
- Both transforms reduce early start40 bucket+ dramatically, but by suppressing true motion too. This is not an acceptable robustness solution.
- Current next slice: either cache transformed-image intent probabilities and apply E22b under the actual transformed visual inputs, or train a visual augmentation/domain-robust candidate before considering live deployment.

Update 2026-07-09 02:38 Asia/Shanghai:
- E24a/E24b completed transformed-image intent-probability caching and E22b-style soft phase gating for downsample_080 and downsample_060.
- Phase gating improves MAE/RMSE and tail quietness under transformed images, and reduces downsample_080 start40 effective motion from 4.4% to 0.3%.
- It does not recover startup/main should-move coverage: downsample_080 startup remains about 48%, and downsample_060 startup remains about 19%.
- Current conclusion: the blocker has moved from phase-gate design to visual-domain robustness of the base policy/features. Next slice should be E25 visual augmentation/domain robustness, not more gate threshold search.

Update 2026-07-09 02:44 Asia/Shanghai:
- Restated deadzone-aware acceptance criterion after user clarification: a candidate must promote should-move same-axis/same-direction crossings and suppress should-stop crossings. A quiet model that fails startup/main effective-motion gates is still rejected.
- E25 random visual low-pass augmentation launched:
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_random_downsample060100_weighted_intent_head_eye2.yaml`.
  - Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_random_downsample060100_weighted_intent_head_eye2`.
  - Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e25_random_downsample_weighted_intent_eye2_train.log`.
  - PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e25_random_downsample_weighted_intent_eye2_train.pid`.
  - Initial PID: 644540.
  - Health check: process alive, resolved config and dataset stats written, training reached epoch 25+, and `intent_axis_dir_loss` / `intent_loss` are present in logs.
- Verification before launch:
  - `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_dataset_image_transform.py -q` passed.
  - `PYTHONPATH=/home/pingfan/Excavator_real_stack/testbed python -m pytest testbed/tests/test_offline_policy_eval.py -q -k image_transform` passed.
  - `python -m py_compile testbed/testbed/data/image_transforms.py` passed.
- After E25 completes, evaluate the same way as E16/E23/E24:
  - Original-image all-train-ready replay and runtime-scaled gate.
  - Infer-only `downsample_080` replay/gate.
  - Infer-only `downsample_060` replay/gate.
  - If E25 recovers transformed startup/main without increasing start40/tail crossings, combine it with E22b-style phase gate.

Update 2026-07-09 03:29 Asia/Shanghai:
- E25 training completed:
  - Best epoch: 1695.
  - Best val loss: 0.114224.
  - Checkpoint: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_random_downsample060100_weighted_intent_head_eye2/policy_best.ckpt`.
- E25 replay outputs:
  - Original: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e25_random_downsample060100_eye2_original_all_train_ready_best`.
  - Downsample080: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e25_random_downsample060100_eye2_downsample_080_all_train_ready_best`.
  - Downsample060: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e25_random_downsample060100_eye2_downsample_060_all_train_ready_best`.
- E25 vs E16 gate output:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e25_random_downsample060100_vs_e16_gate_runtime_scaled`.
- E25 metrics:
  - Original replay: MAE 0.0556, RMSE 0.1058, policy p95 0.8390.
  - Downsample080 replay: MAE 0.0527, RMSE 0.0993, policy p95 0.8452.
  - Downsample060 replay: MAE 0.0535, RMSE 0.1017, policy p95 0.8425.
  - Original startup: policy effective 49.1%, same-dir 51.1%, extra/wrong 14.5%.
  - Downsample080 startup: policy effective 49.1%, same-dir 49.3%, extra/wrong 17.9%.
  - Downsample060 startup: policy effective 49.3%, same-dir 50.2%, extra/wrong 17.2%.
  - Original main: policy effective 94.5%, same-dir 93.5%, extra/wrong 2.1%.
  - Downsample080 main: policy effective 95.8%, same-dir 94.5%, extra/wrong 2.3%.
  - Downsample060 main: policy effective 95.5%, same-dir 94.1%, extra/wrong 2.5%.
  - Original tail: 0 / 504 effective frames.
  - Downsample080 tail: 6 / 504 effective frames.
  - Downsample060 tail: 6 / 504 effective frames.
- Decision: reject E25 as a final candidate. It answers one subproblem, because low-pass main-motion coverage is recovered versus E16 downsample060/downsample080, but it fails the movement-intent acceptance criterion at startup and introduces low-pass tail motion.
- Next slice: launch E26 with narrower stochastic augmentation `random_downsample_080_100_seed26`, keeping all other E25 variables fixed. This tests whether the E25 regression came from the overly broad 0.60-1.00 low-pass range rather than from stochastic augmentation itself.

Update 2026-07-09 03:32 Asia/Shanghai:
- E26 narrower random visual low-pass augmentation launched:
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/baseline_qpos_random_downsample080100_weighted_intent_head_eye2.yaml`.
  - Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_random_downsample080100_weighted_intent_head_eye2`.
  - Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e26_random_downsample080100_weighted_intent_eye2_train.log`.
  - PID file: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/logs/e26_random_downsample080100_weighted_intent_eye2_train.pid`.
  - Initial PID: 671294.
- E26 launch verification:
  - YAML parsed and points to `random_downsample_080_100_seed26`.
  - `build_image_transform("random_downsample_080_100_seed26")` succeeds.
  - No prior E26 checkpoint/log artifacts were present before launch.
  - Health check: process alive, resolved config and dataset stats written, training reached epoch 30+, and `intent_axis_dir_loss` / `intent_loss` are present in logs.
- After E26 completes, evaluate original, downsample080, and downsample060 with the same replay and runtime-scaled deadzone gates used for E25. E26 should be rejected unless it materially improves E25 startup/same-direction metrics while keeping E16-like MAE and no low-pass tail crossings.

Update 2026-07-09 04:15 Asia/Shanghai:
- E26 training completed:
  - Best epoch: 1695.
  - Best val loss: 0.092107.
  - Checkpoint: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/baseline_qpos_random_downsample080100_weighted_intent_head_eye2/policy_best.ckpt`.
- E26 replay outputs:
  - Original: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e26_random_downsample080100_eye2_original_all_train_ready_best`.
  - Downsample080: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e26_random_downsample080100_eye2_downsample_080_all_train_ready_best`.
  - Downsample060: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e26_random_downsample080100_eye2_downsample_060_all_train_ready_best`.
- E26 vs E16/E25 gate output:
  - `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e26_random_downsample080100_vs_e16_e25_gate_runtime_scaled`.
- E26 metrics:
  - Original replay: MAE 0.0494, RMSE 0.0980, policy p95 0.8250.
  - Downsample080 replay: MAE 0.0501, RMSE 0.0995, policy p95 0.8321.
  - Downsample060 replay: MAE 0.0506, RMSE 0.1002, policy p95 0.8259.
  - Original startup: policy effective 45.5%, same-dir 47.7%, extra/wrong 13.6%.
  - Downsample080 startup: policy effective 48.5%, same-dir 49.6%, extra/wrong 15.4%.
  - Downsample060 startup: policy effective 47.9%, same-dir 49.5%, extra/wrong 14.8%.
  - Original main: policy effective 93.8%, same-dir 93.7%, extra/wrong 1.1%.
  - Downsample080 main: policy effective 94.0%, same-dir 92.8%, extra/wrong 2.1%.
  - Downsample060 main: policy effective 92.4%, same-dir 91.1%, extra/wrong 2.3%.
  - Original tail: 0 / 504 effective frames.
  - Downsample080 tail: 6 / 504 effective frames.
  - Downsample060 tail: 6 / 504 effective frames.
- Decision: reject E26 as a final candidate. It improves replay MAE versus E25 but does not recover startup and does not remove low-pass tail crossings. This falsifies the idea that simply narrowing the low-pass range is enough.
- Updated direction: stop spending default training time on low-pass augmentation sweeps. Keep E22b as the current best offline control candidate, then focus the next slice on either:
  - applying E22b-style phase gating to E26 only as a diagnostic, or
  - building a real visual-domain split / embedding-cluster evaluation so texture robustness is measured by held-out domains rather than synthetic low-pass alone.

Update 2026-07-09 16:58 Asia/Shanghai:
- Operator question: the current best gate stack must be explainable as policy logic, not only as experiment ids and scalar metrics.
- Current E52/E51 gate-chain semantics:
  - Base action policy is still E16 eye2 ACT: `video4`, `video5`, `qpos`, with the auxiliary 8-direction intent head.
  - Per-frame gate features are `intent_prob[8] + qpos[4] + qvel[4]`; the lightweight gates do not inspect raw image pixels directly.
  - Phase gate label is data-derived: whether the expert action crosses any runtime-scaled directional deadzone. Runtime action change: inactive frames are attenuated by `inactive_scale=0.50`, not hard-zeroed.
  - Snap gate is deterministic calibration around the measured runtime-scaled deadzone: if a phase-active action is near deadzone and the ACT intent head is confident for that axis/direction, snap across the deadzone margin.
  - Temporal direction gate is a learned causal MLP over offsets `[-10, -5, -2, -1, 0]` of the same low-dimensional features. Runtime action change: directions below probability threshold are attenuated by `inactive_scale=0.75`.
  - Gohome request gate is learned from handoff labels, then guarded by consecutive-frame thresholds: tail candidate `>=0.97` for 10 frames and gohome eligibility `>=0.80` for 3 frames.
  - Fixed constants are therefore mostly operating thresholds selected by offline scans, not hand-coded visual texture rules. Their current weakness is domain proof: thresholds were selected on the 72-104 batch and stress-checked on excluded episodes, but not yet proven on broad held-out visual domains.
- E56 completed:
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/configs/e56_same_dir_only_promotion_eye2.yaml`.
  - Bundle: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/ckpts/e56_same_dir_only_promotion_eye2`.
  - Best epoch: 1730.
  - Best val loss: 0.08552186749875546.
  - Comparison package: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/policy_packages/e56_e52_gates_same_dir_action`.
  - Full-ACT E52-gate eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e56_e52_gates_full_act_all_train_ready`.
  - Deadzone startup/tail eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e56_e52_gates_full_act_all_train_ready_deadzone_runtime_scaled`.
  - Window eval: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/runs/offline_eval/e56_e52_gates_fullact_deadzone_window_eval`.
- E56 metrics versus E52/E51 current best:
  - E52/E51 temporal MAE/RMSE: 0.0408297 / 0.0864513; E56+same gates: 0.0403458 / 0.0881379.
  - E52/E51 gohome recall: 95.8%; E56+same gates: 75.0%.
  - E52/E51 startup first-effective: policy any 68.4%, same-dir 82.6%, extra/wrong 5.7%, 22/24 episodes >=50.
  - E56+same gates startup first-effective: policy any 48.6%, same-dir 54.7%, extra/wrong 8.3%, 10/24 episodes >=50.
  - E52/E51 tail: 0/504 effective frames; E56+same gates tail: 0/504 effective frames.
  - E56 longest-main same-dir is 90.7%, below E52/E51 at 95.2%.
- Decision: reject E56 as a final action-policy replacement. It slightly improves MAE but violates the user acceptance criterion that should-move startup/main crossings must not collapse. This confirms that even same-direction-only deadzone promotion can shift the action policy toward under-motion when judged by runtime deadzone intent gates.
- Current best remains E52/E51: E16 action policy plus the E52 causal gate stack. Next proof gap is not more threshold search; it is held-out visual/domain validation and the 105-109 new-batch evaluation.

Update 2026-07-09 17:04 Asia/Shanghai:
- New data hypothesis from operator: train ACT only on the semantic front part of a digging cycle, ending before the return-swing stage. This is not a mathematical 75% step crop; the cut means "dig/carry/dump complete, return/rotation begins."
- Added a focused semantic-cycle crop data builder:
  - Module: `testbed/testbed/data/semantic_cycle_crop.py`.
  - Tests: `testbed/tests/test_semantic_cycle_crop.py`.
  - Boundary: this module owns reviewed cut annotations and HDF5 prefix-cropping; `resample_20hz.py` remains the 50Hz-to-20Hz sampling owner, and `dataset.py` remains the training loader.
- Annotation artifacts:
  - Blank manual template: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/e58_semantic_cut_template_train_ready.csv`.
  - Heuristic proposal: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/e58_semantic_cut_proposals_needs_review.csv`.
  - Codex-reviewed CSV: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/e58_semantic_cut_codex_reviewed.csv`.
  - Contact sheets: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/contact_sheets`.
  - Overview image: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/e58_semantic_cut_proposal_index.png`.
- Crop construction:
  - Input: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/real_teleop_v1_episodes_72_104_20hz`.
  - Output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/real_teleop_v1_episodes_72_104_20hz_before_return`.
  - Cut rule used for this first reviewed pass: start of the final sustained swing-action run, visually checked against video4/video5 contact sheets as the return-swing start. This remains an experiment label, not a field-safety source of truth.
  - Result: 24 cropped train-ready episodes, lengths 452-651 steps, mean 538.2 steps.
- QC:
  - Output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/qc_before_return`.
  - Dataset QC: 24 / 24 success.
  - Training QC: 24 PASS, 0 WARN, 0 FAIL.
  - Train-ready manifest: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/qc_before_return/train_ready_manifest.json`.
- Main-segment semantic-crop old-batch baseline (E58) launch plan:
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/configs/e58_before_return_semantic_crop_weighted_intent_eye2.yaml`.
  - Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/runs/ckpts/e58_before_return_semantic_crop_weighted_intent_eye2`.
  - Split: same train/val ids as E16, with the split file rewritten for the cropped dataset path.
  - Loader smoke: 24 ids resolved, train 19 / val 5, loader `episode_len=651`.
  - Queue PID: 1029414.
  - Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260708_72_104/semantic_cycle_crop/runs/logs/e58_before_return_semantic_crop_weighted_intent_eye2_train.log`.
  - Queue behavior: waits for E57 PID 1020967 to exit before starting GPU training.
- Main-segment semantic-crop old-batch baseline (E58) acceptance question:
  - Does removing return-swing supervision improve ACT imitation on the remaining dig/carry/dump segment: lower MAE/RMSE, better expert-effective same-direction crossing recall, less over-duration, and no increased should-stop movement inside the cropped segment?
  - E58 does not answer gohome/tail safety by itself, because the return/gohome segment is intentionally outside its action scope. If E58 improves cropped-segment imitation, it should be paired with an explicit downstream handoff/return controller rather than deployed as a full-cycle policy.

Update 2026-07-09 17:13 Asia/Shanghai:
- Main-segment semantic-crop plus new-data baseline (E59) superseded E58 before training started, because the operator asked to include the new 105-109 data in the semantic-cycle dataset experiment.
- New 105-109 semantic crop artifacts:
  - Proposals: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/e59_semantic_cut_proposals_105_109_needs_review.csv`.
  - Reviewed cuts: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/e59_semantic_cut_105_109_codex_reviewed.csv`.
  - Overview image: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/e59_semantic_cut_proposal_index_105_109.png`.
  - Cropped output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/real_teleop_v1_episodes_105_109_20hz_before_return`.
- New crop QC:
  - 105-109 crop QC output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/qc_before_return_105_109`.
  - Result: 5 / 5 PASS after semantic crop.
  - Combined symlink dataset: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/combined_72_109_before_return`.
  - Combined QC output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/qc_combined_72_109_before_return`.
  - Combined QC result: 29 / 29 PASS.
- Main-segment semantic-crop plus new-data baseline (E59) selected training manifest:
  - Path: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/manifests/e59_train_ready_old24_plus105106_val107.json`.
  - Selection: old 24 cropped train-ready episodes plus new 105, 106, 107.
  - Reserved stress eval: 108, 109.
  - Split: original E16 train/val ids, plus 105/106 added to train and 107 added to val.
- Main-segment semantic-crop plus new-data baseline (E59) training launched:
  - Config: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/configs/e59_before_return_semantic_crop_plus105106_val107_weighted_intent_eye2.yaml`.
  - Checkpoint dir: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/ckpts/e59_before_return_semantic_crop_plus105106_val107_weighted_intent_eye2`.
  - Log: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/logs/e59_before_return_semantic_crop_plus105106_val107_weighted_intent_eye2_train.log`.
  - PID: 1034053.
  - Loader smoke: 27 ids resolved; train 21, val 6; loader `episode_len=651`.
  - Debug check: a separate 2-epoch run completed successfully at `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/ckpts/e59_debug_2epoch`.
  - Initial health: process alive under a detached `setsid` session, normalisation stats/resolved config/run metadata written, and epoch progress reached 45+.
- E57 completed while preparing E59:
  - Best epoch: 1675.
  - Best val loss: 0.11986956745386124.

Update 2026-07-09 18:08 Asia/Shanghai:
- Main-segment semantic-crop plus new-data baseline (E59) completed:
  - Checkpoint: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/ckpts/e59_before_return_semantic_crop_plus105106_val107_weighted_intent_eye2/policy_best.ckpt`.
  - Best epoch: 1495.
  - Best val loss: 0.09055651910603046.
  - Correction to the launch note above: the actual generated split is train 22 / val 5, not the intended "105/106 train + 107 val" split. Actual new-data placement is 106 and 107 in train, 105 in val. Use the actual split file as source of truth: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/ckpts/e59_before_return_semantic_crop_plus105106_val107_weighted_intent_eye2/train_val_split.yaml`.
- Formal offline eval was rerun after training completion, using `policy_best.ckpt` only:
  - E16 selected27 replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e16_baseline_selected27_final`.
  - E59 selected27 replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e59_best_selected27`.
  - E16 reserved 108/109 stress replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e16_baseline_stress108109_final`.
  - E59 reserved 108/109 stress replay: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e59_best_stress108109`.
  - Executable-action comparison artifact: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/semantic_cycle_crop/runs/offline_eval/main_segment_semantic_crop_e59_vs_e16_executable_metrics`.
- E59 selected27 outcome versus old full-cycle E16 baseline, evaluated on the same semantic-cropped selected27 data:
  - Overall MAE improved from 0.0515 to 0.0463.
  - Same-direction effective-frame recall was essentially unchanged: 91.5% -> 91.3%.
  - Miss rate on expert-effective movement frames was essentially unchanged: 7.8% -> 7.9%.
  - Idle false-active rate improved: 36.9% -> 29.4%.
  - Effective event precision improved: 73.8% -> 75.8%.
- E59 reserved 108/109 stress outcome:
  - Overall MAE improved from 0.0561 to 0.0527.
  - Same-direction effective-frame recall regressed: 85.8% -> 83.8%.
  - Miss rate regressed: 13.4% -> 15.4%.
  - Idle false-active rate improved strongly: 35.6% -> 13.8%.
  - Effective event precision improved: 77.3% -> 82.6%.
- Decision: E59 is a useful "quieter main-segment ACT" baseline, but not yet a clear solution to should-move startup/main deadzone crossing. The strongest observed improvement is less false active motion during expert-idle frames and slightly better event precision. It does not materially improve same-direction movement recall, and the actual validation split is small and partly different from the intended split, so this should not be treated as a final deployable policy.

Update 2026-07-09 18:34 Asia/Shanghai:
- Operator clarified the critical deadlock question: startup speed is not the acceptance criterion. The relevant liveness gate is whether, after the expert first enters a should-move state, the policy eventually emits a same-axis/same-direction action that crosses the runtime-scaled deadzone. Delay is measured in 20Hz steps, but slow liveness should be separated from a true no-motion deadlock.
- E52/E51 causal temporal gate stack was replayed on all five new full-cycle 105-109 episodes:
  - Output: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/offline_eval/e52_e51_full_act_temporal_gate_new105_109_liveness_test`.
  - Liveness metrics: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/runs/offline_eval/e52_e51_full_act_temporal_gate_new105_109_liveness_test/liveness_metrics`.
  - Dataset: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/real_teleop_v1_episodes_105_109_20hz`.
  - Handoff labels: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/handoff_eligibility_20hz_dwell10_105_109`.
  - Manifest: `/data/pingfan/Excavator_real_stack_data/usb_hdf5_qc_20260709_105_109/manifests/e57_new_all105_109_eval_manifest.json`, intentionally including QC-failed stress episodes 108 and 109.
- New-data liveness result:
  - Eventual any-effective liveness: 5 / 5.
  - Eventual same-direction liveness: 5 / 5.
  - Same-direction liveness within 5 steps: 5 / 5.
  - Max same-direction delay: 4 steps = 0.2 s at 20Hz.
  - Per-episode same-direction delays: episode_105 3 steps, episode_106 0, episode_107 4, episode_108 0, episode_109 2.
  - Mean same-direction frame recall across the five episodes: 88.7%; mean miss rate on expert-effective movement frames: 10.2%.
  - Interpretation: the new-data test does not show a no-motion deadlock for E52/E51. It does not prove closed-loop field success; it only proves recorded-scene offline liveness under the current runtime-scaled deadzone.
- New-data gohome/tail result from the same replay:
  - Gohome event recall: 4 / 5. Episode_106 missed gohome.
  - Pre-tail false-positive gohome: 0 / 5 episodes, 0 frames.
  - Early-but-dwell-region activity: episode_105 had 2 early frames.
  - This means gohome awareness remains a separate risk from action liveness.
- Deferred runtime fallback decision, not yet implemented:
  - If the policy is already near home, qpos remains effectively stationary, and the condition persists for 10 seconds, trigger one timeout auto-gohome request.
  - Intended failure mode: a missed learned gohome request should become a stationary timeout-to-gohome, not a stuck cycle requiring free policy motion.
  - This fallback is meant to address late/missed gohome only. It must not replace the early-gohome safety gate, and it should remain constrained to the near-home stationary tail condition.
