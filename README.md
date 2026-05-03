# Active Perception MVP for ManiSkill PickCube

This project is an MVP for validating a closed-loop active perception idea in
ManiSkill: when RGB-D observations become unreliable under visual pseudo-blur,
the policy triggers a small tactile-style probing motion before continuing the
grasp.

The current implementation is intentionally lightweight. It verifies the
experiment loop, the pseudo-blur trigger, the active probing interface, and the
result visualization pipeline. It also includes an experimental
tactile-conditioned diffusion policy trained from the 500-episode D assist
teacher dataset; that prototype is useful for alignment with Diffusion Policy
framing, but the MLP D distillation checkpoint is currently much stronger in
online rollout.

## Project Scope

The MVP focuses on one task:

- Environment: ManiSkill `PickCube-v1`
- Observation: `rgbd` by default, with `state` as a fallback for smoke tests
- Failure mode: proposal-style visual pseudo-blur profiles that corrupt RGB-D
  boundaries like transparent, dark, reflective, or low-texture objects
- Policy: `ScriptedPickCubePolicy`
- Active perception: visual uncertainty triggers 1-2 small probing actions
- Metrics: success rate, total reward, active probe count, mean uncertainty

## Core Idea

Visual pseudo-blur models scenes where transparent, dark, reflective, or
low-texture objects produce missing or misleading depth boundaries. Instead of
treating tactile/contact feedback as only a passive safety signal, this MVP uses
visual uncertainty as a trigger for active probing.

The simulator task is still ManiSkill `PickCube-v1`, so the object geometry is a
cube. The proposal-aligned part is the observation model: `--scene pseudo_blur`
now supports explicit profiles that degrade RGB-D/position observations in ways
that mimic difficult object appearances:

| Profile | Intended proposal case | Observation corruption |
|---|---|---|
| `transparent` | transparent objects | high depth dropout, center/boundary missing pixels, mild desaturation |
| `dark` | dark objects | depth dropout plus darkened/noisy RGB |
| `reflective` | reflective objects | noisy/missing boundaries plus sparse highlight artifacts |
| `low_texture` | low-texture objects | desaturated RGB and moderate boundary ambiguity |
| `mild` | generic pseudo-blur | lightweight depth noise and dropout |

Use `--pseudo-blur-severity` to scale the profile. This keeps the project honest:
we are not claiming a new real material simulator; we are building a controlled
visual pseudo-blur benchmark for the opening proposal's failure mode.

The loop is:

1. Run RGB-D observation through a pseudo-blur uncertainty detector.
2. Convert visual uncertainty plus tactile/contact confidence into a decision
   ambiguity score.
3. If ambiguity is high and the probing budget is available, request a short
   tactile-style probing motion.
4. Convert the probe/contact outcome into a lightweight boundary confidence
   update.
5. Continue the grasp policy with the updated active-perception context.
6. Log success, reward, uncertainty, probing statistics, boundary confidence,
   and the decision trace.

## Active Perception Skeleton

The decision-to-probing flow is implemented as a lightweight coordinator in
`src/active_perception/pipeline.py`.

It has three explicit responsibilities:

- Estimate whether the current action decision is ambiguous.
- Decide whether the system should spend one probing step.
- Record the reason for each decision so the experiment is auditable.

The coordinator outputs an `ActivePerceptionDecision` object with:

- `ambiguity_score`
- `visual_uncertainty`
- `tactile_confidence`
- `state`
- `should_probe`
- `reason`

This object is passed into the policy through the normal `context` dictionary.
The policy only needs to consume `context["active_probe"]`; richer policy or DP
implementations can later use the full `context["active_perception"]` payload.

After a probe request, `src/tactile/boundary.py` computes a lightweight
`BoundaryRefinement` update. In the MVP fallback path this is a structured
confidence signal rather than a true tactile image boundary, but it establishes
the final feedback link:

```text
decision ambiguity -> tactile probe -> boundary confidence -> updated context
```

## Current Policy

`ScriptedPickCubePolicy` is a placeholder policy for the MVP. It uses a simple
state machine:

1. `approach`
2. `descend`
3. `close_gripper`
4. `transfer`
5. `release`

When `--use-active-probe` is enabled and visual uncertainty exceeds the
threshold, the policy performs a small lateral probing action before continuing
the grasp. This is not a learned Diffusion Policy yet, but it exercises the same
observation-policy-action interface needed by a later DP implementation.

`JointScriptedPickCubePolicy` is the Windows-compatible expert-ish option for
collecting better demonstrations while `pd_ee_delta_pose` is unavailable. It
uses `pd_joint_delta_pos`, consumes C's `active_probe` decision, and follows a
task-shaped sequence: approach the cube, descend, close the gripper, lift, move
toward the goal, then release.

```bash
python main.py --mode mvp --scene pseudo_blur --policy joint_scripted --obs-mode rgbd --use-active-probe --no-video --output-dir results/main_rgbd_joint_scripted
```

Run the same loop under a proposal-style transparent-object pseudo-blur profile:

```bash
python main.py --mode mvp --scene pseudo_blur --pseudo-blur-profile transparent --pseudo-blur-severity 1.0 --policy joint_scripted --obs-mode rgbd --use-active-probe --no-video --output-dir results/main_rgbd_joint_scripted_transparent
```

## Quick Start

Install the base dependencies first:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run a lightweight smoke test:

```bash
python main.py --mode smoke --obs-mode state --max-steps 30 --no-video
```

Run the full local MVP comparison:

```bash
bash scripts/run_local_mvp.sh
```

On Windows PowerShell:

```powershell
.\scripts\run_local_mvp.ps1
```

The script runs three conditions:

1. Clean: `--scene clean --policy scripted`
2. Pseudo-Blur: `--scene pseudo_blur --policy scripted`
3. Active-Probe: `--scene pseudo_blur --policy scripted --use-active-probe`

All runs use:

```text
--mode mvp --obs-mode rgbd --max-steps 120 --seed 42 --no-video
```

Results are written under:

```text
results/local_mvp_rgbd/
```

The summary chart is saved to:

```text
results/mvp_performance_chart.png
```

The decision-flow chart is saved to:

```text
results/mvp_decision_flow_chart.png
```

Decision trace animations can be exported as:

```text
results/active_probe_decision_trace.mp4
```

## Manual Commands

Clean baseline:

```bash
python main.py --mode mvp --scene clean --policy scripted --obs-mode rgbd --max-steps 120 --seed 42 --no-video --output-dir results/local_mvp_rgbd/clean
```

Pseudo-blur baseline:

```bash
python main.py --mode mvp --scene pseudo_blur --policy scripted --obs-mode rgbd --max-steps 120 --seed 42 --no-video --output-dir results/local_mvp_rgbd/pseudo_blur
```

Active probing:

```bash
python main.py --mode mvp --scene pseudo_blur --policy scripted --use-active-probe --obs-mode rgbd --max-steps 120 --seed 42 --no-video --output-dir results/local_mvp_rgbd/active_probe
```

Plot results:

```bash
python scripts/plot_results.py --results-dir results/local_mvp_rgbd
```

Collect active-perception demonstrations for future BC/DP training:

```bash
python scripts/collect_demos.py --num-episodes 5 --scene pseudo_blur --use-active-probe --output-dir data/demos/pickcube_mvp
```

Train a minimal behavior cloning baseline:

```bash
python scripts/train_bc.py --demo-dir data/demos/pickcube_mvp --output-dir runs/bc_mvp
```

Evaluate the trained BC policy in the environment:

```bash
python scripts/evaluate_bc.py --checkpoint runs/bc_mvp/bc_policy.pt --scene pseudo_blur --use-active-probe --output-dir results/bc_eval
```

For the current PickCube oracle-geometry BC checkpoint, the tuned grasp/transfer
assist reaches 18/20 successes (90%) over seeds 600-619. It uses
`pd_joint_pos` with small closed-loop qpos increments so the assist can regrasp
after a lost contact and settle once the cube is close to the goal:

```bash
python scripts/evaluate_bc.py --checkpoint runs/bc_oracle_geometry_success_50ep_gpu/bc_policy.pt --num-episodes 20 --max-steps 50 --seed 600 --obs-mode rgbd --control-mode pd_joint_pos --scene pseudo_blur --use-active-probe --device cuda --grasp-assist --assist-joint-position-scale 0.36 --assist-transfer-gain 2.0 --assist-settle-threshold 0.028 --assist-arm-gain 10.0 --assist-max-arm-command 1.0 --output-dir results/success90_oracle_geometry/bc_oracle_geometry_assist_pd_joint_pos
```

Compare BC against the fallback policy:

```bash
python scripts/evaluate_fallback.py --num-episodes 5 --scene pseudo_blur --use-active-probe --output-dir results/policy_comparison/fallback
python scripts/evaluate_bc.py --checkpoint runs/bc_mvp/bc_policy.pt --num-episodes 5 --scene pseudo_blur --use-active-probe --output-dir results/policy_comparison/bc
python scripts/plot_policy_comparison.py --fallback-csv results/policy_comparison/fallback/fallback_eval_results.csv --bc-csv results/policy_comparison/bc/bc_eval_results.csv
```

Render a decision-trace animation for presentation:

```bash
python scripts/render_decision_trace.py --input results/local_mvp_rgbd/active_probe/active_probe_decision_trace.csv --output results/active_probe_decision_trace.mp4
```

## Outputs

Each experiment directory contains:

- `mvp_results.csv`
- `mvp_results.json`
- `<experiment>_decision_trace.csv`

The CSV includes:

- `condition`
- `success_rate`
- `total_reward`
- `trigger_count`
- `visual_trigger_count`
- `mean_uncertainty`
- `decision_ambiguity_count`
- `probe_request_count`
- `contact_resolved_count`
- `refinement_count`
- `final_boundary_confidence`
- `tactile_contact_count`
- `requested_policy`
- `effective_policy`
- `fallback_used`

`trigger_count` is the number of active probing actions actually executed by
the policy. `visual_trigger_count` is the number of steps where the visual
uncertainty detector fired.

The decision trace CSV records the step-level flow from ambiguity detection to
probe request. It is useful for debugging and for explaining why active probing
was or was not triggered in a run.

Trace rows also include `boundary_confidence`, `confidence_delta`, and
`post_probe_uncertainty`, which show how the tactile probing skeleton feeds back
into later decisions.

## Toward Training

The next step toward a trained policy is demonstration collection. The script
`scripts/collect_demos.py` writes compressed `.npz` episodes under
`data/demos/`, with:

- flattened observations
- actions
- rewards and dones
- visual uncertainty
- boundary confidence
- tactile contact scalars from pairwise finger-cube forces when available
- post-probe uncertainty after tactile-style refinement
- active-perception decision trace
- metadata for scene, policy, fallback status, and success

These files can be loaded with `src/data/dataset.py`. The intended progression
is:

1. Collect scripted or fallback demonstrations.
2. Train a behavior cloning baseline on `(observation, action)` pairs.
3. Evaluate the BC policy back inside ManiSkill.
4. Add richer RGB-D or point-cloud encoders once the environment stack is
   stable.
5. Replace or compare the BC baseline with a tactile-conditioned Diffusion
  Policy once the data and evaluation path are stable.

The current BC baseline appends visual and tactile refinement signals to the
flattened observation vector. Older demo files and checkpoints remain readable,
but new training runs use:

```text
[flattened_observation,
 uncertainty,
 boundary_confidence,
 contact_detected,
 contact_strength,
 left_force_norm,
 right_force_norm,
 net_force_norm,
 pairwise_contact_used,
 post_probe_uncertainty] -> action
```

## D Policy Entry

The D tactile-fusion policy can be exercised through the main runner with:

```bash
python main.py --mode mvp --scene pseudo_blur --policy d --obs-mode rgbd --use-active-probe --no-video --output-dir results/main_rgbd_d_policy_smoke
```

This route uses the Windows-compatible `pd_joint_delta_pos` control mode and
feeds C's visual uncertainty into D's `ActiveTactilePolicy` interface. It is an
integration path for D's active tactile module; without trained D weights, it
should be treated as a smoke/integration baseline rather than a final grasping
policy.

For better behavior-cloning demonstrations than the sinusoidal fallback, collect
with the joint scripted policy:

```bash
python scripts/collect_demos.py --policy joint_scripted --obs-mode rgbd --scene pseudo_blur --use-active-probe --output-dir data/demos/pickcube_joint_scripted
```

To train D from the current 90% grasp/transfer teacher, first collect assist
distillation data, then train `ActiveTactilePolicy` on GPU:

```bash
python scripts/collect_d_assist_demos.py --num-episodes 50 --max-steps 50 --seed 1000 --obs-mode rgbd --scene pseudo_blur --output-dir data/demos/pickcube_d_assist_teacher_50
python scripts/train_d_policy.py --demo-dir data/demos/pickcube_d_assist_teacher_50 --output-dir runs/d_policy_assist_teacher_50_gpu --epochs 30 --batch-size 256 --hidden-dim 128 --device cuda --log-every 5
python scripts/evaluate_d_policy.py --checkpoint runs/d_policy_assist_teacher_50_gpu/d_policy.pt --num-episodes 10 --max-steps 50 --seed 1200 --obs-mode rgbd --control-mode pd_joint_pos --joint-position-scale 0.36 --scene pseudo_blur --device cuda --output-dir results/d_policy_assist_teacher_50_eval
```

The first 50-episode D distillation run collected 47/50 successful teacher
episodes. The GPU-trained D checkpoint reached validation phase accuracy around
87.8%; its first direct rollout reached 2/10 success and 8/10 grasp, so it is a
working D baseline but not yet a replacement for the hand-written assist.

Scaling the same D distillation path to 500 teacher episodes gives a much
stronger D policy:

```bash
python scripts/collect_d_assist_demos.py --num-episodes 500 --max-steps 50 --seed 2000 --obs-mode rgbd --scene pseudo_blur --output-dir data/demos/pickcube_d_assist_teacher_500
python scripts/train_d_policy.py --demo-dir data/demos/pickcube_d_assist_teacher_500 --output-dir runs/d_policy_assist_teacher_500_gpu_h256 --epochs 60 --batch-size 512 --hidden-dim 256 --device cuda --log-every 10
python scripts/evaluate_d_policy.py --checkpoint runs/d_policy_assist_teacher_500_gpu_h256/d_policy.pt --num-episodes 20 --max-steps 50 --seed 3000 --obs-mode rgbd --control-mode pd_joint_pos --joint-position-scale 0.36 --scene pseudo_blur --device cuda --output-dir results/d_policy_assist_teacher_500_eval_seed3000_scale036
```

This 500-demo run collected 469/500 successful teacher episodes (10,591
transitions). The GPU-trained D checkpoint reached validation action loss around
0.0060 and validation phase accuracy around 94.7%. Direct D rollout reached
18/20 success on seeds 3000-3019 and 20/20 success on seeds 1200-1219, with
20/20 grasp in both evaluations.

## Tactile-Conditioned Diffusion Policy

The repository now includes a DDPM-style tactile-conditioned action policy that
uses the same D teacher dataset. The model denoises short action horizons from
conditions built from `vision_features` and `tactile_features`, and includes a
supervised phase head so the denoiser can condition on teacher phases such as
`align_xy`, `descend`, `close`, `transfer`, and `settle`.

Train the horizon-4 prototype on GPU:

```bash
python scripts/train_tactile_diffusion_policy.py --demo-dir data/demos/pickcube_d_assist_teacher_500 --output-dir runs/tactile_dp_teacher_500_phase_h256 --epochs 50 --batch-size 512 --hidden-dim 256 --action-horizon 4 --diffusion-steps 50 --device cuda --phase-loss-weight 0.1 --log-every 10
```

Evaluate with deterministic low-noise sampling and short-plan execution:

```bash
python scripts/evaluate_tactile_diffusion_policy.py --checkpoint runs/tactile_dp_teacher_500_phase_h256/tactile_diffusion_policy.pt --num-episodes 5 --max-steps 50 --seed 3000 --sample-steps 50 --replan-interval 4 --init-noise-scale 0.2 --device cuda --output-dir results/tactile_dp_teacher_500_phase_eval_seed3000_noise02_5ep
```

The direct action-generation diffusion prototype trains successfully
(`val_noise_mse` about 0.338 after 50 epochs) and can run online, but it is
less stable than the MLP D policy. The stronger current route is residual
diffusion: use the 500-demo MLP D policy as a stabilizing base controller and
train diffusion to denoise residual corrections around that action.

Train the residual tactile diffusion policy:

```bash
python scripts/train_tactile_diffusion_policy.py --demo-dir data/demos/pickcube_d_assist_teacher_500 --output-dir runs/tactile_dp_residual_d500_h1_h256 --epochs 40 --batch-size 512 --hidden-dim 256 --action-horizon 1 --diffusion-steps 50 --device cuda --phase-loss-weight 0.1 --base-d-checkpoint runs/d_policy_assist_teacher_500_gpu_h256/d_policy.pt --log-every 10
```

Evaluate it with conservative residual scaling:

```bash
python scripts/evaluate_tactile_diffusion_policy.py --checkpoint runs/tactile_dp_residual_d500_h1_h256/tactile_diffusion_policy.pt --base-d-checkpoint runs/d_policy_assist_teacher_500_gpu_h256/d_policy.pt --num-episodes 20 --max-steps 50 --seed 3000 --sample-steps 50 --replan-interval 1 --init-noise-scale 0.0 --residual-scale 0.05 --device cuda --output-dir results/tactile_dp_residual_d500_h1_seed3000_scale005_fixed_20ep
```

Evaluate the same residual DP under the proposal-aligned transparent-object
pseudo-blur profile:

```bash
python scripts/evaluate_tactile_diffusion_policy.py --checkpoint runs/tactile_dp_residual_d500_h1_h256/tactile_diffusion_policy.pt --base-d-checkpoint runs/d_policy_assist_teacher_500_gpu_h256/d_policy.pt --num-episodes 20 --max-steps 50 --seed 3000 --sample-steps 50 --replan-interval 1 --init-noise-scale 0.0 --residual-scale 0.05 --scene pseudo_blur --pseudo-blur-profile transparent --pseudo-blur-severity 1.0 --device cuda --output-dir results/proposal_pseudoblur/tdp_residual_transparent_seed3000
```

For the stronger proposal profiles, use condition calibration to keep normalized
vision/tactile inputs inside the D/DP training distribution:

```bash
python scripts/evaluate_tactile_diffusion_policy.py --checkpoint runs/tactile_dp_residual_d500_h1_h256/tactile_diffusion_policy.pt --base-d-checkpoint runs/d_policy_assist_teacher_500_gpu_h256/d_policy.pt --num-episodes 20 --max-steps 50 --seed 3000 --sample-steps 50 --replan-interval 1 --init-noise-scale 0.0 --residual-scale 0.05 --condition-clip-sigma 2.0 --scene pseudo_blur --pseudo-blur-profile transparent --pseudo-blur-severity 1.0 --device cuda --output-dir results/proposal_pseudoblur_clip/tdp_residual_transparent_clip2_seed3000
```

For a compact proposal benchmark, repeat the command with
`--pseudo-blur-profile dark`, `reflective`, and `low_texture`, then compare
against `--scene clean`. The result CSVs record `pseudo_blur_profile` and
`pseudo_blur_severity` so the table can distinguish ordinary PickCube from the
visual pseudo-blur object scenarios.

The proposal-style stress tests are summarized at
`results/proposal_pseudoblur/proposal_pseudoblur_summary.md` and
`results/proposal_pseudoblur_clip/proposal_pseudoblur_clip_comparison.md`.
Without condition calibration, strong profiles push the visual condition features
outside the training distribution and the policy stays in `align_xy`. With
`--condition-clip-sigma 2.0`, the same residual DP recovers 95% success on all
four opening-proposal profiles:

| Profile | No calibration | Calibrated (`clip_sigma=2.0`) | Grasp after calibration |
|---|---:|---:|---:|
| transparent | 0/20 | 19/20 | 20/20 |
| dark | 0/20 | 19/20 | 20/20 |
| reflective | 0/20 | 19/20 | 20/20 |
| low-texture | 0/20 | 19/20 | 20/20 |
| mild pseudo-blur | - | 19/20 without calibration | 20/20 |

The current practical fix is inference-time condition clipping, which is a
distribution-calibration layer rather than a new teacher. Profile-aware teacher
data and retraining remain useful future work, but they are no longer required
to reach the 90% success target in this simulated proposal benchmark.

Residual tactile diffusion reached 18/20 success with 20/20 grasp on seeds
3000-3019, and 19/20 success with 20/20 grasp on seeds 1200-1219. This meets
the 90% target while keeping the policy explicitly diffusion-conditioned: the
diffusion model samples a tactile-conditioned residual, and the D MLP provides
the stable nominal action.

The current key result table can be regenerated with:

```bash
python scripts/summarize_key_results.py --results-dir results --output-dir results/summary
```

The generated Markdown table is saved at
`results/summary/key_results_summary.md`. The latest residual-scale ablation is:

| Method | Split | Success | Grasp |
|---|---:|---:|---:|
| D-only path (`residual_scale=0.00`) | seeds 3000-3019 | 19/20 | 20/20 |
| D-only path (`residual_scale=0.00`) | seeds 1200-1219 | 20/20 | 20/20 |
| Residual DP (`residual_scale=0.05`) | seeds 3000-3019 | 18/20 | 20/20 |
| Residual DP (`residual_scale=0.05`) | seeds 1200-1219 | 19/20 | 20/20 |
| Residual DP (`residual_scale=0.10`) | seeds 3000-3019 | 19/20 | 20/20 |
| Residual DP (`residual_scale=0.10`) | seeds 1200-1219 | 18/20 | 20/20 |

This means the next experimental work should focus on reporting and optional
handoff integration rather than mixing BC and D/DP datasets. BC remains the
visual imitation baseline; residual tactile diffusion is the current main
method.

## Repository Layout

```text
main.py
src/
  active_perception/pipeline.py
  env/wrapper.py
  models/policies.py
  perception/pseudo_blur.py
  tactile/boundary.py
  tactile/contact.py
scripts/
  collect_demos.py
  train_bc.py
  evaluate_bc.py
  evaluate_fallback.py
  plot_policy_comparison.py
  render_decision_trace.py
  run_local_mvp.sh
  plot_results.py
requirements.txt
```

## Notes

- The MVP is designed for reproducibility and project delivery, not maximum
  grasping performance.
- If RGB-D rendering is unavailable on a local machine, use `--obs-mode state`
  for smoke testing.
- On Windows without `pin/pinocchio`, ManiSkill may reject the Panda
  end-effector controller used by `ScriptedPickCubePolicy`. In that case the
  runner falls back to a simple sine policy and records `fallback_used=True`.
- When the environment falls back to `state` observations, pseudo-blur is
  represented by a surrogate uncertainty signal. This keeps the active
  perception loop testable, but it is not a substitute for a full RGB-D
  rendering experiment.
- A tactile-conditioned diffusion policy is now implemented. The strongest
  current variant is residual diffusion around the trained D MLP controller.
