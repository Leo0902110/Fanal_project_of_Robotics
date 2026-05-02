# Active Perception MVP for ManiSkill PickCube

This project is an MVP for validating a closed-loop active perception idea in
ManiSkill: when RGB-D observations become unreliable under visual pseudo-blur,
the policy triggers a small tactile-style probing motion before continuing the
grasp.

The current implementation is intentionally lightweight. It verifies the
experiment loop, the pseudo-blur trigger, the active probing interface, and the
result visualization pipeline. A full Diffusion Policy training stack is not
implemented in this MVP; the code keeps a compatible policy interface so a
trained DP module can replace the scripted controller later.

## Project Scope

The MVP focuses on one task:

- Environment: ManiSkill `PickCube-v1`
- Observation: `rgbd` by default, with `state` as a fallback for smoke tests
- Failure mode: visual pseudo-blur simulated by depth noise and dropout
- Policy: `ScriptedPickCubePolicy`
- Active perception: visual uncertainty triggers 1-2 small probing actions
- Metrics: success rate, total reward, active probe count, mean uncertainty

## Core Idea

Visual pseudo-blur models scenes where transparent, dark, reflective, or
low-texture objects produce missing or misleading depth boundaries. Instead of
treating tactile/contact feedback as only a passive safety signal, this MVP uses
visual uncertainty as a trigger for active probing.

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

## Quick Start

Recommended final-deliverable entry:

```bash
bash scripts/run_final_submission.sh
```

Chinese delivery notes:

```text
FINAL_PROJECT_DELIVERABLE.md
```

Install the base dependencies first:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For BC training/evaluation, `torch` is also required. On Colab it is usually
preinstalled. For local runs, prefer Python 3.12 for the simulator stack.

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
python scripts/collect_demos.py --num-episodes 5 --scene pseudo_blur --policy scripted --use-active-probe --output-dir data/demos/pickcube_mvp
```

Train a minimal behavior cloning baseline:

```bash
python scripts/train_bc.py --demo-dir data/demos/pickcube_mvp --output-dir runs/bc_mvp
```

Run the end-to-end BC pipeline:

```bash
bash scripts/run_bc_pipeline.sh
```

The pipeline now validates that collected demos come from successful
`--policy scripted` rollouts before training starts. If the scripted baseline is
not succeeding, the script exits early instead of training on bad trajectories.

Use a specific interpreter when needed:

```bash
PYTHON_BIN=python3.12 bash scripts/run_bc_pipeline.sh
```

Evaluate the trained BC policy in the environment:

```bash
python scripts/evaluate_bc.py --checkpoint runs/bc_mvp/bc_policy.pt --scene pseudo_blur --use-active-probe --output-dir results/bc_eval
```

Compare BC against the fallback policy:

```bash
python scripts/evaluate_fallback.py --num-episodes 5 --scene pseudo_blur --use-active-probe --output-dir results/policy_comparison/fallback
python scripts/evaluate_bc.py --checkpoint runs/bc_mvp/bc_policy.pt --num-episodes 5 --scene pseudo_blur --use-active-probe --output-dir results/policy_comparison/bc
python scripts/plot_policy_comparison.py --fallback-csv results/policy_comparison/fallback/fallback_eval_results.csv --bc-csv results/policy_comparison/bc/bc_eval_results.csv
```

## Full Mechanical-Arm Render

To verify the real ManiSkill arm instead of the mock fallback, run the render
entry on a machine with working ManiSkill/SAPIEN rendering:

```bash
bash scripts/run_render_mecharm.sh
```

The run is considered a true mechanical-arm demo only when
`results/render_active_probe/mvp_results.csv` reports:

```text
env_backend=maniskill
fallback_used=False
video_path=results/render_active_probe/active_probe.mp4
```

If `env_backend=mock`, the algorithm loop still ran, but the visual arm render
did not initialize in that runtime.

## Minimal Full Diffusion Policy

The MVP now includes a small conditional Diffusion Policy path. It predicts a
short future action sequence from the same active-perception features used by
BC: flattened observation, visual uncertainty, boundary confidence, probe
state, probe point, and refined grasp target.

Run the full policy pipeline:

```bash
bash scripts/run_diffusion_policy_pipeline.sh
```

The pipeline performs:

1. collect successful scripted demonstrations;
2. validate the demo manifest;
3. train `runs/dp_mvp/diffusion_policy.pt`;
4. evaluate the learned policy;
5. compare against the sine fallback policy.

For quicker GitHub/Colab smoke verification, reduce the workload:

```bash
NUM_EPISODES=10 TRAIN_EPOCHS=2 DIFFUSION_STEPS=10 bash scripts/run_diffusion_policy_pipeline.sh
```

The main config reference is:

```text
configs/dp_pickcube.yaml
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
- active-perception decision trace
- metadata for scene, policy, fallback status, and success

These files can be loaded with `src/data/dataset.py`. The intended progression
is:

1. Collect scripted or fallback demonstrations.
2. Train a behavior cloning baseline on `(observation, action)` pairs.
3. Evaluate the BC policy back inside ManiSkill.
4. Add richer RGB-D or point-cloud encoders once the environment stack is
   stable.
5. Replace the BC baseline with a Diffusion Policy once the data and evaluation
   path are stable.

The current BC baseline already appends `uncertainty` and
`boundary_confidence` to the flattened observation vector:

```text
[flattened_observation, uncertainty, boundary_confidence] -> action
```

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
- The final DP module can be added later by implementing the same `predict`
  interface used by the current policies.
