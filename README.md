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
2. If uncertainty is high, execute a short probing motion during approach.
3. Continue the scripted grasp state machine.
4. Log success, reward, uncertainty, and probing statistics.

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

## Outputs

Each experiment directory contains:

- `mvp_results.csv`
- `mvp_results.json`

The CSV includes:

- `condition`
- `success_rate`
- `total_reward`
- `trigger_count`
- `visual_trigger_count`
- `mean_uncertainty`
- `tactile_contact_count`
- `requested_policy`
- `effective_policy`
- `fallback_used`

`trigger_count` is the number of active probing actions actually executed by
the policy. `visual_trigger_count` is the number of steps where the visual
uncertainty detector fired.

## Repository Layout

```text
main.py
src/
  env/wrapper.py
  models/policies.py
  perception/pseudo_blur.py
  tactile/contact.py
scripts/
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
