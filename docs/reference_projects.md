# Reference Projects for Training Roadmap

This document records external projects worth borrowing structure from as this
MVP grows from an active-perception prototype into a trained policy pipeline.

## ManiSkill Learning from Demonstrations

- Link: https://maniskill.readthedocs.io/en/v3.0.0b21/user_guide/learning_from_demos/
- Repository: https://github.com/haosulab/ManiSkill

ManiSkill is the first reference to follow because our environment and task are
already ManiSkill-based. Its learning-from-demonstrations guide provides the
closest official path for trajectory collection, replay, and imitation learning
baselines.

Borrow for this project:

- Standardized demonstration collection and replay flow.
- Evaluation conventions for success rate and task metrics.
- Future migration from local `.npz` demo files to ManiSkill-style trajectory
  files when the full RGB-D / controller stack is stable.

## robomimic

- Link: https://robomimic.github.io/docs/
- Repository: https://github.com/ARISE-Initiative/robomimic

robomimic is a mature imitation-learning codebase built around demonstration
datasets, training configs, algorithm modules, and evaluation scripts. It is a
good template for behavior cloning before moving to diffusion policies.

Borrow for this project:

- HDF5-style dataset organization.
- Separation between dataset loading, policy implementation, training loop, and
  evaluation.
- BC-first training approach before trying heavier generative policies.

## Diffusion Policy

- Repository: https://github.com/real-stanford/diffusion_policy
- Project page: https://diffusion-policy.cs.columbia.edu/

The original Diffusion Policy codebase is the reference for action-sequence
diffusion in robot manipulation. It uses workspace/config abstractions and
predicts action horizons conditioned on observation history.

Borrow for this project:

- Observation horizon and action horizon design.
- Dataset normalization and replay-buffer ideas.
- Training loop organization for a future `train_diffusion_policy.py`.

## 3D Diffusion Policy / DP3

- Project page: https://3d-diffusion-policy.github.io/
- Paper: https://arxiv.org/abs/2403.03954

DP3 is useful if this project moves from low-dimensional state or RGB-D images
to point-cloud observations. It keeps 3D representations compact and conditions
diffusion policies on 3D visual features plus robot state.

Borrow for this project:

- Point-cloud observation encoder strategy.
- Compact 3D visual representation before action prediction.
- A possible upgrade path after behavior cloning works.

## LeRobot

- Repository: https://github.com/huggingface/lerobot

LeRobot is useful later if we want a modern robotics-training stack with
standardized datasets, policies, and evaluation scripts. It is more framework
heavy than we need right now, but its dataset/policy registry direction is worth
watching.

Borrow for this project:

- Dataset and model packaging conventions.
- Policy registry and training CLI style.
- Possible future compatibility with broader robotics tooling.

## Tactile / Force-Aware Policy References

- TacDiffusion: https://github.com/popnut123/TacDiffusion
- FARM project page: https://tactile-farm.github.io/

These are closer to the tactile side of the idea than to our current codebase.
They are useful for framing how tactile or force signals can condition robot
policies, even if their implementation details are not directly compatible with
ManiSkill PickCube.

Borrow for this project:

- Tactile/force feature conditioning.
- Metrics for showing that contact feedback changes policy behavior.
- Language for explaining why active tactile probing is more than a safety
  signal.

## Current Local Roadmap

The practical path for this repository is:

1. Keep collecting `.npz` demos with `scripts/collect_demos.py`.
2. Train a small behavior cloning model with `scripts/train_bc.py`.
3. Add evaluation for the trained BC model.
4. Export or convert demos to a ManiSkill/robomimic-style HDF5 format.
5. Replace the BC model with a Diffusion Policy once the data and evaluation
   path are stable.
