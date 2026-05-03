# Disk Cleanup Plan

This project now has a final material-object story and presentation demos. The largest removable files are early/raw BC demonstration datasets, not the final videos or summaries.

## Must Keep For Final Result Sharing

Keep these files/directories for the current teacher-facing story:

- `src/`, `models/`, `utils/`, `scripts/`, `README.md`, `docs/`
- `runs/tactile_dp_residual_d500_h1_h256/`
- `runs/d_policy_assist_teacher_500_gpu_h256/`
- `data/demos/pickcube_d_assist_teacher_500/`
- `results/material_object_position_demo/`
- `results/material_object_eval/`
- `results/proposal_pseudoblur_clip/`
- `results/proposal_pseudoblur/`
- `results/summary/`

Why: these support the final story: residual tactile-conditioned diffusion policy, material-level object profiles, multi-position demo videos, and key result tables.

## Largest Safe-To-Delete Raw Data

These are raw BC/early demo datasets. Delete them if you do not need to retrain BC baselines:

| Directory | Approx. size | Keep only if... |
| --- | ---: | --- |
| `data/demos/pickcube_joint_scripted_500/` | 1788 MB | You want to retrain BC all-demo |
| `data/demos/pickcube_joint_scripted_success_oracle_geometry/` | 264 MB | You want to retrain BC oracle geometry |
| `data/demos/pickcube_joint_scripted_success_oracle_geometry_raw/` | 264 MB | You want raw success-only extraction history |
| `data/demos/pickcube_joint_scripted_success_only/` | 264 MB | You want to retrain BC success-only |
| `data/demos/pickcube_rgbd_active_v1/` | 45 MB | You want early active-perception BC data |
| `data/demos/pickcube_joint_scripted_smoke/` | 4 MB | You want smoke demo files |
| `data/demos/pickcube_comprehensive_smoke/` | <1 MB | You want old smoke demo files |

Expected reclaim: about **2.63 GB**.

## Old Checkpoints You Can Delete If Only Showing Final Results

These are not needed for the final material-object demo or residual DP result:

- `runs/bc_benchmark_cpu_5ep/`
- `runs/bc_benchmark_gpu_5ep/`
- `runs/bc_joint_scripted_500/`
- `runs/bc_oracle_geometry_success_50ep_gpu/`
- `runs/bc_success_only_50ep_gpu/`
- `runs/bc_rgbd_active_v1/`
- `runs/bc_rgbd_active_v1_norm/`
- `runs/bc_rgbd_active_v2_norm/`
- `runs/bc_comprehensive_smoke/`
- `runs/d_policy_assist_teacher_50_gpu/`
- `runs/d_policy_assist_teacher_smoke/`
- `runs/tactile_dp_teacher_500_gpu_h256/`
- `runs/tactile_dp_teacher_500_phase_h1_h256/`
- `runs/tactile_dp_teacher_500_phase_h256/`
- `runs/tactile_dp_teacher_500_phase_smoke/`
- `runs/tactile_dp_teacher_500_smoke/`

Expected reclaim: about **221 MB**.

## Old Results Mostly Not Worth Keeping

Old smoke/debug/direct-DP result folders are small. Delete them only if you want a tidy workspace; expected reclaim is about **2 MB**.

The final material-object videos and summaries are tiny and useful, so do not delete them.

## Do Not Delete Unless You Are Rebuilding The Environment

- `.venv/`

This is much larger than the project artifacts, but deleting it removes PyTorch, CUDA, ManiSkill, SAPIEN, and ffmpeg packages. Only delete `.venv/` if you are ready to reinstall dependencies.

## Recommended Cleanup Command

First dry-run:

```powershell
.\scripts\cleanup_disk_space.ps1 -Tier RawDemos
```

If the list looks right, execute:

```powershell
.\scripts\cleanup_disk_space.ps1 -Tier RawDemos -Execute
```

To also remove old non-final checkpoints:

```powershell
.\scripts\cleanup_disk_space.ps1 -Tier RawDemos,OldRuns -Execute
```
