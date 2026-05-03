# 最终交付版项目说明

这份仓库现在按“课程展示可交付”来组织，主线目标不是追求一个已经成熟的
Diffusion Policy，而是交付一个可以稳定演示的闭环 MVP：

- 任务环境：ManiSkill `PickCube-v1`
- 观测方式：`rgbd`
- 干扰场景：pseudo-blur 造成视觉边界不清晰
- 主策略：`ScriptedPickCubePolicy`
- 主动感知：视觉不确定时触发小幅 probing
- 输出结果：成功率、reward、不确定性、probe 次数、decision trace

## 1. 你真正应该提交的主线

最终展示和报告建议以这条主线为准：

1. `clean`：无视觉干扰
2. `pseudo_blur`：有视觉干扰但不主动 probe
3. `active_probe`：有视觉干扰且启用主动探测

这三组对比已经能完整体现项目主题：

`视觉退化 -> 触发主动探测 -> 利用触觉式反馈修正决策`

## 2. 最终推荐入口

### 本地/通用最终入口

```bash
bash scripts/run_final_submission.sh
```

可选地把 BC 附加实验也一起跑：

```bash
RUN_BC_PIPELINE=1 bash scripts/run_final_submission.sh
```

### Colab 最终入口

```bash
bash scripts/run_colab_training_demo.sh
```

这个脚本会先：

- 跑 scripted baseline
- 自动校验 baseline 是否成功
- 再决定是否继续进入 BC pipeline

如果 scripted baseline 没有成功，它会直接停下，不会继续拿失败轨迹训练。

### 真实机械臂渲染入口

如果你要验证完整机械臂画面，而不是 mock fallback：

```bash
bash scripts/run_render_mecharm.sh
```

结果必须满足：

- `env_backend=maniskill`
- `fallback_used=False`
- `video_path` 指向生成的 mp4

否则说明算法闭环运行了，但当前机器没有成功初始化 ManiSkill/SAPIEN 渲染。

### 完整 policy 入口

当前仓库增加了一个最小条件 Diffusion Policy 路线：

```bash
bash scripts/run_diffusion_policy_pipeline.sh
```

快速验证可用：

```bash
NUM_EPISODES=10 TRAIN_EPOCHS=2 DIFFUSION_STEPS=10 bash scripts/run_diffusion_policy_pipeline.sh
```

它会生成：

- `runs/dp_mvp/diffusion_policy.pt`
- `runs/dp_mvp/diffusion_metrics.json`
- `results/dp_mvp/diffusion_eval/diffusion_eval_results.csv`

这条路线才是从 scripted MVP 走向完整学习式 policy 的正式路径。

## 3. 结果文件怎么看

核心结果在：

- `results/final_submission/clean/mvp_results.csv`
- `results/final_submission/pseudo_blur/mvp_results.csv`
- `results/final_submission/active_probe/mvp_results.csv`

建议重点看这些字段：

- `success_rate`
- `total_reward`
- `mean_uncertainty`
- `trigger_count`
- `probe_request_count`
- `final_boundary_confidence`
- `requested_policy`
- `effective_policy`
- `env_backend`
- `fallback_used`

## 4. 现在的项目结论

### 已经完成并可展示的部分

- scripted MVP 主线
- pseudo-blur 视觉不确定性检测
- active probing 闭环接口
- tactile-style boundary refinement 记录
- CSV / JSON / decision trace 输出
- Colab 与本地的统一脚本入口
- 自动结果校验，避免失败 demo 继续训练

### 仍属于附加实验的部分

- BC 训练与评估

当前 BC 流程已经可以完整跑通，但不应被当成最终核心成果。更稳妥的表述是：

`我们已经实现了可运行的 imitation-learning pipeline，用于后续替换 scripted controller；当前课程最终展示以 scripted active-perception MVP 为主。`

### 新增完整 policy 路线

新增 Diffusion Policy 框架后，可以更准确地表述为：

`我们已经实现了从主动感知 scripted MVP 到 sequence-level learned policy 的最小训练/评估路径；最终性能需要依赖更多真实 ManiSkill 成功 demonstrations。`

## 5. 最终答辩/报告建议说法

你可以直接这样讲：

1. 我们先在 ManiSkill PickCube 中构建了一个可重复的主动感知 MVP。
2. 在 pseudo-blur 场景下，系统先估计视觉不确定性。
3. 当边界信息不足时，策略触发 probing 动作。
4. probing 结果被转成 boundary confidence，再反馈给后续抓取决策。
5. 我们保留了 BC/DP 兼容接口，为后续学习式策略替换做好准备。

## 6. 如果在 Colab 上失败，优先怎么排查

先看结果中的：

- `env_backend`
- `fallback_used`
- `init_error`

判断方法：

- `env_backend=maniskill`：说明真环境在跑
- `env_backend=mock`：说明当前是在 fallback 路径验证闭环
- `fallback_used=True`：说明没有完全按目标环境执行

如果 scripted baseline 本身都没有成功，不要继续训练，先修 baseline。

## 7. 一句话总结

这份最终版本的核心不是“学出了最强策略”，而是已经交付了一条：

`可运行、可解释、可比较、可在 Colab/本地复现实验结果的主动感知项目主线`
