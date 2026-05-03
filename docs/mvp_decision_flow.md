# MVP 决策流程图

这份图用于解释项目从最小可行原型到当前 material-stress 实验的决策路径。核心逻辑不是“一开始就做复杂模型”，而是每一步都用实验结果决定是否升级方案。

## 老师汇报版

```mermaid
flowchart TD
    A[研究目标：困难视觉材质下稳定抓取] --> B[先做 PickCube MVP]
    B --> C{脚本专家能否稳定完成接近、夹取、抬升?}
    C -- 否 --> C1[调 joint_scripted 抓取和 lift 阶段]
    C1 --> C
    C -- 是 --> D[采集 demos]

    D --> E{BC / sine baseline 是否能说明问题?}
    E -- 否 --> E1[加入 D-assist teacher 和主动探测]
    E1 --> F[训练 D policy]
    E -- 是 --> F

    F --> G[训练 tactile-conditioned residual diffusion policy]
    G --> H{普通/伪模糊场景 success 是否接近 90%?}
    H -- 否 --> H1[补数据、调 residual scale、采样步数和评估参数]
    H1 --> G
    H -- 是 --> I[进入材质鲁棒性实验]

    I --> J{只改渲染材质是否可信?}
    J -- 否：不同材质结果太像 --> K[材料外观 + material-matched observation stress]
    J -- 是 --> L[保留为初步 ablation]
    K --> M[重新采集 500 条 material-stress demos]
    M --> N[重训 D policy 和 residual tactile diffusion policy]
    N --> O[每类材质 50 episodes 评估]
    O --> P{总体 success 是否达到约 90%?}
    P -- 否 --> P1[继续增加数据或调整透明/暗色 profile]
    P1 --> M
    P -- 是 --> Q[形成实验故事：91% success, 100% grasp, Wilson 95% CI]
```

## 工程执行版

```mermaid
flowchart LR
    A[定义 MVP 指标] --> B[脚本专家]
    B --> C[demo collection]
    C --> D[BC baseline]
    D --> E[D-assist policy]
    E --> F[residual tactile diffusion]
    F --> G[material-stress retraining]
    G --> H[50 episodes/profile evaluation]
    H --> I[charts + report + demo]

    B -. gate .-> B1{专家成功率可用?}
    C -. gate .-> C1{demo 位置和 seed 足够多样?}
    D -. gate .-> D1{baseline 差异能被解释?}
    F -. gate .-> F1{success 接近 90%?}
    G -. gate .-> G1{观测流真的随材质变难?}
    H -. gate .-> H1{样本数和置信区间足够严谨?}

    B1 -- 否 --> B
    C1 -- 否 --> C
    D1 -- 否 --> E
    F1 -- 否 --> F
    G1 -- 否 --> G
    H1 -- 否 --> H
```

## 关键决策点

| 阶段 | 判断问题 | 通过标准 | 不通过时的动作 |
| --- | --- | --- | --- |
| MVP 环境 | 任务能否跑通并复现实验? | PickCube 可稳定 reset、rollout、记录指标 | 修 wrapper、控制模式、seed 和视频保存 |
| 专家策略 | demos 是否像真实抓取动作? | 有接近、闭合、抬升、移动阶段 | 调 joint delta、gripper 时序、lift 高度 |
| 数据质量 | demo 是否足够多样? | 不同初始位置、不同 seed、成功 episode 可筛选 | 扩展采集范围，重采失败段 |
| 模型路线 | BC 是否够用? | BC 能超过 sine，并能作为 baseline | 升级到 D-assist 和 diffusion policy |
| 触觉融合 | 触觉/主动探测是否带来收益? | residual TDP 超过 D-only 或 baseline | 调触觉特征、condition clip、residual scale |
| 材质实验 | 材质变化是否真的影响观测? | 不同材质下 RGB-D / pseudo-blur 输入不同 | 使用 material-matched visual stress |
| 结果可信度 | 结果是否能讲给老师? | 每类 50 episodes，有 CI，有 demo，有局限性说明 | 增大评估规模或重画图表 |

## 当前项目所处位置

当前项目已经走到最后一个 gate：material-stress 版本已经完成重新采集、重新训练和 50 episodes/profile 评估。最终结果为 182/200 success，即 91%；200/200 grasp，即 100%。因此现在的 MVP 故事可以表述为：

1. 先用最小 PickCube MVP 证明抓取任务闭环可运行。
2. 发现单纯 BC 和 render-only material 实验不足以支撑“困难材质鲁棒性”。
3. 引入 D-assist teacher、触觉条件 residual diffusion 和 material-matched visual stress。
4. 用更真实的材质压力重新训练，并用更大规模评估支撑最终结论。