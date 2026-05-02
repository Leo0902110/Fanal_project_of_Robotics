# Google Colab 一键训练演示

目标不是本地可跑，而是让 Colab 上完整跑通：

`采 demo -> 训练 BC -> 评估 BC -> 对照 fallback -> 输出结果表`

## 1. 挂载并进入仓库

```python
from google.colab import drive
drive.mount('/content/drive')
```

```python
%cd /content/drive/MyDrive
!mkdir -p Robotics_Final
%cd Robotics_Final

import os
if not os.path.exists("Fanal_project_of_Robotics"):
    !git clone https://github.com/Leo0902110/Fanal_project_of_Robotics.git
else:
    %cd Fanal_project_of_Robotics
    !git pull
    %cd ..

%cd Fanal_project_of_Robotics
```

## 2. 一键完整训练演示

标准版本：

```python
!bash scripts/run_colab_training_demo.sh
```

快速验证版本：

```python
!NUM_EPISODES=8 MAX_STEPS=60 TRAIN_EPOCHS=5 bash scripts/run_colab_training_demo.sh
```

如果你只想跑训练，不想先跑 smoke 和 baseline：

```python
!RUN_SMOKE_FIRST=0 RUN_MVP_BASELINES=0 bash scripts/run_colab_training_demo.sh
```

如果你还想顺便尝试渲染演示视频：

```python
!RENDER_DEMO_VIDEO=1 bash scripts/run_colab_training_demo.sh
```

## 3. 看结果

```python
import pandas as pd

bc = pd.read_csv("results/colab_vtabr/bc_eval/bc_eval_results.csv")
fallback = pd.read_csv("results/colab_vtabr/fallback_eval/fallback_eval_results.csv")

print("BC mean reward:", bc["total_reward"].mean())
print("Fallback mean reward:", fallback["total_reward"].mean())

bc, fallback
```

## 4. 关键产物

```text
data/demos/colab_vtabr/
runs/colab_vtabr/bc_policy.pt
runs/colab_vtabr/bc_metrics.json
results/colab_vtabr/bc_eval/bc_eval_results.csv
results/colab_vtabr/fallback_eval/fallback_eval_results.csv
```

## 5. 说明

- 这个入口优先保证 Colab 上“完整训练演示”跑通。
- 如果 ManiSkill 渲染或设备初始化失败，当前代码会优先保证训练闭环不断，而不是直接中断整个流程。
- 真正要做最终课程展示时，再单独打开 `Robotics_Project_Colab_Render_MechArm.ipynb` 处理机械臂视频渲染。
