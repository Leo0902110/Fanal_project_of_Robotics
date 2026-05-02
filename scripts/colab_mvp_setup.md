# Google Colab MVP 运行说明

在 Colab 中按顺序运行下面这些单元格即可。

## 1. 挂载 Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## 2. 拉取或更新代码

公开仓库：

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

如果仓库是私有的，不要把 token 写进 notebook。请用 Colab Secrets 或临时手动认证。

## 3. 安装依赖

```python
!python -m pip install --upgrade pip
!pip install -r requirements.txt
!pip install -r requirements-A.txt
```

## 4. 先跑 smoke test

```python
!python main.py --mode smoke --obs-mode state --max-steps 30 --no-video
```

如果这个成功，说明 ManiSkill 的基本仿真链路可用。

## 5. 跑 RGBD MVP 实验

```python
!python main.py --mode mvp --obs-mode rgbd --max-steps 120 --output-dir results/mvp
```

如果 Colab 的 Vulkan/RGBD 渲染失败，先退回 state 版本：

```python
!python main.py --mode mvp --obs-mode state --max-steps 120 --no-video --output-dir results/mvp_state
```

## 6. 跑完整 BC 训练链

推荐直接使用一键 Colab 脚本：

```python
!bash scripts/run_colab_training_demo.sh
```

更短的专用说明见：

```text
scripts/colab_training_demo.md
```

如果你只想调用训练链本体，也可以直接跑：

```python
!bash scripts/run_bc_pipeline.sh
```

如果你想先缩短验证时间，可以减少 episode 和 epoch：

```python
!NUM_EPISODES=8 MAX_STEPS=60 TRAIN_EPOCHS=5 bash scripts/run_bc_pipeline.sh
```

## 7. 查看结果

```python
import pandas as pd
pd.read_csv("results/mvp/mvp_results.csv")
```

MVP 结果文件：

```text
results/mvp/mvp_results.csv
results/mvp/mvp_results.json
results/mvp/*.mp4
```

BC 训练链结果文件：

```text
data/demos/pickcube_vtabr/
runs/bc_vtabr/bc_policy.pt
runs/bc_vtabr/bc_metrics.json
results/bc_vtabr/bc_eval/bc_eval_results.csv
results/bc_vtabr/fallback_eval/fallback_eval_results.csv
```

请额外关注结果里的 `env_backend` 列：

- `maniskill` 表示真 ManiSkill 环境已正常使用
- `maniskill_state_fallback` 表示只退到了 state 版本
- `mock` 表示 RGBD 初始化失败后，为了保留完整视觉主动感知训练链而使用了 mock fallback

## 8. 专门渲染机械臂视频

如果目标是生成 ManiSkill/SAPIEN 中 Panda 机械臂的 MP4 展示视频，请打开项目根目录下的：

```text
Robotics_Project_Colab_Render_MechArm.ipynb
```

Colab 里先选择：

```text
Runtime -> Change runtime type -> GPU
```

然后按 notebook 顺序运行。核心渲染命令是：

```python
!python main.py \
  --mode mvp \
  --scene pseudo_blur \
  --policy scripted \
  --use-active-probe \
  --obs-mode rgbd \
  --max-steps 120 \
  --seed 42 \
  --output-dir results/colab_render_active_probe
```

注意：渲染视频时不要加 `--no-video`。如果 Vulkan/RGBD 渲染失败，说明当前 Colab runtime 不适合渲染，请换一个 GPU runtime，或先用 state fallback 跑指标。
