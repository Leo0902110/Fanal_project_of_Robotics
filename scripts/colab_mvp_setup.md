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

## 6. 查看结果

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
