# MacBook Air M4 local development setup

This project can run local smoke tests and MVP experiments on Apple Silicon,
but ManiSkill/SAPIEN need MoltenVK access. Commands that create ManiSkill
environments may need to run from a normal Terminal session, not a sandboxed
tool process.

## 1. System Vulkan dependencies

```bash
brew install molten-vk vulkan-loader vulkan-tools
```

Verify that MoltenVK can see the Apple GPU:

```bash
VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json \
vulkaninfo --summary
```

Expected signal: the device list includes `Apple M4` and `driverName = MoltenVK`.

## 2. Python 3.10 environment

The local environment used for verification is stored inside the repo:

```bash
conda create -p ./.conda-robotics python=3.10 -y
```

Install the minimum local development dependencies:

```bash
PYTHONNOUSERSITE=1 .conda-robotics/bin/python -m pip install \
  -r requirements.txt \
  torch torchvision torchaudio \
  imageio imageio-ffmpeg \
  mani_skill sapien pyyaml pin
```

## 3. Verified local commands

Use these environment variables for ManiSkill on macOS:

```bash
export VK_ICD_FILENAMES=/opt/homebrew/etc/vulkan/icd.d/MoltenVK_icd.json
export SAPIEN_VULKAN_LIBRARY_PATH=/opt/homebrew/lib/libvulkan.dylib
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/matplotlib
```

Run a quick local smoke test:

```bash
.conda-robotics/bin/python main.py \
  --mode smoke \
  --obs-mode state \
  --max-steps 30 \
  --no-video \
  --output-dir results/local_smoke_state
```

Run the local state MVP:

```bash
.conda-robotics/bin/python main.py \
  --mode mvp \
  --obs-mode state \
  --max-steps 120 \
  --no-video \
  --output-dir results/local_mvp_state
```

Run the local RGBD MVP:

```bash
.conda-robotics/bin/python main.py \
  --mode mvp \
  --obs-mode rgbd \
  --policy scripted \
  --max-steps 120 \
  --no-video \
  --output-dir results/local_mvp_rgbd
```

## 4. Local vs Colab split

Use the Mac for:

- Code development.
- Smoke tests.
- Small state/RGBD MVP runs.
- Debugging perception, tactile, and active probing interfaces.

Use Colab or a CUDA/Linux machine for:

- Long training jobs.
- Large data generation.
- Heavy rendering.
- Final experiment batches that need stable runtime.
