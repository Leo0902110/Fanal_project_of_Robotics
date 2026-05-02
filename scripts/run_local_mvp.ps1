$ErrorActionPreference = "Stop"

$Seed = 42
$MaxSteps = 120
$RootDir = "results/local_mvp_rgbd"
$Python = ".\.venv\Scripts\python.exe"

New-Item -ItemType Directory -Force -Path $RootDir | Out-Null

& $Python main.py `
  --mode mvp `
  --scene clean `
  --policy scripted `
  --obs-mode rgbd `
  --max-steps $MaxSteps `
  --seed $Seed `
  --no-video `
  --output-dir "$RootDir/clean"

& $Python main.py `
  --mode mvp `
  --scene pseudo_blur `
  --policy scripted `
  --obs-mode rgbd `
  --max-steps $MaxSteps `
  --seed $Seed `
  --no-video `
  --output-dir "$RootDir/pseudo_blur"

& $Python main.py `
  --mode mvp `
  --scene pseudo_blur `
  --policy scripted `
  --use-active-probe `
  --obs-mode rgbd `
  --max-steps $MaxSteps `
  --seed $Seed `
  --no-video `
  --output-dir "$RootDir/active_probe"

& $Python scripts/plot_results.py --results-dir $RootDir
