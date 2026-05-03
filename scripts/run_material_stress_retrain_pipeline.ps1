param(
    [int]$EpisodesPerProfile = 125,
    [int]$EvalEpisodes = 20,
    [string]$Device = "cuda",
    [int]$CollectSeedBase = 6100,
    [int]$EvalSeedBase = 7100,
    [double]$PseudoBlurSeverity = 1.0,
    [switch]$SkipCollect,
    [switch]$SkipTrain,
    [switch]$SkipEval
)

$ErrorActionPreference = "Stop"
$Python = "e:/Fanal_project_ROBTIES/.venv/Scripts/python.exe"

$DemoDir = "data/demos/pickcube_material_stress_d_assist_teacher_500"
$DRunDir = "runs/d_policy_material_stress_500_h256"
$TdpRunDir = "runs/tactile_dp_material_stress_residual_h1_h256"
$EvalRoot = "results/material_stress_eval"

$Profiles = @(
    @{ Name = "transparent"; CollectSeed = $CollectSeedBase + 0;    EvalSeed = $EvalSeedBase + 0;    Start = 0 },
    @{ Name = "dark";        CollectSeed = $CollectSeedBase + 1000; EvalSeed = $EvalSeedBase + 1000; Start = $EpisodesPerProfile },
    @{ Name = "reflective";  CollectSeed = $CollectSeedBase + 2000; EvalSeed = $EvalSeedBase + 2000; Start = 2 * $EpisodesPerProfile },
    @{ Name = "low_texture"; CollectSeed = $CollectSeedBase + 3000; EvalSeed = $EvalSeedBase + 3000; Start = 3 * $EpisodesPerProfile }
)

Write-Host "Material-stress retrain pipeline"
Write-Host "Demo dir: $DemoDir"
Write-Host "D run dir: $DRunDir"
Write-Host "TDP run dir: $TdpRunDir"
Write-Host "Eval root: $EvalRoot"
Write-Host "Episodes/profile: $EpisodesPerProfile; Eval episodes/profile: $EvalEpisodes"

if (-not $SkipCollect) {
    foreach ($Profile in $Profiles) {
        $Name = $Profile.Name
        Write-Host "Collecting material-stress demos for $Name"
        & $Python scripts/collect_d_assist_demos.py `
            --num-episodes $EpisodesPerProfile `
            --start-index $Profile.Start `
            --max-steps 50 `
            --seed $Profile.CollectSeed `
            --obs-mode rgbd `
            --scene material_object `
            --object-profile $Name `
            --material-visual-stress `
            --material-stress-profile auto `
            --pseudo-blur-severity $PseudoBlurSeverity `
            --output-dir $DemoDir
    }
}

if (-not $SkipTrain) {
    Write-Host "Training D policy on material-stress demos"
    & $Python scripts/train_d_policy.py `
        --demo-dir $DemoDir `
        --output-dir $DRunDir `
        --epochs 50 `
        --batch-size 512 `
        --hidden-dim 256 `
        --device $Device `
        --log-every 10

    Write-Host "Training residual tactile diffusion policy on material-stress demos"
    & $Python scripts/train_tactile_diffusion_policy.py `
        --demo-dir $DemoDir `
        --output-dir $TdpRunDir `
        --epochs 80 `
        --batch-size 512 `
        --hidden-dim 256 `
        --action-horizon 1 `
        --diffusion-steps 100 `
        --base-d-checkpoint "$DRunDir/d_policy.pt" `
        --device $Device `
        --log-every 10
}

if (-not $SkipEval) {
    foreach ($Profile in $Profiles) {
        $Name = $Profile.Name
        $EvalSeed = $Profile.EvalSeed
        $OutDir = "$EvalRoot/tdp_residual_${Name}_seed${EvalSeed}_${EvalEpisodes}eps"
        Write-Host "Evaluating material-stress profile $Name with seed $EvalSeed"
        & $Python scripts/evaluate_tactile_diffusion_policy.py `
            --checkpoint "$TdpRunDir/tactile_diffusion_policy.pt" `
            --base-d-checkpoint "$DRunDir/d_policy.pt" `
            --num-episodes $EvalEpisodes `
            --max-steps 50 `
            --seed $EvalSeed `
            --obs-mode rgbd `
            --control-mode pd_joint_pos `
            --joint-position-scale 0.36 `
            --sample-steps 50 `
            --replan-interval 1 `
            --init-noise-scale 0.0 `
            --residual-scale 0.05 `
            --condition-clip-sigma 2.0 `
            --scene material_object `
            --object-profile $Name `
            --material-visual-stress `
            --material-stress-profile auto `
            --pseudo-blur-severity $PseudoBlurSeverity `
            --device $Device `
            --output-dir $OutDir
    }

    & $Python scripts/summarize_material_stress_results.py --results-dir $EvalRoot
}
