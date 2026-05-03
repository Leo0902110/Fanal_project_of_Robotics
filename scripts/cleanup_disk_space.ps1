param(
    [ValidateSet('Temp','RawDemos','OldRuns','OldResults')]
    [string[]]$Tier = @('RawDemos'),
    [switch]$Execute
)

$ErrorActionPreference = 'Stop'

$cleanupTargets = [ordered]@{
    Temp = @(
        '~$port(1).docx',
        '__pycache__',
        'src/__pycache__',
        'src/env/__pycache__'
    )
    RawDemos = @(
        'data/demos/pickcube_joint_scripted_500',
        'data/demos/pickcube_joint_scripted_success_oracle_geometry',
        'data/demos/pickcube_joint_scripted_success_oracle_geometry_raw',
        'data/demos/pickcube_joint_scripted_success_only',
        'data/demos/pickcube_rgbd_active_v1',
        'data/demos/pickcube_joint_scripted_smoke',
        'data/demos/pickcube_comprehensive_smoke'
    )
    OldRuns = @(
        'runs/bc_benchmark_cpu_5ep',
        'runs/bc_benchmark_gpu_5ep',
        'runs/bc_joint_scripted_500',
        'runs/bc_oracle_geometry_success_50ep_gpu',
        'runs/bc_success_only_50ep_gpu',
        'runs/bc_rgbd_active_v1',
        'runs/bc_rgbd_active_v1_norm',
        'runs/bc_rgbd_active_v2_norm',
        'runs/bc_comprehensive_smoke',
        'runs/d_policy_assist_teacher_50_gpu',
        'runs/d_policy_assist_teacher_smoke',
        'runs/tactile_dp_teacher_500_gpu_h256',
        'runs/tactile_dp_teacher_500_phase_h1_h256',
        'runs/tactile_dp_teacher_500_phase_h256',
        'runs/tactile_dp_teacher_500_phase_smoke',
        'runs/tactile_dp_teacher_500_smoke'
    )
    OldResults = @(
        'results/action_diagnostics',
        'results/bc_comprehensive_smoke',
        'results/bc_rgbd_active_v2_norm_eval',
        'results/bc_state_postfix_smoke',
        'results/c_rgbd_comparison_seed_410',
        'results/c_rgbd_comparison_seed_411',
        'results/c_rgbd_comparison_seed_412',
        'results/c_rgbd_comparison_v1',
        'results/debug_d_base_seed3000_12steps',
        'results/d_policy_assist_teacher_50_eval',
        'results/d_policy_assist_teacher_50_eval_scale_0.24',
        'results/d_policy_assist_teacher_50_eval_scale_0.3',
        'results/d_policy_assist_teacher_50_eval_scale_0.36',
        'results/d_policy_assist_teacher_50_eval_scale_0.42',
        'results/d_policy_assist_teacher_50_eval_scale_0.5',
        'results/fallback_comprehensive_smoke',
        'results/fallback_rgbd_active_v1_eval',
        'results/fallback_rgbd_c_smoke',
        'results/fallback_state_postfix_smoke',
        'results/grasp_assist_smoke',
        'results/main_rgbd_active_probe',
        'results/main_rgbd_d_policy_smoke',
        'results/main_rgbd_joint_scripted_compare',
        'results/main_rgbd_joint_scripted_smoke',
        'results/main_rgbd_joint_scripted_smoke_v2',
        'results/main_rgbd_joint_scripted_smoke_v3',
        'results/main_rgbd_sine_compare',
        'results/pseudo_blur_profile_smoke',
        'results/tactile_dp_teacher_500_eval_seed3000_10ep',
        'results/tactile_dp_teacher_500_phase_eval_seed3000_10ep',
        'results/tactile_dp_teacher_500_phase_eval_seed3000_fullsample_2ep',
        'results/tactile_dp_teacher_500_phase_eval_seed3000_noise02_5ep',
        'results/tactile_dp_teacher_500_phase_eval_seed3000_zero_init_5ep',
        'results/tactile_dp_teacher_500_phase_h1_eval_seed3000_noise02_5ep',
        'results/tactile_dp_teacher_500_phase_h1_eval_seed3000_zero_init_10ep',
        'results/tactile_dp_teacher_500_smoke_eval'
    )
}

function Get-PathSizeBytes {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    $item = Get-Item -LiteralPath $Path -Force
    if ($item.PSIsContainer) {
        $sum = (Get-ChildItem -LiteralPath $item.FullName -Recurse -File -Force -ErrorAction SilentlyContinue | Measure-Object Length -Sum).Sum
        if ($null -eq $sum) { return 0 }
        return [int64]$sum
    }
    return [int64]$item.Length
}

$selected = New-Object System.Collections.Generic.List[string]
foreach ($name in $Tier) {
    foreach ($target in $cleanupTargets[$name]) {
        if (-not $selected.Contains($target)) {
            $selected.Add($target)
        }
    }
}

$rows = foreach ($target in $selected) {
    if (Test-Path -LiteralPath $target) {
        $bytes = Get-PathSizeBytes $target
        [pscustomobject]@{
            Path = $target
            MB = [math]::Round($bytes / 1MB, 2)
            Exists = $true
        }
    } else {
        [pscustomobject]@{
            Path = $target
            MB = 0
            Exists = $false
        }
    }
}

$totalMB = [math]::Round((($rows | Measure-Object MB -Sum).Sum), 2)
Write-Host "Selected tiers: $($Tier -join ', ')"
Write-Host "Mode: $(if ($Execute) { 'DELETE' } else { 'DRY RUN' })"
Write-Host "Estimated reclaim: $totalMB MB"
$rows | Sort-Object MB -Descending | Format-Table -AutoSize

if (-not $Execute) {
    Write-Host "Dry run only. Add -Execute to delete these targets."
    exit 0
}

foreach ($row in $rows) {
    if ($row.Exists) {
        Write-Host "Deleting $($row.Path)"
        Remove-Item -LiteralPath $row.Path -Recurse -Force
    }
}

Write-Host "Cleanup complete. Reclaimed approximately $totalMB MB."