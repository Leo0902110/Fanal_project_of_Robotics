param(
    [string]$Python = "$env:USERPROFILE\workspace\.venv\Scripts\python.exe",
    [string]$ConfigJson = "diffusion_baseline\experiments\configs\grid_small.json"
)
$ErrorActionPreference = "Stop"
$projectRoot = $PSScriptRoot

$config = Get-Content $ConfigJson | ConvertFrom-Json
$results = @()
$i = 0
foreach ($exp in $config.experiments) {
    $i++
    $ws = $exp.warmup_steps
    $bs = $exp.batch_size
    $ds = $exp.num_diffusion_steps
    $amp = $exp.amp
    $ns = if ($exp.num_steps) { $exp.num_steps } else { 500 }
    Write-Host "  [$i] Running: warmup=$ws batch=$bs diffusion=$ds amp=$amp steps=$ns"
    $outFile = Join-Path $projectRoot "_exp$i.txt"
    $t0 = Get-Date
    & $Python diffusion_baseline\training\run_env_train.py `
        --device cuda --env_id PickCube-v1 `
        --warmup_steps $ws --batch_size $bs --num_diffusion_steps $ds --amp $amp `
        --num_steps $ns --save_every 9999 --use_tensorboard False --num_envs 1 --seed 0 `
        > $outFile 2>&1
    $exitCode = $LASTEXITCODE
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    $raw = Get-Content $outFile -Raw
    Remove-Item $outFile -Force -ErrorAction SilentlyContinue
    $sr = ""
    $rmean = ""
    $loss = ""
    if ($raw -match 'diagnosis_success_rate=([\d.e\-]+)') { $sr = $matches[1] }
    if ($raw -match 'diagnosis_return_min=([\d.e\-]+) mean=([\d.e\-]+) max=([\d.e\-]+)') { $rmean = $matches[2] }
    if ($raw -match 'final_loss=([\d.e\-]+)') { $loss = $matches[1] }
    $row = [PSCustomObject]@{
        run_id = $i - 1
        timestamp = (Get-Date -Format "yyyy-MM-ddTHH:mm:ss")
        exit_code = $exitCode
        warmup_steps = $ws
        batch_size = $bs
        num_diffusion_steps = $ds
        amp = $amp
        num_steps = $ns
        device = "cuda"
        env_id = "PickCube-v1"
        seed = 0
        success_rate = if ($sr) { [double]$sr } else { "" }
        return_mean = if ($rmean) { [double]$rmean } else { "" }
        return_min = ""
        return_max = ""
        final_loss = if ($loss) { [double]$loss } else { "" }
        duration_sec = [math]::Round($elapsed, 1)
    }
    $results += $row
    Write-Host "  [$i] Done: sr=$sr rmean=$rmean loss=$loss dur=$([math]::Round($elapsed, 1))s exit=$exitCode"
}
$csvPath = Join-Path $PSScriptRoot "diffusion_baseline\experiments\results.csv"
$results | Export-Csv -Path $csvPath -NoTypeInformation -Encoding UTF8
Write-Host "`nResults written to $csvPath"
Write-Host "Total experiments: $($results.Count)"
Get-Content $csvPath | Select-Object -First 4
