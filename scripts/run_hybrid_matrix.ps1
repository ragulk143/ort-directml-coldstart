
$outFile = "results\hybrid_crossover_raw.csv"

$header = "model,gpu_device,total_frames,trial,decision,ttfi_ms,switch_frame,bg_gpu_init_ms,cpu_phase_med_ms,gpu_phase_med_ms,overall_p95_ms,workload_wall_ms,process_wall_ms,cpu_estimate_ms,gpu_estimate_ms,pred_cpu_remaining_ms,pred_gpu_remaining_ms,handoff_overhead_ms"

if (Test-Path $outFile) {
    Write-Host "ERROR: Output file already exists. Refusing to overwrite it."
    exit 1
}

$header | Out-File -FilePath $outFile -Encoding utf8

$jobs = @()
$models = @(
    "models/squeezenet.onnx",
    "models/mobilenetv2.onnx"
)

$devices = @(0, 1)

foreach ($model in $models) {
    foreach ($device in $devices) {
        for ($trial = 1; $trial -le 10; $trial++) {
            $jobs += [PSCustomObject]@{
                Model = $model
                Device = $device
                Trial = $trial
            }
        }
    }
}

$shuffled = $jobs | Get-Random -Count $jobs.Count

$total = $shuffled.Count
$i = 0

foreach ($job in $shuffled) {
    $i++

    Write-Host "[$i/$total] $($job.Model) | device $($job.Device) | trial $($job.Trial)"

    $line = & .\hybrid.exe `
        --model $job.Model `
        --device $job.Device `
        --frames 800 `
        --trial $job.Trial

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Trial failed. Stopping."
        exit 1
    }

    $line | Select-Object -Last 1 |
        Out-File -FilePath $outFile -Append -Encoding utf8

    Start-Sleep -Seconds 2
}

Write-Host "DONE! Results saved to $outFile"
