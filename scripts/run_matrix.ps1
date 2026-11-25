$outFile = "results\coldstart_raw.csv"
"model,ep,param,trial,session_create_ms,first_inf_ms,steady_median_ms,steady_p95_ms,cold_gap_ms,checksum" | Out-File -FilePath $outFile -Encoding UTF8

$models = @("models/squeezenet.onnx", "models/mobilenetv2.onnx")
$configs = @(
    @{ ep = "cpu"; flag = "--threads"; val = 1 },
    @{ ep = "cpu"; flag = "--threads"; val = 4 },
    @{ ep = "dml"; flag = "--device";  val = 0 },
    @{ ep = "dml"; flag = "--device";  val = 1 }
)

$jobs = @()
foreach ($m in $models) {
    foreach ($c in $configs) {
        for ($t = 1; $t -le 20; $t++) {
            $jobs += [PSCustomObject]@{ Model = $m; EP = $c.ep; Flag = $c.flag; Val = $c.val; Trial = $t }
        }
    }
}
$shuffled = $jobs | Get-Random -Count $jobs.Count

$total = $shuffled.Count
$i = 0
foreach ($job in $shuffled) {
    $i++
    Write-Host "[$i / $total] $($job.Model) | $($job.EP) $($job.Val) | trial $($job.Trial)"
    $line = & .\bench.exe --model $job.Model --ep $job.EP $job.Flag $job.Val --runs 100 --trial $job.Trial
    $line | Out-File -FilePath $outFile -Append -Encoding UTF8
    Start-Sleep -Seconds 2
}
Write-Host "DONE! Results saved to $outFile"
