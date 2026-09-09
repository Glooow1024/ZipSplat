# Reconnect current and previous viewers; training runs independently on vllm1.
param([int[]]$Ports = @(18766, 18767))
$ErrorActionPreference = 'Stop'
foreach ($monitorPort in $Ports) {
$listening = Get-NetTCPConnection -LocalPort $monitorPort -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Start-Process -FilePath ssh.exe -ArgumentList @(
        '-N', '-L', "127.0.0.1:${monitorPort}:127.0.0.1:${monitorPort}",
        '-o', 'ExitOnForwardFailure=yes', '-o', 'ServerAliveInterval=30',
        '-o', 'ServerAliveCountMax=3', 'vllm1'
    ) -WindowStyle Hidden | Out-Null
}
Write-Output "Training monitor: http://127.0.0.1:${monitorPort}/"
}
