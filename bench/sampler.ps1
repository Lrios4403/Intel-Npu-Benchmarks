# Samples package power (RAPL), total CPU load and, for the PIDs listed in
# $PidFile, every GPU/NPU engine's utilization, about once a second, as JSON lines.
param([string]$Out, [string]$PidFile)
$ErrorActionPreference = 'SilentlyContinue'
$base = @(
  '\Energy Meter(rapl_package0_pkg)\Power',
  '\Energy Meter(rapl_package0_pp0)\Power',
  '\Energy Meter(rapl_package0_pp1)\Power',
  '\Processor(_Total)\% Processor Time'
)
while ($true) {
  $paths = @($base)
  if (Test-Path $PidFile) {
    foreach ($p in ((Get-Content $PidFile -Raw) -split '[,\s]+' | Where-Object { $_ })) {
      $paths += "\GPU Engine(pid_${p}_*)\Utilization Percentage"
    }
  }
  $s = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1
  if ($s) {
    $o = [ordered]@{ t = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0; v = @{} }
    foreach ($c in $s.CounterSamples) { $o.v[$c.Path] = [math]::Round($c.CookedValue, 2) }
    ($o | ConvertTo-Json -Compress -Depth 4) | Add-Content -Path $Out -Encoding utf8
  }
}
