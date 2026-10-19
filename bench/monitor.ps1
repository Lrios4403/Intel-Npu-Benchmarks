# System-wide resource monitor for the benchmark runs. Appends one JSON line per
# sample (~every 2 s): CPU, RAM, RAPL package power, every GPU/NPU engine's
# utilization (summed per adapter LUID and engine type), GPU/NPU adapter memory,
# thermal zones, disk throughput, benchmark process memory, the RTX via
# nvidia-smi, and the active phase (read from ovms.exe's command line).
param([string]$Out, [int]$IntervalMs = 2000)
$ErrorActionPreference = 'SilentlyContinue'
$paths = @(
  '\Processor Information(_Total)\% Processor Utility',
  '\Memory\Available MBytes',
  '\Memory\Committed Bytes',
  '\Energy Meter(rapl_package0_pkg)\Power',
  '\Energy Meter(rapl_package0_pp0)\Power',
  '\Energy Meter(rapl_package0_pp1)\Power',
  # Filtered by engine type: \GPU Engine(*) has 1,100+ instances and takes ~12 s to read
  '\GPU Engine(*engtype_neural)\Utilization Percentage',
  '\GPU Engine(*engtype_compute*)\Utilization Percentage',
  '\GPU Engine(*engtype_3d*)\Utilization Percentage',
  '\GPU Adapter Memory(*)\Dedicated Usage',
  '\GPU Adapter Memory(*)\Shared Usage',
  '\Thermal Zone Information(*)\High Precision Temperature',
  '\PhysicalDisk(_Total)\Disk Bytes/sec'
)
# Adapter LUIDs on this laptop (from OpenVINO's DEVICE_LUID)
$names = @{ '0x0001bb2a' = 'npu'; '0x00019fd4' = 'igpu'; '0x0001a65a' = 'rtx' }
function Adapter($luid) { if ($names.ContainsKey($luid)) { $names[$luid] } else { $luid } }
while ($true) {
  $started = [DateTime]::UtcNow
  $s = Get-Counter -Counter $paths -SampleInterval 1 -MaxSamples 1
  $values = @{}; $engines = @{}; $adapterMem = @{}; $temps = @{}
  foreach ($c in $s.CounterSamples) {
    $p = ($c.Path.ToLower() -replace '^\\\\[^\\]+', '')
    if ($p -match 'gpu engine\(pid_\d+_luid_0x[0-9a-f]+_(0x[0-9a-f]+)_phys_\d+_eng_\d+_engtype_([^)]*)\)') {
      $type = ($matches[2] -replace '#\d+$', ''); if (-not $type) { $type = 'other' }
      $key = "$(Adapter $matches[1])|$type"; $engines[$key] = [double]$engines[$key] + $c.CookedValue
    } elseif ($p -match 'gpu adapter memory\(luid_0x[0-9a-f]+_(0x[0-9a-f]+)_phys_\d+\)\\(dedicated|shared) usage') {
      $adapterMem["$(Adapter $matches[1])|$($matches[2])"] = [math]::Round($c.CookedValue / 1GB, 2)
    } elseif ($p -match 'thermal zone information\((.+)\)') {
      $temps[$matches[1]] = [math]::Round($c.CookedValue / 10 - 273.15, 1)  # tenths of a kelvin -> °C
    } else {
      $values[$p] = [math]::Round($c.CookedValue, 2)
    }
  }
  $eng = @{}; foreach ($k in $engines.Keys) { if ($engines[$k] -gt 0.5) { $eng[$k] = [math]::Round($engines[$k], 1) } }
  $procs = @{}
  foreach ($n in 'ovms', 'ollama', 'python') {
    $ps = Get-Process $n -ErrorAction SilentlyContinue
    if ($ps) { $procs[$n] = [math]::Round((($ps | Measure-Object WorkingSet64 -Sum).Sum) / 1GB, 2) }
  }
  $phase = 'idle'
  $ovms = Get-CimInstance Win32_Process -Filter "Name='ovms.exe'" | Select-Object -First 1
  if ($ovms) {
    $phase = 'ovms'
    if ($ovms.CommandLine -match '--target_device (\S+)') { $phase += "-$($matches[1])" }
    if ($ovms.CommandLine -match '--enable_prefix_caching (\S+)') { $phase += "-prefix-$($matches[1])" }
  } elseif ($procs.ContainsKey('ollama') -and $procs['ollama'] -gt 2) { $phase = 'ollama' }
  $nv = ((& nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu --format=csv,noheader,nounits) -split ',') | ForEach-Object { $_.Trim() }
  $o = [ordered]@{
    t = [DateTimeOffset]::Now.ToUnixTimeMilliseconds() / 1000.0; phase = $phase; v = $values; eng = $eng;
    adapter_mem_gb = $adapterMem; temps_c = $temps; procs_ws_gb = $procs;
    rtx = @{ power_w = $nv[0]; util_pct = $nv[1]; vram_mb = $nv[2]; temp_c = $nv[3] };
    c_free_gb = [math]::Round((Get-PSDrive C).Free / 1GB, 1)
  }
  ($o | ConvertTo-Json -Compress -Depth 5) | Add-Content -Path $Out -Encoding utf8
  $wait = $IntervalMs - ([DateTime]::UtcNow - $started).TotalMilliseconds
  if ($wait -gt 0) { Start-Sleep -Milliseconds $wait }
}
