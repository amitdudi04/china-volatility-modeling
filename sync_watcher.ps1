$folder = "G:\Volatility Forecasting, Regime Dynamics & Risk Modeling using GARCH"
$offlineFolder = "G:\Volatility Forecasting, Regime Dynamics & Risk Modeling using GARCH\offline"

Write-Host "Initializing Real-Time Backup Synchronizer..."

# Clean up any existing watchers to avoid memory leaks if run multiple times
Get-EventSubscriber | Where-Object { $_.SourceObject -is [System.IO.FileSystemWatcher] } | Unregister-Event

$watcher = New-Object System.IO.FileSystemWatcher
$watcher.Path = $folder
$watcher.IncludeSubdirectories = $true
$watcher.EnableRaisingEvents = $true

$action = {
    try {
        $path = $Event.SourceEventArgs.FullPath
        # Exclude hidden system folders and the offline backup directory itself
        if ($path -match "\\\.git" -or $path -match "\\\.venv" -or $path -match "\\offline") { return }
        
        $relativePath = $path.Substring($folder.Length + 1)
        $destPath = Join-Path $offlineFolder $relativePath
        
        if (Test-Path $path -PathType Leaf) {
            $destDir = Split-Path $destPath
            if (-not (Test-Path $destDir)) { New-Item -ItemType Directory -Force -Path $destDir | Out-Null }
            # Micro-delay avoids access violation if IDE is still holding the file lock
            Start-Sleep -Milliseconds 150
            Copy-Item -Path $path -Destination $destPath -Force
            Write-Host "[SYNCED] $relativePath -> offline\"
        }
    } catch {}
}

Register-ObjectEvent $watcher 'Changed' -Action $action | Out-Null
Register-ObjectEvent $watcher 'Created' -Action $action | Out-Null
Register-ObjectEvent $watcher 'Renamed' -Action $action | Out-Null

Write-Host "Background Synchronizer Started. Live mirroring to \offline is active."
# Keep script alive in the background
while($true) { Start-Sleep -Seconds 3600 }
