# CONFIGURATION
$MetadataPath   = "G:\deepfake_training_datasets\ASVspoof-2021\keys\DF\CM\trial_metadata.txt"
$SourceAudioDir = "G:\deepfake_training_datasets\ASVspoof-2021\ASVspoof2021_DF_eval\flac"
$OutputDir      = "G:\deepfake_training_datasets\ASVspoof-2021\ASVspoof2021_DF_eval\separated"
$SpoofDir       = Join-Path $OutputDir "spoofed"
$BonafideDir    = Join-Path $OutputDir "bonafide"
$MaxJobs        = 6

New-Item -ItemType Directory -Path $SpoofDir -Force | Out-Null
New-Item -ItemType Directory -Path $BonafideDir -Force | Out-Null

$lines = Get-Content $MetadataPath | Where-Object { $_ -match "DF_E_" }

$lines | ForEach-Object -Parallel -ScriptBlock {
    param (
        $line,
        $sourceAudioDir,
        $spoofDir,
        $bonafideDir
    )

    $parts = $line -split '\s+'
    if ($parts.Count -lt 6) { return }

    $filename = $parts[1]
    $label    = $parts[5].ToLower()
    $srcFile  = Join-Path $sourceAudioDir "$filename.flac"

    if (Test-Path $srcFile) {
        $dstDir  = if ($label -eq "spoof") { $spoofDir } else { $bonafideDir }
        $dstFile = Join-Path $dstDir "$filename.flac"

        try {
            Copy-Item -Path $srcFile -Destination $dstFile -Force
            Write-Host "Copied: $filename to $label"
        } catch {
            Write-Warning "Failed to copy $filename"
        }
    }
} -ArgumentList $SourceAudioDir, $SpoofDir, $BonafideDir -ThrottleLimit $MaxJobs

Write-Host "Separation complete."
