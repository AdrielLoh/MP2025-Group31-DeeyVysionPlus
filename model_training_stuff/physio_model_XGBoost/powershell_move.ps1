# === CONFIG ===
# $sourceFolder = "E:\deepfake_training_datasets\Common-Voice-Bonafide-Only\clips"
# $destinationFolder = "E:\deepfake_training_datasets\VoiceWukong\for-training\real"
# $prefix = ""  # <- Your custom prefix here

# # === Ensure destination exists ===
# if (-not (Test-Path $destinationFolder)) {
#     New-Item -Path $destinationFolder -ItemType Directory
# }

# # === Get .mp4 files recursively from the source folder
# $filesToMove = Get-ChildItem -Path $sourceFolder -Recurse -File | Where-Object { $_.Extension -eq ".wav" }

# foreach ($file in $filesToMove) {
#     $newFileName = $prefix + $file.Name
#     $destPath = Join-Path $destinationFolder $newFileName

#     # If file exists, append _1, _2, etc. to avoid overwrite
#     if (Test-Path $destPath) {
#         $base = [System.IO.Path]::GetFileNameWithoutExtension($newFileName)
#         $ext = $file.Extension
#         $count = 1
#         do {
#             $newName = "$base" + "_$count$ext"
#             $destPath = Join-Path $destinationFolder $newName
#             $count++
#         } while (Test-Path $destPath)
#     }

#     Move-Item -Path $file.FullName -Destination $destPath
#     Write-Host "Moved: $($file.FullName) -> $destPath"
# }


# === CONFIG ===
# $targetFolder = "G:\deepfake_training_datasets\Physio_Model\TRAINING\real"
# $oldPrefix = "deeperforensics_"
# $newPrefix = "dfdc_"

# # === Get all files starting with the old prefix
# $filesToRename = Get-ChildItem -Path $targetFolder -File | Where-Object { $_.Name -like "$oldPrefix*" }

# foreach ($file in $filesToRename) {
#     $newName = $file.Name -replace "^$oldPrefix", $newPrefix
#     $newPath = Join-Path $targetFolder $newName

#     # Ensure no overwrite
#     if (Test-Path $newPath) {
#         Write-Warning "Skipping (target already exists): $newPath"
#         continue
#     }

#     Rename-Item -Path $file.FullName -NewName $newName
#     Write-Host "Renamed: $($file.Name) -> $newName"
# }




# --- MOVE SCRIPT 2 ---
# # Set source and destination folders
# $sourceFolder = "G:\deepfake_training_datasets\DeeperForensics\training\fake"
# $destinationFolder = "G:\deepfake_training_datasets\DeeperForensics\legacy_files\potential_multiple_faces_FAKE-ONLY"

# # Create destination folder if it doesn't exist
# if (-not (Test-Path $destinationFolder)) {
#     New-Item -Path $destinationFolder -ItemType Directory
# }

# # Process files recursively
# Get-ChildItem -Path $sourceFolder -Recurse -File | ForEach-Object {
#     $baseNameLength = $_.BaseName.Length

#     if ($baseNameLength -ne 12) {
#         $destinationPath = Join-Path $destinationFolder $_.Name

#         # Ensure no overwrite
#         if (Test-Path $destinationPath) {
#             $timestamp = Get-Date -Format "yyyyMMddHHmmss"
#             $destinationPath = Join-Path $destinationFolder ("$($_.BaseName)_$timestamp$($_.Extension)")
#         }

#         Move-Item -Path $_.FullName -Destination $destinationPath
#         Write-Host "Moved: $($_.FullName) -> $destinationPath"
#     }
# }



# # --- MOVE SCRIPT 3 ---
# # === CONFIG ===
# $sourceFolder = "G:\deepfake_training_datasets\Physio_Model\TRAINING\fake"
# $destinationFolder = "G:\deepfake_training_datasets\Physio_Model\VALIDATION\fake"
# $numberOfFilesToMove = 5239

# # === Ensure destination exists ===
# if (-not (Test-Path $destinationFolder)) {
#     New-Item -Path $destinationFolder -ItemType Directory
# }

# # === Function to compute SHA256 hash of a file ===
# function Get-FileHashSHA256($filePath) {
#     return (Get-FileHash -Path $filePath -Algorithm SHA256).Hash
# }

# # === Collect hashes of existing destination files ===
# Write-Host "Hashing existing destination files..."
# $existingHashes = @{}
# Get-ChildItem -Path $destinationFolder -File | ForEach-Object {
#     try {
#         $hash = Get-FileHashSHA256 $_.FullName
#         $existingHashes[$hash] = $true
#     } catch {
#         Write-Warning "Failed to hash: $($_.FullName)"
#     }
# }

# # === Get all files in source folder ===
# $allFiles = Get-ChildItem -Path $sourceFolder -File

# # === Shuffle and select N files ===
# $selectedFiles = $allFiles | Get-Random -Count ([Math]::Min($numberOfFilesToMove, $allFiles.Count))

# $moveCount = 0
# foreach ($file in $selectedFiles) {
#     try {
#         $hash = Get-FileHashSHA256 $file.FullName

#         if ($existingHashes.ContainsKey($hash)) {
#             Write-Host "Skipping (duplicate hash): $($file.Name)"
#             continue
#         }

#         # Prepare destination path
#         $destPath = Join-Path $destinationFolder $file.Name

#         if (Test-Path $destPath) {
#             $base = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
#             $ext = $file.Extension
#             $count = 1
#             do {
#                 $newName = "$base" + "_$count$ext"
#                 $destPath = Join-Path $destinationFolder $newName
#                 $count++
#             } while (Test-Path $destPath)
#         }

#         Move-Item -Path $file.FullName -Destination $destPath
#         Write-Host "Moved: $($file.Name) -> $destPath"

#         # Store the new hash
#         $existingHashes[$hash] = $true
#         $moveCount++
#     } catch {
#         Write-Warning "Error processing $($file.FullName): $_"
#     }

#     # Stop if we've reached the target count
#     if ($moveCount -ge $numberOfFilesToMove) {
#         break
#     }
# }

# Write-Host "`nDone. Total files moved: $moveCount"



# === MOVE SCRIPT 4 ===
# === CONFIG ===
$destinationFolder = "E:\deepfake_training_datasets\VoiceWukong\fake"
$sourceFolder = "E:\deepfake_training_datasets\VoiceWukong\for-training\fake"
$numberOfFilesToMove = 70000

# === Ensure destination exists ===
if (-not (Test-Path $destinationFolder)) {
    New-Item -Path $destinationFolder -ItemType Directory
}

# === Get all files in source folder
$allFiles = Get-ChildItem -Path $sourceFolder -File

# === Shuffle and select N files
$selectedFiles = $allFiles | Get-Random -Count ([Math]::Min($numberOfFilesToMove, $allFiles.Count))

$moveCount = 0
foreach ($file in $selectedFiles) {
    try {
        # Prepare destination path
        $destPath = Join-Path $destinationFolder $file.Name

        # If a file with the same name exists, append _1, _2, etc.
        if (Test-Path $destPath) {
            $base = [System.IO.Path]::GetFileNameWithoutExtension($file.Name)
            $ext = $file.Extension
            $count = 1
            do {
                $newName = "$base" + "_$count$ext"
                $destPath = Join-Path $destinationFolder $newName
                $count++
            } while (Test-Path $destPath)
        }

        Move-Item -Path $file.FullName -Destination $destPath
        Write-Host "Moved: $($file.Name) -> $destPath"
        $moveCount++
    } catch {
        Write-Warning "Error moving $($file.FullName): $_"
    }

    # Stop if target count reached
    if ($moveCount -ge $numberOfFilesToMove) {
        break
    }
}

Write-Host "`nDone. Total files moved: $moveCount"



# ========================================================
# $sourceFolder = "G:\deepfake_training_datasets\Physio_Model\TRAINING\real-temp"
# $destinationFolder = "G:\deepfake_training_datasets\Physio_Model\TRAINING\real-non-frontal"

# # Ensure the destination exists
# if (-not (Test-Path $destinationFolder)) {
#     New-Item -Path $destinationFolder -ItemType Directory
# }

# # Get all files ending with '_camera_left.mp4'
# $files = Get-ChildItem -Path $sourceFolder -Filter "*_camera_left.mp4" -File

# foreach ($file in $files) {
#     $destPath = Join-Path $destinationFolder $file.Name

#     # If file does not exist at destination, move it
#     if (-not (Test-Path $destPath)) {
#         Move-Item -Path $file.FullName -Destination $destPath
#         Write-Host "Moved: $($file.Name)"
#     } else {
#         Write-Warning "Skipped (already exists): $($file.Name)"
#     }
# }


# ===========================================================================================
# # Set the target directory
# $TargetDir = "G:\deepfake_training_datasets\Physio_Model\TRAINING\real"

# # File types to check
# $VideoExts = "*.mp4","*.avi","*.mov","*.mkv","*.flv","*.wmv"

# # Threshold duration in seconds (1 min 50 sec = 110 sec)
# $DurationThreshold = 110

# # Loop over each filetype
# foreach ($ext in $VideoExts) {
#     # Get all videos (add -Recurse if you want subfolders too)
#     Get-ChildItem -Path $TargetDir -Filter $ext | ForEach-Object {
#         $file = $_.FullName

#         # Get video duration using ffprobe (from ffmpeg suite, must be installed)
#         $ffprobeResult = & ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file"
#         $duration = [double]$ffprobeResult

#         if ($duration -ge $DurationThreshold) {
#             Write-Host "Deleting: $file (Duration: $([math]::Round($duration,2)) sec)"
#             Remove-Item "$file"
#         }
#     }
# }
