# # Set your source and target folders here
# $sourceFolder = "G:\deepfake_training_datasets\DeeperForensics\manipulated_videos_part_08"
# $targetFolder = "G:\deepfake_training_datasets\DeeperForensics\fake"

# # Create the target folder if it doesn't exist
# if (-not (Test-Path $targetFolder)) {
#     New-Item -Path $targetFolder -ItemType Directory
# }

# # Function to generate a random filename
# function Get-RandomFilename {
#     return -join ((65..90) + (97..122) + (48..57) | Get-Random -Count 12 | ForEach-Object {[char]$_})
# }

# # Get all .mp4 files recursively
# Get-ChildItem -Path $sourceFolder -Recurse -Filter *.mp4 | ForEach-Object {
#     $randomName = Get-RandomFilename
#     $newPath = Join-Path $targetFolder ($randomName + ".mp4")

#     # Ensure no filename clash
#     while (Test-Path $newPath) {
#         $randomName = Get-RandomFilename
#         $newPath = Join-Path $targetFolder ($randomName + ".mp4")
#     }

#     Move-Item -Path $_.FullName -Destination $newPath
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



# --- MOVE SCRIPT 3 ---
# === CONFIG ===
$sourceFolder = "G:\deepfake_training_datasets\DeeperForensics\training\real"
$destinationFolder = "G:\deepfake_training_datasets\DeeperForensics\validation\real"
$numberOfFilesToMove = 2753  # Change this to how many random files you want to move

# === Ensure destination exists ===
if (-not (Test-Path $destinationFolder)) {
    New-Item -Path $destinationFolder -ItemType Directory
}

# === Get all files in source folder ===
$allFiles = Get-ChildItem -Path $sourceFolder -File

# === Shuffle and select N files ===
$selectedFiles = $allFiles | Get-Random -Count ([Math]::Min($numberOfFilesToMove, $allFiles.Count))

foreach ($file in $selectedFiles) {
    $destPath = Join-Path $destinationFolder $file.Name

    # If a file with the same name exists, append a number
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
    Write-Host "Moved: $($file.FullName) -> $destPath"
}
