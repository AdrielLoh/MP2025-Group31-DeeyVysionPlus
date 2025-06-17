# Set the folder to scan
$folder = "G:\deepfake_training_datasets\DeeperForensics\training\fake"

# Hash algorithm (e.g., SHA256, SHA1, MD5)
$hashAlgorithm = "SHA256"

# Create a hashtable to store file hashes
$hashTable = @{}

# Recursively get all files
Get-ChildItem -Path $folder -Recurse -File | ForEach-Object {
    try {
        $hash = Get-FileHash -Path $_.FullName -Algorithm $hashAlgorithm
        if ($hashTable.ContainsKey($hash.Hash)) {
            $hashTable[$hash.Hash] += ,$_.FullName
        } else {
            $hashTable[$hash.Hash] = @($_.FullName)
        }
    } catch {
        Write-Warning "Failed to hash file: $_.FullName"
    }
}

# Output duplicates
Write-Host "`n--- Duplicate Files Found ---`n"
$duplicates = $hashTable.GetEnumerator() | Where-Object { $_.Value.Count -gt 1 }

if ($duplicates) {
    foreach ($entry in $duplicates) {
        Write-Host "Hash: $($entry.Key)"
        $entry.Value | ForEach-Object { Write-Host "`t$_" }
        Write-Host ""
    }
} else {
    Write-Host "No duplicate files found."
}
