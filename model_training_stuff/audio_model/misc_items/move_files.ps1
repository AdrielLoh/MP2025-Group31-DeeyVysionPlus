$source = "G:\deepfake_training_datasets\Common-Voice-Bonafide-Only\clips"
$destination = "G:\deepfake_training_datasets\training_audio\training\real"
$files = Get-ChildItem -Path $source -File | Get-Random -Count 14593
$files | ForEach-Object { Move-Item $_.FullName -Destination $destination }