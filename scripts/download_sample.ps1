param(
  [string]$Url = 'https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4',
  [string]$Dest = 'assets\media\sample_4k.mp4'
)
New-Item -ItemType Directory -Force -Path (Split-Path $Dest) | Out-Null
Write-Host "Downloading $Url to $Dest..."
try {
  Invoke-WebRequest -Uri $Url -OutFile $Dest -UseBasicParsing
  Write-Host "Done. Saved to $Dest"
} catch {
  Write-Error "Download failed: $_"
}
