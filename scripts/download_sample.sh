#!/usr/bin/env bash
# Simple downloader for a sample 4K drone video
# Usage: ./download_sample.sh [URL] [DEST]
URL=${1:-https://cdn.pixabay.com/video/2024/03/10/203678-922748476_large.mp4}
DEST=${2:-assets/media/sample_4k.mp4}
mkdir -p "$(dirname "$DEST")"
echo "Downloading $URL to $DEST..."
if command -v curl >/dev/null 2>&1; then
  curl -L --progress-bar -o "$DEST" "$URL"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$DEST" "$URL"
else
  echo "Please install curl or wget, or provide the file manually." >&2
  exit 2
fi
echo "Done. Saved to $DEST"
