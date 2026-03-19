#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"

mkdir -p "${RAW_DIR}"

# Helper function to download a file using curl or wget
download_file() {
    local url="$1"
    local output="$2"

    if command -v curl >/dev/null 2>&1; then
        echo "Downloading with curl: $url"
        curl -L "$url" -o "$output"
    elif command -v wget >/dev/null 2>&1; then
        echo "Downloading with wget: $url"
        wget -O "$output" "$url"
    else
        echo "Error: Neither curl nor wget is installed. Please install one of them to proceed." >&2
        exit 1
    fi
}

# Download and unzip POI data
download_file "https://personal.ntu.edu.sg/gaocong/data/poidata.zip" "${RAW_DIR}/poidata.zip"
unzip -o "${RAW_DIR}/poidata.zip" -d "${RAW_DIR}"

# Download Gowalla check-in data
download_file "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz" "${RAW_DIR}/loc-gowalla_totalCheckins.txt.gz"