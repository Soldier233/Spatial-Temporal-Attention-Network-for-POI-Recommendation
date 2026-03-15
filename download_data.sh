#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${ROOT_DIR}/data/raw"

mkdir -p "${RAW_DIR}"

curl -L "https://personal.ntu.edu.sg/gaocong/data/poidata.zip" -o "${RAW_DIR}/poidata.zip"
unzip -o "${RAW_DIR}/poidata.zip" -d "${RAW_DIR}"

curl -L "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz" -o "${RAW_DIR}/loc-gowalla_totalCheckins.txt.gz"
