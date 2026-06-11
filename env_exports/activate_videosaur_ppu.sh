#!/usr/bin/env bash
# Source this file from the repository root:
#   source env_exports/activate_videosaur_ppu.sh

set -euo pipefail

_videosaur_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

source "${_videosaur_root}/.venv/bin/activate"

export PYTHONPATH="${_videosaur_root}${PYTHONPATH:+:${PYTHONPATH}}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/videosaur_mpl_config}"
export PIP_CONSTRAINT="${_videosaur_root}/env_exports/videosaur_ppu_constraints.txt"

mkdir -p "${MPLCONFIGDIR}"

echo "Activated VideoSAUR PPU environment at ${_videosaur_root}/.venv"
