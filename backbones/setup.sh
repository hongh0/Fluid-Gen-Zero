#!/bin/bash
# Clone the three video backbones at the commits Fluid-Gen-Zero was developed
# against and copy our files on top of them.
#
#   bash backbones/setup.sh [TARGET_DIR]        # default: third_party/
#
# Each backbone keeps its own conda environment and weights; follow its upstream
# README for both.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
DST=$(realpath -m "${1:-$HERE/../third_party}")
mkdir -p "$DST"

fetch () {   # fetch NAME URL COMMIT OVERLAY
  local name=$1 url=$2 commit=$3 overlay=$4
  if [ ! -d "$DST/$name/.git" ]; then
    git clone "$url" "$DST/$name"
  fi
  git -C "$DST/$name" checkout -q "$commit"
  cp -rv "$HERE/$overlay/." "$DST/$name/" | sed "s#^#  [$name] #"
}

fetch VACE     https://github.com/ali-vilab/VACE.git     48eb44f1c4be87cc65a98bff985a26976841e9f3 vace
fetch Tora     https://github.com/alibaba/Tora.git       2fc99cb8a4d7f52bae6d0169f033ffca69bf601c tora
fetch Wan-Move https://github.com/ali-vilab/Wan-Move.git 80c58a7d2ad175fa82a4d57f79f2a1415317dcfa wan_move
echo "backbones ready under $DST"
