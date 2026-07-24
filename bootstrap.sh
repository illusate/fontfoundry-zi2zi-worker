#!/bin/bash
set -e
export PYTHONPATH=/runpod-volume/zi2zi-jit/site-packages:/runpod-volume/zi2zi-jit/zi2zi-jit-src
export ZI2ZI_WEIGHTS=/runpod-volume/zi2zi-jit
export ZI2ZI_SOURCE_FONT=/runpod-volume/zi2zi-jit/fonts/source.ttf
export ZI2ZI_VARIANT=B

mkdir -p /runpod-volume/zi2zi-jit/zi2zi-jit-src /runpod-volume/zi2zi-jit/fonts

if [ ! -f /runpod-volume/zi2zi-jit/zi2zi-jit-src/denoiser.py ]; then
  git clone --depth 1 https://github.com/kaonashi-tyc/zi2zi-JiT.git /runpod-volume/zi2zi-jit/zi2zi-jit-src
fi

if [ ! -f /runpod-volume/zi2zi-jit/handler.py ]; then
  curl -fsSL -o /runpod-volume/zi2zi-jit/handler.py https://raw.githubusercontent.com/illusate/fontfoundry-zi2zi-worker/main/handler.py
fi

if [ ! -f /runpod-volume/zi2zi-jit/fonts/source.ttf ]; then
  curl -fsSL -o /runpod-volume/zi2zi-jit/fonts/source.ttf https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf
fi

python3 -u /runpod-volume/zi2zi-jit/handler.py
