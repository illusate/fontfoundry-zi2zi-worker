# Minimal FontFoundry × zi2zi-JiT worker image.
# Runtime Python packages live on the RunPod Network Volume at /runpod-volume/zi2zi-jit/site-packages
# so they don't need to be baked into the image (keeps the image small enough to push from CN).
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps + 中文字体（作为源字体/content font）
RUN apt-get update && apt-get install -y --no-install-recommends \
    git fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# 确保源字体存在；优先用 Noto Sans CJK，找不到再让环境变量指定
RUN mkdir -p /app/fonts \
    && cp /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc /app/fonts/source.ttf 2>/dev/null \
    || cp /usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc /app/fonts/source.ttf 2>/dev/null \
    || echo "WARNING: built-in source font not found, worker will need ZI2ZI_SOURCE_FONT"

# zi2zi-JiT 模型源码（仓库较小；权重走网络卷，不打进镜像）
RUN git clone --depth 1 https://github.com/kaonashi-tyc/zi2zi-JiT.git /app/zi2zi-jit

COPY handler.py .

# 从 Network Volume 加载运行时 pip 包 + 源码 import
ENV PYTHONPATH=/runpod-volume/zi2zi-jit/site-packages:/runpod-volume/zi2zi-jit/zi2zi-jit-src:$PYTHONPATH
ENV ZI2ZI_WEIGHTS=/runpod-volume/zi2zi-jit
ENV ZI2ZI_SOURCE_FONT=/app/fonts/source.ttf
ENV ZI2ZI_VARIANT=B

CMD ["python3", "-u", "handler.py"]
