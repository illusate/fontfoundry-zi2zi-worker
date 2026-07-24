FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app/fonts \
    && cp /usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc /app/fonts/source.ttf 2>/dev/null \
    || cp /usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc /app/fonts/source.ttf 2>/dev/null \
    || echo "WARNING: built-in source font not found"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir git+https://ghfast.top/https://github.com/kaonashi-tyc/zi2zi-JiT.git 2>/dev/null \
    || pip install --no-cache-dir git+https://github.com/kaonashi-tyc/zi2zi-JiT.git

RUN git clone --depth 1 https://ghfast.top/https://github.com/kaonashi-tyc/zi2zi-JiT.git /app/zi2zi-jit 2>/dev/null \
    || git clone --depth 1 https://github.com/kaonashi-tyc/zi2zi-JiT.git /app/zi2zi-jit

COPY train_handler.py .

ENV ZI2ZI_WEIGHTS=/runpod-volume/zi2zi-jit
ENV ZI2ZI_SOURCE_FONT=/app/fonts/source.ttf
ENV ZI2ZI_VARIANT=B

CMD ["python3", "-u", "train_handler.py"]
