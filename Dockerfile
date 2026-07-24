# FontFoundry × zi2zi-JiT worker image.
# Weights/styles/loras live on the RunPod Network Volume at /runpod-volume/zi2zi-jit
# so they are not baked into the image.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
WORKDIR /app

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends     git curl fonts-noto-cjk     && rm -rf /var/lib/apt/lists/*

# Python runtime dependencies (self-contained, do not rely on Network Volume site-packages)
COPY requirements.inference.txt .
RUN pip install --no-cache-dir -r requirements.inference.txt

# zi2zi-JiT 模型源码
RUN git clone --depth 1 https://github.com/kaonashi-tyc/zi2zi-JiT.git /app/zi2zi-jit

# 源字体/content font
RUN mkdir -p /app/fonts     && curl -fsSL -o /app/fonts/source.ttf        https://raw.githubusercontent.com/notofonts/noto-cjk/main/Sans/OTF/SimplifiedChinese/NotoSansCJKsc-Regular.otf

COPY handler.py .

# 只把镜像内的源码加入路径；pip 包已安装在镜像里，不依赖网络卷 site-packages
ENV PYTHONPATH=/app/zi2zi-jit:$PYTHONPATH
ENV ZI2ZI_WEIGHTS=/runpod-volume/zi2zi-jit
ENV ZI2ZI_SOURCE_FONT=/app/fonts/source.ttf
ENV ZI2ZI_VARIANT=B

CMD ["python3", "-u", "handler.py"]
