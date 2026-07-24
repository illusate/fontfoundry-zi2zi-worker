# FontFoundry LoRA fine-tuning worker for zi2zi-JiT
# Built on top of the inference image to reuse the base PyTorch/cuda environment,
# zi2zi-JiT source, and source font.
FROM illusate/fontfoundry-zi2zi:latest

ENV PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.train.txt .
RUN pip install --no-cache-dir -r requirements.train.txt

COPY train_handler.py .

ENV PYTHONPATH=/app/zi2zi-jit:$PYTHONPATH
ENV ZI2ZI_WEIGHTS=/runpod-volume/zi2zi-jit
ENV ZI2ZI_SOURCE_FONT=/app/fonts/source.ttf
ENV ZI2ZI_VARIANT=B

CMD ["python3", "-u", "train_handler.py"]
