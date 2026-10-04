# CPU training/test image - used for the smoke test of the whole pipeline:
#   docker build -f tools/train.Dockerfile -t genai-train .
#   docker run --rm -e GENAI_PROFILE=smoke -e GENAI_SYNTHETIC=1 -e GENAI_WORKERS=0 -v ./outputs_smoke:/w/outputs genai-train
FROM python:3.11-slim
WORKDIR /w
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src ./src
COPY run_all.py .
CMD ["python", "-u", "run_all.py"]
