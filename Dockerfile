FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/data \
    HF_HOME=/data/hf \
    PORT=8000

# CPU-only PyTorch keeps the image small. For a GPU host, build from a CUDA base
# image and drop the --index-url.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY server.py .

# Runs as uid 1000, which is also what Hugging Face Spaces expects
RUN useradd --uid 1000 --create-home app && mkdir -p /data && chown app:app /data
USER app
VOLUME ["/data"]
EXPOSE 8000

# The first start downloads about 1.3 GB of weights per checkpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",\"8000\")}/healthz',timeout=4)"

CMD ["python", "server.py"]
