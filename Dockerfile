FROM python:3.12-slim

# XGBoost needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first so Docker can cache this layer when only code changes
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# Only what the service needs at runtime (train.py and the data stay out of the image)
COPY features.py predict.py model.bin ./

# Local default is 9696; cloud platforms (Cloud Run, Render) set PORT themselves
ENV PORT=9696
EXPOSE 9696

# "exec" makes gunicorn the main process so it receives stop signals correctly
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT} --workers 2 predict:app"]
