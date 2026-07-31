FROM python:3.11-slim

WORKDIR /app

# Linux libraries required by OpenCV/Ultralytics
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Only requirements first, so Docker caches this layer and doesn't
# reinstall everything on every code change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Now the rest of the code
COPY . .

# Replace the Git LFS pointer with the real YOLO model file
RUN python download_model.py

# Railway/most PaaS providers inject $PORT — Streamlit must bind to it,
# not to its default 8501, or the platform's proxy won't find it.
EXPOSE 8080
CMD ["sh", "-c", "streamlit run src/app.py --server.port=${PORT:-8080} --server.address=0.0.0.0"]
