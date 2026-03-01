# Paperclip Daemon — for testing/CI without a real GPU
# For production, install directly on the provider machine:
#   pip install paperclip-daemon
FROM python:3.12-slim

WORKDIR /app

# Install core deps (no GPU)
COPY requirements.txt .
RUN pip install --no-cache-dir httpx click rich

# Install the package
COPY . .
RUN pip install --no-cache-dir -e .

# Default: mock mode so no GPU is required
ENV PAPERCLIP_MOCK=1
ENV PAPERCLIP_API_URL=http://api:8000

ENTRYPOINT ["paperclip"]
CMD ["--help"]
