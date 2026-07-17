FROM python:3.12-slim

# No .pyc files; unbuffered stdout so logs appear live in the cloud log stream.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install deps first: this layer is cached and only rebuilds when requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Run as a non-root user — cloud platforms and security reviews expect this.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Cloud hosts tell the app which port to listen on via $PORT; default 8000 locally.
ENV PORT=8000
EXPOSE 8000

# Production start command. No --reload (that's dev-only, and docker-compose
# overrides this command for local development).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
