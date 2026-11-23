FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 KERAS_BACKEND=jax JAX_PLATFORM_NAME=cpu
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt && pip install --no-cache-dir 'gunicorn>=22,<24'
COPY . .
RUN useradd --system --create-home wce && mkdir -p /app/uploads && chown -R wce:wce /app/uploads
USER wce
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "120", "app:app"]
