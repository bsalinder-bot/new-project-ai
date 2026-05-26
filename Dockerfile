FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --upgrade pip \
		&& python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Create a non-root user
RUN adduser --disabled-password --gecos '' spiuser || true
RUN chown -R spiuser:spiuser /app

USER spiuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
	CMD curl -f http://127.0.0.1:5000/ || exit 1

CMD ["gunicorn", "--config", "gunicorn.conf.py", "app:app"]
