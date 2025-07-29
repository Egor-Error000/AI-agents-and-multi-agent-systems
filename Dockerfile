FROM python:3.11-slim
WORKDIR /app
COPY agent.py .
COPY requirements.txt .
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -r requirements.txt
RUN mkdir -p /app/data
ENV DEEPSEEK_API_KEY=""
ENV WOLFRAM_APP_ID=""
ENV SERPAPI_KEY=""
ENV PYTHONUNBUFFERED=1
CMD ["python", "agent.py"]
