FROM python:3.11-slim

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir torch==2.2.2+cu121 torchvision==0.17.2+cu121 --index-url https://download.pytorch.org/whl/cu121

COPY src ./src
COPY examples ./examples
COPY .env.example ./
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "src.chatbot_api:app", "--host", "0.0.0.0", "--port", "8000"]
