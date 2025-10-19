FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

COPY pyproject.toml requirements.txt* .
RUN uv pip install --system -r requirements.txt

COPY app/ ./app/
COPY model/ ./model/

EXPOSE 8000

CMD ["python", "app/main.py"]