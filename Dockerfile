FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    libpq-dev \
    curl \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir pip-tools
RUN pip install --upgrade pip
RUN pip-compile requirements.in && pip install -r requirements.txt 
CMD ["/bin/bash"]
