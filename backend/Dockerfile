FROM --platform=linux/arm64 python:3.9

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    cmake \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install basic build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install problematic dependencies first
RUN pip install --no-cache-dir \
    numpy==1.26.4 \
    cython==0.29.37 \
    blis==0.7.11 \
    spacy==3.7.2 \
    torch==2.2.2 \
    thinc==8.2.1

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .

# Install remaining Python packages
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"] 