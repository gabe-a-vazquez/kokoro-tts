FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    cmake \
    pkg-config \
    nginx \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip and install wheel
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install backend requirements
COPY backend/requirements.txt .
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Set up frontend
WORKDIR /app/frontend

# Install frontend dependencies
COPY frontend/package*.json ./
RUN npm install

# Copy all frontend files (except those in .dockerignore)
COPY frontend/ ./

# Set build environment
ENV NEXT_PUBLIC_API_URL=http://localhost:7860 \
    NEXT_TELEMETRY_DISABLED=1 \
    NODE_ENV=production \
    DISABLE_ESLINT_PLUGIN=true

# Build frontend
RUN npm run build

# Set up backend and supervisor
WORKDIR /app
COPY backend/ ./backend/
COPY supervisord.conf /etc/supervisor/conf.d/
COPY nginx.conf /etc/nginx/nginx.conf
COPY docker-entrypoint.sh /

RUN pip install supervisor && \
    mkdir -p /var/log/supervisor && \
    chmod +x /docker-entrypoint.sh

# Expose only the Nginx port
EXPOSE 7860

ENTRYPOINT ["/docker-entrypoint.sh"] 