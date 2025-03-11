# syntax=docker/dockerfile:1.4

FROM node:18 AS base

# Install dependencies only when needed
FROM base AS deps
WORKDIR /app/frontend

# Install dependencies based on the preferred package manager
COPY frontend/package.json ./
RUN npm install --force

# Rebuild the source code only when needed
FROM base AS builder
WORKDIR /app/frontend
COPY --from=deps /app/frontend/node_modules ./node_modules
COPY frontend/ .

# Disable telemetry during build
ENV NEXT_TELEMETRY_DISABLED 1
ENV NODE_ENV production

RUN npm run build

# Use Python image for the backend
FROM python:3.11-slim AS runner

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    nodejs \
    npm \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN addgroup --system --gid 1001 appgroup && \
    adduser --system --uid 1001 appuser

# Set environment variables
ENV MODEL_CACHE_DIR=/app/models \
    NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ backend/

# Create model cache directory with proper permissions
RUN mkdir -p $MODEL_CACHE_DIR && \
    chown -R appuser:appgroup $MODEL_CACHE_DIR

# Copy built frontend from builder
COPY --from=builder --chown=appuser:appgroup /app/frontend/.next frontend/.next
COPY --from=builder --chown=appuser:appgroup /app/frontend/public frontend/public
COPY --from=builder --chown=appuser:appgroup /app/frontend/package.json frontend/package.json

# Install frontend production dependencies
WORKDIR /app/frontend
RUN npm install --omit=dev

# Create start script
WORKDIR /app
RUN echo '#!/bin/bash\n\
cd /app/frontend && NODE_ENV=production npm run start -- -p 3000 &\n\
cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 7860\n\
' > start.sh && \
    chmod +x start.sh && \
    chown appuser:appgroup start.sh

# Switch to non-root user
USER appuser

# Expose ports
EXPOSE 7860 3000

# Start both services
CMD ["./start.sh"]
