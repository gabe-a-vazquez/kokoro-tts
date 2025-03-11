FROM node:18-slim AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files first for better caching
COPY frontend/package*.json ./

# Clean install dependencies, including SWC
RUN npm cache clean --force && \
    npm install

# Copy frontend source code
COPY frontend/ .

# Verify the lib directory exists and files are copied
RUN test -d src/lib || mkdir -p src/lib

# Build frontend with verbose output
RUN NODE_ENV=production npm run build

# Use Python image for the backend
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV MODEL_CACHE_DIR=/app/models \
    NODE_ENV=production

# Copy backend requirements and install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ backend/

# Create model cache directory
RUN mkdir -p $MODEL_CACHE_DIR

# Copy built frontend from previous stage
COPY --from=frontend-builder /app/frontend/.next frontend/.next
COPY --from=frontend-builder /app/frontend/public frontend/public
COPY --from=frontend-builder /app/frontend/package.json frontend/package.json

# Install only production dependencies for frontend
WORKDIR /app/frontend
RUN npm install --omit=dev

# Create start script
WORKDIR /app
RUN echo '#!/bin/bash\n\
cd /app/frontend && NODE_ENV=production npm run start &\n\
cd /app/backend && uvicorn main:app --host 0.0.0.0 --port 7860\n\
' > start.sh

RUN chmod +x start.sh

# Expose port
EXPOSE 7860

# Start both services
CMD ["./start.sh"]
