FROM node:20-slim as frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend files
COPY frontend/package*.json ./
RUN npm install

COPY frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    g++ \
    git \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js for serving the frontend
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/.next ./.next
COPY --from=frontend-builder /app/frontend/public ./public
COPY --from=frontend-builder /app/frontend/package*.json ./

# Copy backend files
COPY requirements.txt .
COPY packages.txt .
COPY app.py .
COPY en.txt .
COPY frankenstein5k.md .
COPY gatsby5k.md .

# Create constraints file for pre-built wheels
RUN echo "spacy==3.7.2\n\
numpy==1.26.4\n\
Cython==0.29.37\n\
--prefer-binary" > constraints.txt

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Install frontend runtime dependencies
RUN npm install --production

# Expose port
EXPOSE 7860

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NODE_ENV=production

# Start both services using a shell script
COPY <<EOF ./start.sh
#!/bin/bash
# Start the Next.js frontend in the background
npm start &
# Start the Python backend
python app.py
EOF

RUN chmod +x ./start.sh

# Command to run both services
CMD ["./start.sh"] 