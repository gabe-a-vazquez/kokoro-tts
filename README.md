---
title: Kokoro TTS
emoji: ❤️
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: true
license: apache-2.0
short_description: Next.js + FastAPI Text-to-Speech Application
---

# Kokoro TTS

A modern Text-to-Speech application built with Next.js and FastAPI, powered by the Kokoro TTS model.

## Features

- Multiple voice options with different personalities
- Adjustable speech speed
- GPU acceleration support (when available)
- Modern UI built with Next.js and shadcn/ui
- Real-time audio generation
- REST API built with FastAPI

## Architecture

- **Frontend**: Next.js application with TypeScript and shadcn/ui components
- **Backend**: FastAPI server with Kokoro TTS integration
- **Deployment**: Containerized with Docker for Hugging Face Spaces

## API Endpoints

- `GET /api/v1/voices` - Get available voice options
- `POST /api/v1/generate` - Generate speech from text
- `WS /api/v1/stream` - Stream audio generation

## Local Development

1. Start the backend:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
```

2. Start the frontend:

```bash
cd frontend
npm install
npm run dev
```

The application will be available at:

- Frontend: http://localhost:3000
- Backend: http://localhost:8001
