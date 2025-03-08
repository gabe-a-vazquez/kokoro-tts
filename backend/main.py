from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Kokoro TTS API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure base URL for Hugging Face Spaces
SPACE_ID = os.getenv('SPACE_ID', '')
BASE_URL = f"https://{SPACE_ID}.hf.space" if SPACE_ID else "http://localhost:7860"

app.include_router(router, prefix="/api/v1")

@app.get("/")
async def read_root():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "name": "Kokoro TTS API"
    }
