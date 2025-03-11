from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import router
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Kokoro TTS API")

# Configure CORS
SPACE_ID = os.getenv('SPACE_ID', '')
BASE_URL = f"https://{SPACE_ID}.hf.space" if SPACE_ID else "http://localhost:7860"

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes without prefix - prefix is handled by uvicorn root_path
app.include_router(router)

@app.get("/")
async def read_root():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "name": "Kokoro TTS API"
    }
