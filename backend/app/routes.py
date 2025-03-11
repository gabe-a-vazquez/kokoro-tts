from fastapi import APIRouter, HTTPException, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
from utils.model_manager import ModelManager
from utils.resource_manager import ResourceManager
import numpy as np

router = APIRouter()
model_manager = ModelManager()
resource_manager = ResourceManager()

class TTSRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    speed: float = 1.0
    use_gpu: bool = True

class VoiceOption(BaseModel):
    id: str
    name: str
    emoji: str
    gender: str
    language: str

VOICE_CHOICES = {
    'af_heart': VoiceOption(id='af_heart', name='Heart', emoji='❤️', gender='Female', language='English'),
    'af_bella': VoiceOption(id='af_bella', name='Bella', emoji='🔥', gender='Female', language='English'),
    'af_nicole': VoiceOption(id='af_nicole', name='Nicole', emoji='🎧', gender='Female', language='English'),
    'af_aoede': VoiceOption(id='af_aoede', name='Aoede', emoji='', gender='Female', language='English'),
    'af_kore': VoiceOption(id='af_kore', name='Kore', emoji='', gender='Female', language='English'),
    'af_sarah': VoiceOption(id='af_sarah', name='Sarah', emoji='', gender='Female', language='English'),
    'af_nova': VoiceOption(id='af_nova', name='Nova', emoji='', gender='Female', language='English'),
    'af_sky': VoiceOption(id='af_sky', name='Sky', emoji='', gender='Female', language='English'),
    'af_alloy': VoiceOption(id='af_alloy', name='Alloy', emoji='', gender='Female', language='English'),
    'af_jessica': VoiceOption(id='af_jessica', name='Jessica', emoji='', gender='Female', language='English'),
    'af_river': VoiceOption(id='af_river', name='River', emoji='', gender='Female', language='English'),
}

@router.get("/voices")
async def get_voices() -> List[VoiceOption]:
    """Get available voice options"""
    return list(VOICE_CHOICES.values())

@router.post("/generate")
async def generate_speech(request: TTSRequest):
    """Generate speech from text"""
    try:
        resource_manager.check_resources()
        audio = await model_manager.generate_speech(
            text=request.text,
            voice=request.voice,
            speed=request.speed,
            use_gpu=request.use_gpu
        )
        # Convert to float32 and ensure values are between -1 and 1
        audio_normalized = audio.numpy().astype(np.float32)
        return {
            "sample_rate": 24000,
            "audio": audio_normalized.tolist()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    """Stream audio generation"""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            async for sample_rate, audio_chunk in model_manager.generate_speech_stream(
                text=data["text"],
                voice=data.get("voice", "af_heart"),
                speed=data.get("speed", 1.0),
                use_gpu=data.get("use_gpu", True)
            ):
                await websocket.send_json({
                    "sample_rate": sample_rate,
                    "audio": audio_chunk.tolist()
                })
    except Exception as e:
        await websocket.close(code=1000, reason=str(e))
