import os
from pathlib import Path
import torch
from kokoro import KModel, KPipeline
import numpy as np
from typing import AsyncGenerator
import spaces
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    def __init__(self):
        try:
            self.model_cache_dir = Path("./models")
            self.model_cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize models with proper device placement
            self.cuda_available = torch.cuda.is_available()
            logger.info(f"CUDA available: {self.cuda_available}")
            
            self.models = {}
            # Initialize CPU model
            logger.info("Initializing CPU model...")
            self.models[False] = KModel().to('cpu').eval()
            
            # Initialize GPU model if available
            if self.cuda_available:
                logger.info("Initializing GPU model...")
                self.models[True] = KModel().to('cuda').eval()
            
            # Initialize pipelines
            logger.info("Initializing pipelines...")
            self.pipelines = {
                lang_code: KPipeline(lang_code=lang_code, model=False) 
                for lang_code in 'ab'
            }
            
            # Configure voice pronunciations
            self.pipelines['a'].g2p.lexicon.golds['kokoro'] = 'kˈOkəɹO'
            self.pipelines['b'].g2p.lexicon.golds['kokoro'] = 'kˈQkəɹQ'
            logger.info("ModelManager initialization complete")
        except Exception as e:
            logger.error(f"Error initializing ModelManager: {str(e)}")
            raise

    @spaces.GPU(duration=30)
    def forward_gpu(self, ps, ref_s, speed):
        try:
            return self.models[True](ps, ref_s, speed)
        except Exception as e:
            logger.error(f"Error in GPU forward pass: {str(e)}")
            raise

    async def generate_speech(self, text: str, voice: str = "af_heart", speed: float = 1.0, use_gpu: bool = True):
        """Generate speech from text"""
        try:
            use_gpu = use_gpu and self.cuda_available
            logger.info(f"Generating speech with params: text='{text}', voice='{voice}', speed={speed}, use_gpu={use_gpu}")
            
            pipeline = self.pipelines[voice[0]]
            pack = pipeline.load_voice(voice)
            
            for _, ps, _ in pipeline(text, voice, speed):
                ref_s = pack[len(ps)-1]
                try:
                    if use_gpu:
                        logger.info("Using GPU for generation")
                        audio = self.forward_gpu(ps, ref_s, speed)
                    else:
                        logger.info("Using CPU for generation")
                        audio = self.models[False](ps, ref_s, speed)
                    return audio
                except Exception as e:
                    logger.error(f"Error during speech generation: {str(e)}")
                    if use_gpu:
                        logger.info("GPU generation failed, falling back to CPU")
                        audio = self.models[False](ps, ref_s, speed)
                        return audio
                    raise
        except Exception as e:
            logger.error(f"Error in generate_speech: {str(e)}")
            raise

    async def generate_speech_stream(self, text: str, voice: str = "af_heart", speed: float = 1.0, use_gpu: bool = True):
        """Stream speech generation"""
        use_gpu = use_gpu and self.cuda_available
        pipeline = self.pipelines[voice[0]]
        pack = pipeline.load_voice(voice)
        first = True
        
        for _, ps, _ in pipeline(text, voice, speed):
            ref_s = pack[len(ps)-1]
            try:
                if use_gpu:
                    audio = self.forward_gpu(ps, ref_s, speed)
                else:
                    audio = self.models[False](ps, ref_s, speed)
                yield 24000, audio.numpy()
                if first:
                    first = False
                    yield 24000, torch.zeros(1).numpy()
            except Exception as e:
                if use_gpu:
                    # Fallback to CPU
                    audio = self.models[False](ps, ref_s, speed)
                    yield 24000, audio.numpy()
                    if first:
                        first = False
                        yield 24000, torch.zeros(1).numpy()
                else:
                    raise e

    @staticmethod
    def get_device():
        return 'cuda' if torch.cuda.is_available() else 'cpu'
