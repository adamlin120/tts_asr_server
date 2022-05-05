# -*- coding: utf-8 -*-
import base64
import logging
import os
import sys

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from transformers import pipeline

root = logging.getLogger()
root.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


class AsrRequest(BaseModel):
    audio: str = Field(example="base64 encoded audio string")


app = FastAPI()

asr_model_name_or_path = os.getenv("ASR_MODEL_NAME", "facebook/wav2vec2-base-960h")
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model=asr_model_name_or_path,
    device=0 if torch.cuda.is_available() else -1,
)


@app.get("/")
def root():
    return {"message": "Hello World. Please use /tts, /asr or check the docs on /docs"}


@app.post("/asr")
async def asr(request: AsrRequest):
    audio = request.audio
    audio_bytes = base64.b64decode(audio)
    result = asr_pipeline(audio_bytes)
    return result
