#!/usr/bin/env python3

"""
vibevoice_bridge_client.py

Client for the VibeVoice Bridge (port 8094) which provides:
- REST TTS: POST /tts -> returns audio
- REST STT: POST /stt -> returns transcription (uses Ollama Whisper)
- WebSocket TTS streaming: WS /stream -> streams audio chunks

This provides an alternative TTS path through the Bridge service, which
can also handle STT via Ollama Whisper, creating a complete voice loop
through the NovaMaster stack.
"""

import json
import logging
import requests
from typing import Optional


class VibeVoiceBridgeClient:
    """Client for VibeVoice Bridge REST/WS API."""

    def __init__(self, bridge_url: str = "http://127.0.0.1:8094"):
        self.bridge_url = bridge_url.rstrip('/')

    def text_to_speech(self, text: str, voice: str = "en-Carter_man") -> Optional[bytes]:
        """Synthesize speech via the Bridge REST TTS endpoint.

        Returns:
            Audio bytes (WAV format) or None on failure.
        """
        try:
            resp = requests.post(
                f"{self.bridge_url}/tts",
                json={"text": text, "voice": voice},
                timeout=10
            )
            if resp.status_code == 200:
                content_type = resp.headers.get('content-type', '')
                if 'audio' in content_type or 'octet-stream' in content_type:
                    return resp.content
                # If JSON response, extract audio data
                if 'json' in content_type:
                    data = resp.json()
                    if 'audio' in data:
                        import base64
                        return base64.b64decode(data['audio'])
            logging.error(f"Bridge TTS error: status {resp.status_code}")
            return None
        except Exception as e:
            logging.error(f"Bridge TTS request failed: {e}")
            return None

    def speech_to_text(self, audio_data: bytes, language: str = "en") -> Optional[str]:
        """Transcribe audio via the Bridge STT endpoint (Ollama Whisper).

        Args:
            audio_data: WAV audio bytes
            language: Language code for transcription

        Returns:
            Transcribed text or None on failure.
        """
        try:
            resp = requests.post(
                f"{self.bridge_url}/stt",
                files={"audio": ("audio.wav", audio_data, "audio/wav")},
                data={"language": language},
                timeout=15
            )
            if resp.status_code == 200:
                result = resp.json()
                return result.get('text', '').strip()
            logging.error(f"Bridge STT error: status {resp.status_code}")
            return None
        except Exception as e:
            logging.error(f"Bridge STT request failed: {e}")
            return None

    def health(self) -> bool:
        """Check if the Bridge service is healthy."""
        try:
            resp = requests.get(f"{self.bridge_url}/health", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False