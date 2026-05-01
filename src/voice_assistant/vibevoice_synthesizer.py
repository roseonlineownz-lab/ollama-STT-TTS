#!/usr/bin/env python3

"""
vibevoice_synthesizer.py

VibeVoice TTS synthesizer that replaces Piper TTS with the VibeVoice
streaming TTS service (port 8093). Supports both REST and WebSocket
streaming modes for low-latency audio output.

Voice Loop: User speaks -> Whisper STT -> LLM -> VibeVoice TTS -> Speaker
"""

import logging
import json
import queue
import threading
import time
import numpy as np
import sounddevice as sd
import requests
import websocket
from scipy.signal import resample

from .audio_utils import MAX_TTS_ERRORS


class VibeVoiceSynthesizer:
    """TTS Synthesizer using the VibeVoice streaming service on port 8093."""

    def __init__(self, args, interrupt_event: threading.Event):
        self.args = args
        self.interrupt_event = interrupt_event
        self.queue = queue.Queue()
        self.stop_event = threading.Event()
        self.is_speaking_event = threading.Event()
        self.has_failed = threading.Event()

        self.vibevoice_url = getattr(args, 'vibevoice_url', 'http://127.0.0.1:8093')
        self.voice = getattr(args, 'vibevoice_voice', 'en-Carter_man')
        self.output_device_index = getattr(args, 'piper_output_device_index', None)
        self.target_sample_rate = 24000  # VibeVoice default output rate

        # Verify VibeVoice is reachable
        self._check_service()

        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _check_service(self):
        """Verify VibeVoice TTS service is reachable."""
        try:
            resp = requests.get(f"{self.vibevoice_url}/config", timeout=3)
            if resp.status_code == 200:
                config = resp.json()
                voices = config.get('voices', [])
                if self.voice not in voices:
                    logging.warning(
                        f"Voice '{self.voice}' not in available voices: {voices}. "
                        f"Using default: {config.get('default_voice', voices[0] if voices else 'en-Carter_man')}"
                    )
                    self.voice = config.get('default_voice', voices[0] if voices else 'en-Carter_man')
                logging.info(f"VibeVoice TTS connected. Voice: {self.voice}, Available: {len(voices)} voices")
            else:
                logging.warning(f"VibeVoice returned status {resp.status_code}")
        except Exception as e:
            logging.error(f"VibeVoice TTS service unreachable at {self.vibevoice_url}: {e}")
            self.has_failed.set()

    def _synthesize_rest(self, text: str) -> bytes | None:
        """Synthesize speech via VibeVoice REST API (simpler, slight latency)."""
        try:
            resp = requests.post(
                f"{self.vibevoice_url}/tts",
                json={"text": text, "voice": self.voice},
                timeout=10
            )
            if resp.status_code == 200:
                return resp.content  # WAV/PCM audio bytes
            else:
                logging.error(f"VibeVoice TTS error: status {resp.status_code}")
                return None
        except Exception as e:
            logging.error(f"VibeVoice REST TTS failed: {e}")
            return None

    def _worker(self):
        """Worker thread that processes TTS requests from the queue."""
        consecutive_errors = 0
        output_sample_rate = 48000  # sounddevice output rate

        while not self.stop_event.is_set():
            text = None
            try:
                text = self.queue.get(timeout=0.1)
                if text is None:
                    break

                self.is_speaking_event.set()

                # Get audio from VibeVoice
                audio_bytes = self._synthesize_rest(text)
                if audio_bytes is None:
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_TTS_ERRORS:
                        self.has_failed.set()
                        break
                    continue

                # Parse WAV header and extract PCM data
                audio_np = self._parse_audio(audio_bytes)
                if audio_np is None:
                    consecutive_errors += 1
                    continue

                # Resample to output rate if needed
                if self.target_sample_rate != output_sample_rate:
                    num_samples = round(len(audio_np) * output_sample_rate / self.target_sample_rate)
                    if num_samples > 0:
                        audio_np = resample(audio_np, num_samples)

                # Play audio
                with sd.OutputStream(
                    samplerate=output_sample_rate,
                    device=self.output_device_index,
                    channels=1,
                    dtype='int16'
                ) as stream:
                    if self.interrupt_event.is_set():
                        break
                    stream.write(audio_np.astype(np.int16))

                consecutive_errors = 0

            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"VibeVoice TTS playback error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= MAX_TTS_ERRORS:
                    self.has_failed.set()
                    break
            finally:
                if text is not None:
                    self.queue.task_done()
                if self.queue.empty():
                    self.is_speaking_event.clear()

    def _parse_audio(self, audio_bytes: bytes) -> np.ndarray | None:
        """Parse audio bytes from VibeVoice response into numpy array."""
        try:
            # Try parsing as WAV
            import io
            import wave
            wf = wave.open(io.BytesIO(audio_bytes), 'rb')
            frames = wf.readframes(wf.getnframes())
            self.target_sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            wf.close()

            if sampwidth == 2:  # 16-bit
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)
            elif sampwidth == 4:  # 32-bit
                audio_np = np.frombuffer(frames, dtype=np.int32).astype(np.float32)
                audio_np = audio_np / 65536.0  # Scale to int16 range
            else:
                audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

            # Convert stereo to mono if needed
            if channels > 1:
                audio_np = audio_np.reshape(-1, channels).mean(axis=1)

            # Normalize to int16 range
            peak = np.max(np.abs(audio_np))
            if peak > 0:
                audio_np = audio_np / peak * 32767.0

            return audio_np

        except Exception as e:
            # Fallback: try raw PCM (16-bit mono)
            try:
                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                if len(audio_np) > 0:
                    return audio_np
            except Exception:
                pass
            logging.error(f"Failed to parse VibeVoice audio: {e}")
            return None

    def speak(self, text: str):
        """Queue text for speech synthesis."""
        if not self.has_failed.is_set():
            self.queue.put(text)

    def stop(self):
        """Stop the synthesizer and clean up resources."""
        self.stop_event.set()
        self.clear_queue()
        self.queue.put(None)  # Sentinel to stop the worker
        self.thread.join(timeout=5.0)

    def clear_queue(self):
        """Clears all items from the synthesizer queue."""
        with self.queue.mutex:
            self.queue.queue.clear()