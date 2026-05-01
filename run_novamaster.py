#!/usr/bin/env python3

"""
run_novamaster.py - NovaMaster Voice Assistant Entry Point

Integrates the ollama-STT-TTS voice loop with NovaMaster's VibeVoice TTS
service instead of Piper TTS. The complete voice loop is:

    User speaks -> Whisper STT (local) -> Ollama LLM -> VibeVoice TTS -> Speaker

Configuration: config.novamaster.ini
Systemd service: novamaster-voice.service
"""

import logging
import sys
import os
import tracemalloc
import warnings

# Suppress known warnings
warnings.filterwarnings("ignore", message="Specified provider 'CUDAExecutionProvider' is not in available provider names.")
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from voice_assistant.config_manager import load_config_and_args, get_ollama_client
from voice_assistant.audio_input import AudioInput
from voice_assistant.audio_utils import SENTENCE_END_PUNCTUATION, monitor_memory
from voice_assistant.transcriber import Transcriber
from voice_assistant.llm_handler import LLMHandler
from voice_assistant.voice_assistant import VoiceAssistant

# Import VibeVoice synthesizer
from voice_assistant.vibevoice_synthesizer import VibeVoiceSynthesizer

import threading
import time
import numpy as np
import re
import gc


class NovaMasterVoiceAssistant:
    """Voice assistant using VibeVoice TTS instead of Piper TTS."""

    def __init__(self, args, client):
        self.args = args
        self.interrupt_event = threading.Event()
        self.conversation_count = 0
        self.is_handling_conversation = False

        # Wake word detection improvements
        self.last_wakeword_time = 0
        self.wakeword_cooldown = 1.0
        self.consecutive_detection_count = 0
        self.required_consecutive = 2

        # Initialize Subsystems - use VibeVoice instead of Piper
        self.audio = AudioInput(args)
        self.transcriber = Transcriber(args)
        self.tts = VibeVoiceSynthesizer(args, self.interrupt_event)
        self.llm = LLMHandler(client, args)

        # Wake word setup
        if not os.path.exists(args.wakeword_model_path):
            raise FileNotFoundError(f"Wakeword model missing: {args.wakeword_model_path}")

        from openwakeword.model import Model
        self.oww_model = Model(wakeword_model_paths=[args.wakeword_model_path])
        self.wakeword_key = list(self.oww_model.models.keys())[0]
        logging.info(f"Wakeword model loaded with key: {self.wakeword_key}")

    def run(self):
        logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
        self.audio.start()

        score_history = []
        weighted_scores = []

        try:
            while True:
                if self.is_handling_conversation:
                    time.sleep(0.01)
                    continue

                chunk = self.audio.get_chunk()
                if not chunk:
                    time.sleep(0.001)
                    continue

                int16_audio = np.frombuffer(chunk, dtype=np.int16)
                prediction = self.oww_model.predict(int16_audio)
                score = prediction.get(self.wakeword_key, 0)

                score_history.append(score)
                if len(score_history) > 100:
                    score_history.pop(0)

                current_time = time.time()

                weighted_scores.append(score)
                if len(weighted_scores) > 5:
                    weighted_scores.pop(0)
                avg_score = sum(weighted_scores) / len(weighted_scores)

                if score > self.args.wakeword_threshold:
                    if current_time - self.last_wakeword_time > self.wakeword_cooldown:
                        self.consecutive_detection_count += 1

                        if (self.consecutive_detection_count >= self.required_consecutive and
                            avg_score > self.args.wakeword_threshold * 0.85):

                            logging.info(f"Wakeword detected! (score: {score:.2f}, avg: {avg_score:.2f})")
                            self.last_wakeword_time = current_time
                            self.consecutive_detection_count = 0
                            weighted_scores.clear()
                            self.oww_model.reset()

                            self.is_handling_conversation = True
                            self._handle_conversation()

                            score_history.clear()
                            logging.info(f"Ready! Listening for '{self.args.wakeword}'...")
                    else:
                        time_since_last = current_time - self.last_wakeword_time
                        logging.debug(f"Wakeword in cooldown (time since last: {time_since_last:.2f}s)")
                else:
                    if self.consecutive_detection_count > 0:
                        self.consecutive_detection_count = 0

        except KeyboardInterrupt:
            logging.info("Stopping...")
        self.cleanup()

    def _process_plugins(self, text: str) -> str:
        """Process simple plugins like [current time]."""
        if "[current time]" in text.lower():
            current_time = time.strftime("%I:%M %p")
            text = re.sub(r'\[current time\]', current_time, text, flags=re.IGNORECASE)
        return text

    def _handle_conversation(self):
        try:
            self.audio.stop()
            self.audio.clear_buffer()

            self.tts.speak("Yes?")
            self.tts.queue.join()

            self.interrupt_event.clear()

            self.audio.start()
            time.sleep(0.4)

            audio_np = self.audio.record_phrase(self.interrupt_event, self.args.listen_timeout)
            self.audio.stop()

            if audio_np is None:
                self.audio.start()
                return

            # Transcribe with retry
            user_text = self._transcribe_with_retry(audio_np)
            del audio_np

            if not user_text or not user_text.strip():
                self.audio.start()
                return

            # Trim wake word
            original_text = user_text
            if self.args.trim_wake_word:
                user_text = self._trim_wakeword(user_text)
                if user_text != original_text:
                    logging.debug(f"Wake word trimmed: '{original_text}' -> '{user_text}'")

            if not user_text or not user_text.strip():
                self.audio.start()
                return

            # Take first sentence
            sentences = re.split(r'(?<=[.?!])\s+', user_text)
            if sentences and sentences[0] != user_text:
                user_text = sentences[0]

            user_text = self._process_plugins(user_text)
            logging.info(f"You: {user_text}")

            # Check exit/reset commands
            user_text_lower = user_text.lower()
            if "exit" in user_text_lower or "goodbye" in user_text_lower:
                self.tts.speak("Goodbye.")
                self.tts.queue.join()
                exit(0)

            if "new chat" in user_text_lower or "reset chat" in user_text_lower:
                self.llm.reset_history()
                self.tts.speak("Chat history cleared.")
                self.tts.queue.join()
                self.audio.start()
                return

            # Stream LLM response to TTS
            sentence_buffer = ""
            for token in self.llm.chat_stream(user_text):
                if token is None:
                    break
                if self.interrupt_event.is_set():
                    self.tts.clear_queue()
                    break

                sentence_buffer += token
                if any(p in token for p in SENTENCE_END_PUNCTUATION):
                    sentence = sentence_buffer.strip()
                    if sentence:
                        self.tts.speak(sentence)
                    sentence_buffer = ""

            if sentence_buffer.strip() and not self.interrupt_event.is_set():
                self.tts.speak(sentence_buffer.strip())

            self.tts.queue.join()
            self.conversation_count += 1

            if self.args.gc_interval > 0 and self.conversation_count % self.args.gc_interval == 0:
                gc.collect()

            self.audio.start()
        finally:
            self.is_handling_conversation = False

    def _transcribe_with_retry(self, audio_np, max_retries=3):
        """Transcribe with progressive threshold relaxation."""
        original_logprob = self.args.whisper_avg_logprob
        original_nospeech = self.args.whisper_no_speech_prob

        threshold_steps = [
            (original_logprob, original_nospeech),
            (original_logprob - 0.15, original_nospeech + 0.1),
            (original_logprob - 0.3, original_nospeech + 0.2),
        ]

        for attempt in range(min(max_retries, len(threshold_steps))):
            logprob_threshold, nospeech_threshold = threshold_steps[attempt]
            self.args.whisper_avg_logprob = logprob_threshold
            self.args.whisper_no_speech_prob = nospeech_threshold

            user_text = self.transcriber.transcribe(audio_np)
            if user_text and user_text.strip():
                self.args.whisper_avg_logprob = original_logprob
                self.args.whisper_no_speech_prob = original_nospeech
                return user_text

        self.args.whisper_avg_logprob = original_logprob
        self.args.whisper_no_speech_prob = original_nospeech
        return ""

    def _trim_wakeword(self, text: str) -> str:
        """Trim wake word from transcription."""
        wakeword = self.args.wakeword.lower()
        text_lower = text.lower().strip()

        core_wakeword_names = ["jarvis", "jarlis", "jarvas", "jarves", "jarvys", "jarvois", "nova"]
        patterns_to_match = [re.escape(wakeword)]
        for name in core_wakeword_names:
            patterns_to_match.append(r"(?:hey\s*)?" + re.escape(name))
            patterns_to_match.append(re.escape(name))

        # Match at start
        combined_start_pattern = r"^(?:" + "|".join(patterns_to_match) + r")\b[.,!?]*\s*"
        match_start = re.match(combined_start_pattern, text_lower, re.IGNORECASE)
        if match_start:
            return text[len(match_start.group(0)):].strip()

        # Match at end
        combined_end_pattern = r"\s*\b(?:" + "|".join(patterns_to_match) + r")[.,!?]*$"
        match_end = re.search(combined_end_pattern, text_lower, re.IGNORECASE)
        if match_end:
            return text[:match_end.start()].strip()

        return text

    def cleanup(self):
        logging.debug("Starting cleanup")
        self.audio.stop()
        self.tts.stop()
        self.transcriber.close()
        logging.debug("Cleanup complete")


def setup_logging():
    log_format = "%(levelname)s %(asctime)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def main():
    """NovaMaster Voice Assistant entry point."""
    try:
        setup_logging()
    except Exception as e:
        print(f"FATAL: Could not set up logging: {e}", file=sys.stderr)
        sys.exit(1)

    # Override config file path to use NovaMaster config
    import voice_assistant.config_manager as cm
    nm_config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.novamaster.ini')
    cm.CONFIG_FILE_NAME = nm_config

    args, _, should_exit = load_config_and_args()

    # Add NovaMaster-specific attributes
    if not hasattr(args, 'vibevoice_url'):
        args.vibevoice_url = 'http://127.0.0.1:8093'
    if not hasattr(args, 'vibevoice_voice'):
        args.vibevoice_voice = 'en-Carter_man'
    if not hasattr(args, 'tts_engine'):
        args.tts_engine = 'vibevoice'

    logging.info(f"NovaMaster Voice Assistant starting...")
    logging.info(f"  LLM model: {args.ollama_model}")
    logging.info(f"  Whisper model: {args.whisper_model}")
    logging.info(f"  TTS engine: VibeVoice ({args.vibevoice_url})")
    logging.info(f"  Voice: {args.vibevoice_voice}")
    logging.info(f"  Wakeword: '{args.wakeword}'")

    if args.debug:
        tracemalloc.start()

    assistant = None
    try:
        if should_exit:
            sys.exit(0)

        ollama_client = get_ollama_client(args.ollama_host)
        if ollama_client is None:
            logging.warning("Ollama server not reachable. Assistant will run but cannot respond.")

        assistant = NovaMasterVoiceAssistant(args, ollama_client)
        assistant.run()

    except IOError as e:
        logging.critical(f"FATAL: Audio initialization error: {e}")
    except (RuntimeError, OSError, ValueError) as e:
        logging.critical(f"FATAL: Model loading error: {e}")
    except Exception as e:
        logging.critical(f"Unexpected error: {e}", exc_info=True)
    finally:
        if assistant:
            assistant.cleanup()


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        pass