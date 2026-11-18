"""
Voice Assistant Engine coordinating Audio Ingestion -> STT Transcription -> Groq Intelligence -> TTS Speech.
"""

from datetime import datetime
import time
from typing import Callable, Dict, Optional

from voice_assistant.groq_client import GroqVoiceBrain
from voice_assistant.tts import TTSEngine


class VoiceAssistant:
    """Coordinates voice-driven question answering and spoken replies."""

    def __init__(self, groq_api_key: Optional[str] = None):
        self.brain = GroqVoiceBrain(api_key=groq_api_key)
        self.tts = TTSEngine(rate=180, volume=1.0)
        self.last_user_query = ""
        self.last_assistant_reply = ""
        
        # Callbacks for telemetry & dashboard
        self.on_query_callback: Optional[Callable[[str], None]] = None
        self.on_reply_callback: Optional[Callable[[str, float], None]] = None

    def process_command_text(self, text_query: str) -> Dict[str, any]:
        """
        Takes transcribed text from speech, queries Groq, and speaks the answer aloud.
        
        Returns:
            dict containing response details and latencies.
        """
        if not text_query or not text_query.strip():
            msg = "I did not hear a command. Please try speaking after the wake word."
            self.tts.speak(msg)
            return {"response_text": msg, "latency_ms": 0.0, "tool_used": None}

        self.last_user_query = text_query.strip()
        if self.on_query_callback:
            try:
                self.on_query_callback(self.last_user_query)
            except Exception:
                pass

        print(f"\n[Voice Assistant] Processing Query: \"{self.last_user_query}\"")
        result = self.brain.query(self.last_user_query)
        reply = result.get("response_text", "")
        latency_ms = result.get("latency_ms", 0.0)
        tool_used = result.get("tool_used")

        self.last_assistant_reply = reply
        print(f"[Voice Assistant] Speaking Answer : \"{reply}\"")
        if tool_used:
            print(f"[Voice Assistant] Tool Invoked   : {tool_used}")
        print(f"[Voice Assistant] Groq Latency   : {latency_ms:.1f} ms\n")

        # Speak answer aloud asynchronously
        self.tts.speak(reply)

        if self.on_reply_callback:
            try:
                self.on_reply_callback(reply, latency_ms)
            except Exception:
                pass

        return result

    def close(self):
        """Clean shutdown of speech engine."""
        self.tts.stop()
