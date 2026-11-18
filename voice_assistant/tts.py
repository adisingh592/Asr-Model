"""
Asynchronous, Non-Blocking Text-To-Speech (TTS) Engine for Voice Assistant.
Supports:
1. pyttsx3 (SAPI5 offline synthesis)
2. Native Windows PowerShell Speech Synthesizer fallback (Zero-dependency on Windows)
"""

import os
import queue
import subprocess
import threading
import time
from typing import Callable, Optional

# Optional import of pyttsx3
try:
    import pyttsx3
    HAS_PYTTSX3 = True
except ImportError:
    HAS_PYTTSX3 = False


class TTSEngine:
    """Non-blocking Speech Synthesizer managing an asynchronous speech queue."""

    def __init__(self, rate: int = 175, volume: float = 1.0):
        self.rate = rate
        self.volume = volume
        self.queue = queue.Queue()
        self.is_running = True
        self.engine = None
        self.on_speech_started: Optional[Callable[[str], None]] = None
        self.on_speech_finished: Optional[Callable[[str], None]] = None

        self._init_engine()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _init_engine(self):
        if HAS_PYTTSX3:
            try:
                self.engine = pyttsx3.init()
                self.engine.setProperty("rate", self.rate)
                self.engine.setProperty("volume", self.volume)
                # Try selecting a pleasant English voice
                voices = self.engine.getProperty("voices")
                if voices:
                    for v in voices:
                        if "zira" in v.name.lower() or "david" in v.name.lower():
                            self.engine.setProperty("voice", v.id)
                            break
            except Exception as e:
                print(f"[TTS] pyttsx3 init notice: {e}. Falling back to native Windows synthesizer.")
                self.engine = None

    def _speech_worker(self):
        while self.is_running:
            try:
                text = self.queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if not text:
                continue

            if self.on_speech_started:
                try:
                    self.on_speech_started(text)
                except Exception:
                    pass

            # Primary: pyttsx3
            spoken = False
            if self.engine is not None:
                try:
                    self.engine.say(text)
                    self.engine.runAndWait()
                    spoken = True
                except Exception as e:
                    print(f"[TTS] pyttsx3 error: {e}")
                    spoken = False

            # Secondary fallback: Windows PowerShell System.Speech
            if not spoken and os.name == "nt":
                try:
                    # Clean text for safe PowerShell execution
                    clean_text = text.replace('"', '""').replace("'", "''")
                    ps_cmd = (
                        f"Add-Type -AssemblyName System.Speech; "
                        f"$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                        f"$synth.Rate = 1; "
                        f"$synth.Speak('{clean_text}');"
                    )
                    subprocess.run(
                        ["powershell", "-NoProfile", "-NonInteractive", "-Command", ps_cmd],
                        check=False,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        timeout=15
                    )
                    spoken = True
                except Exception as e:
                    print(f"[TTS] Native Windows speech fallback error: {e}")

            if self.on_speech_finished:
                try:
                    self.on_speech_finished(text)
                except Exception:
                    pass

            self.queue.task_done()

    def speak(self, text: str):
        """Queues text to be spoken asynchronously without blocking caller."""
        if text and text.strip():
            self.queue.put(text.strip())

    def stop(self):
        """Stops the speech worker thread."""
        self.is_running = False
        if self.engine:
            try:
                self.engine.stop()
            except Exception:
                pass


if __name__ == "__main__":
    tts = TTSEngine()
    print("Testing TTS Engine...")
    tts.speak("Hello! I am your ESP32 Voice Assistant powered by Groq.")
    time.sleep(4)
