"""
Interactive Standalone Laptop Microphone Test for Voice AI Assistant.
Runs directly on your laptop without requiring an ESP32 or external server.

Features:
1. Direct Voice Test: Speak a query (e.g. 'What is the weather in Delhi?') -> Transcribed -> Groq Brain -> TTS Audio Out.
2. Full KWS Wake Word Test: Say 'Khus Khus' into laptop mic -> Triggers Assistant -> Speaks answer.
3. Quick Diagnostic Check: Tests Weather API, Groq connection, and TTS speech output.
"""

import argparse
import os
import queue
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import sounddevice as sd
from scipy.signal import firwin, lfilter, lfilter_zi

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent))
from voice_assistant import VoiceAssistant
from voice_assistant.weather_tool import get_live_weather

TARGET_RATE = 16000
CHUNK_SAMPLES = 512
CANDIDATE_RATES = (16000, 48000, 32000, 96000)

audio_queue = queue.Queue()


def supported_rate(device):
    for rate in CANDIDATE_RATES:
        try:
            sd.check_input_settings(device=device, samplerate=rate, channels=1, dtype='int16')
            return rate
        except Exception:
            continue
    return None


def pick_input_device():
    """Returns (device_index, rate) of the active physical microphone."""
    devices = sd.query_devices()

    # 1. Prefer real microphone arrays or internal mics on stable APIs (MME, DirectSound, WASAPI)
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and d.get("hostapi", -1) in (0, 1, 2):
            name_lower = d["name"].lower()
            if any(k in name_lower for k in ("microphone array", "smart sound", "realtek mic", "internal", "built-in")):
                rate = supported_rate(idx)
                if rate:
                    return idx, rate

    # 2. Prefer any real microphone (excluding speakers, stereo mix, midi)
    for idx, d in enumerate(devices):
        if d.get("max_input_channels", 0) > 0 and d.get("hostapi", -1) in (0, 1, 2):
            name_lower = d["name"].lower()
            if not any(b in name_lower for b in ("speaker", "output", "stereo mix", "wave", "midi", "mapper")):
                rate = supported_rate(idx)
                if rate:
                    return idx, rate

    # 3. Default input device if valid
    default_dev = sd.default.device[0]
    if default_dev is not None and default_dev >= 0:
        rate = supported_rate(default_dev)
        if rate:
            return default_dev, rate

    return None, None


class Decimator:
    """Anti-alias FIR filter + decimation to 16 kHz."""
    def __init__(self, factor):
        self.factor = factor
        if factor > 1:
            self.taps = firwin(97, 0.9 * (TARGET_RATE / 2), fs=TARGET_RATE * factor)
            self.zi = lfilter_zi(self.taps, 1.0) * 0.0

    def __call__(self, block):
        if self.factor == 1:
            return block
        filtered, self.zi = lfilter(self.taps, 1.0, block.astype(np.float32), zi=self.zi)
        return np.clip(filtered[::self.factor], -32768, 32767).astype(np.int16)


def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata[:, 0].copy())


def record_seconds(duration_sec: float, device: int, rate: int, decimate: Decimator) -> bytes:
    """Records audio from laptop mic for a specified duration and displays live visual meter."""
    while not audio_queue.empty():
        audio_queue.get_nowait()

    collected_chunks = []
    total_samples = 0
    needed_samples = int(TARGET_RATE * duration_sec)

    with sd.InputStream(samplerate=rate, blocksize=CHUNK_SAMPLES * decimate.factor, dtype='int16',
                        channels=1, device=device, callback=audio_callback):
        while total_samples < needed_samples:
            block = audio_queue.get()
            pcm16 = decimate(block)
            collected_chunks.append(pcm16.tobytes())
            total_samples += len(pcm16)

            # Live audio level meter
            rms = np.sqrt(np.mean((pcm16.astype(np.float32) / 32768.0) ** 2))
            bars = int(min(15, rms * 75))
            meter = '#' * bars + '-' * (15 - bars)
            sec_done = total_samples / TARGET_RATE
            print(f"  🎙️ [Listening]: [{meter}] {sec_done:.1f}s / {duration_sec:.1f}s", end="\r", flush=True)

    print()
    return b"".join(collected_chunks)


def run_quick_diagnostics(assistant: VoiceAssistant):
    """Verifies Weather API, Groq connection, and TTS speech."""
    print("\n" + "=" * 65)
    print("  🧪 RUNNING QUICK DIAGNOSTICS")
    print("=" * 65)

    print("\n[1/3] Testing Free Live Weather Tool (Open-Meteo)...")
    weather_res = get_live_weather("Delhi")
    print(f"      Result: {weather_res}")

    print("\n[2/3] Testing Groq Cloud Intelligence...")
    groq_res = assistant.brain.query("Give me a one-sentence motivation quote.")
    print(f"      Answer : \"{groq_res.get('response_text')}\"")
    print(f"      Latency: {groq_res.get('latency_ms', 0):.1f} ms")

    print("\n[3/3] Testing Speech Synthesis (TTS)...")
    assistant.tts.speak("Diagnostics complete. Your voice assistant is working properly.")
    time.sleep(3)
    print("      [+] Diagnostics Successful!\n")


def run_direct_voice_test(assistant: VoiceAssistant, device: int, rate: int, decimate: Decimator):
    """Directly records question from mic, transcribes, and speaks response."""
    from moonshine_streaming_engine import MoonshineStreamingEngine
    print("[*] Loading Ultra-Fast Moonshine Streaming ASR Engine...")
    asr_engine = MoonshineStreamingEngine(language="en")

    print("\n" + "=" * 65)
    print("  🎙️ DIRECT LAPTOP MIC VOICE ASSISTANT TEST (Moonshine + Groq)")
    print("=" * 65)
    print("Instructions:")
    print("  1. Press Enter when ready.")
    print("  2. Speak your question into the laptop mic for 4 seconds.")
    print("     (Examples: 'What is the weather in Delhi?', 'What time is it?')")
    print("  3. The assistant will transcribe with Moonshine, query Groq, and speak.")
    print("-" * 65)

    while True:
        try:
            input("\n👉 Press [Enter] to start recording (or Ctrl+C to exit)... ")
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        print("\n[*] 🔴 RECORDING NOW - Speak into your laptop microphone...")
        pcm_bytes = record_seconds(duration_sec=4.0, device=device, rate=rate, decimate=decimate)

        t_stt_0 = time.perf_counter()
        print("\n[*] Transcribing audio with Moonshine Edge ASR...")
        query = asr_engine.transcribe_audio(pcm_bytes)
        stt_ms = (time.perf_counter() - t_stt_0) * 1000.0
        print(f"[*] Transcription completed in {stt_ms:.1f} ms!")

        if not query:
            print("[-] No speech detected. Try speaking louder or moving closer to mic.\n")
            continue

        print(f"\n[+] Transcribed: \"{query}\"")
        assistant.process_command_text(query)
        time.sleep(1)


def run_wake_word_test(assistant: VoiceAssistant, device: int, rate: int, decimate: Decimator):
    """Full end-to-end test including 'Khus Khus' wake-word detection."""
    import torch
    import torch.nn.functional as F
    from kws.config import AUDIO_CFG, MODEL_CFG
    from kws.features import LogMelFeatureExtractor
    from kws.model import KhusKhusKWS
    from moonshine_streaming_engine import MoonshineStreamingEngine

    print("\n[*] Loading KWS Wake Word Model ('Khus Khus')...")
    device_torch = torch.device('cpu')
    kws_model = KhusKhusKWS(MODEL_CFG).to(device_torch)
    feat_extractor = LogMelFeatureExtractor(AUDIO_CFG).to(device_torch)
    feat_extractor.eval()

    kws_path = 'models/kws/best_model.pth'
    if os.path.exists(kws_path):
        ckpt = torch.load(kws_path, map_location=device_torch)
        kws_model.load_state_dict(ckpt.get('model_state_dict', ckpt))
        kws_model.eval()
        print(f"[+] Loaded KWS model from {kws_path}")
    else:
        print(f"[-] KWS checkpoint not found at {kws_path}")
        return

    print("[*] Loading Ultra-Fast Moonshine Streaming ASR Engine...")
    asr_engine = MoonshineStreamingEngine(language="en")

    print("\n" + "=" * 65)
    print("  ⚡ FULL WAKE-WORD ('Khus Khus') + VOICE ASSISTANT TEST")
    print("=" * 65)
    print("Say 'Khus Khus' into your laptop mic. When detected, speak your question!\n")

    audio_buf = bytearray()
    MAX_BUF = 16000 * 2  # 1-second rolling buffer
    last_check = 0.0

    while not audio_queue.empty():
        audio_queue.get_nowait()

    print("  [Standby - Say 'Khus Khus'] Mic: [------------] Wake Conf:  0.0%", end="\r", flush=True)

    with sd.InputStream(samplerate=rate, blocksize=CHUNK_SAMPLES * decimate.factor, dtype='int16',
                        channels=1, device=device, callback=audio_callback):
        while True:
            try:
                block = audio_queue.get(timeout=2.0)
            except queue.Empty:
                continue
            pcm16 = decimate(block)
            audio_buf.extend(pcm16.tobytes())
            if len(audio_buf) > MAX_BUF:
                audio_buf = audio_buf[-MAX_BUF:]

            now = time.time()
            if len(audio_buf) == MAX_BUF and (now - last_check >= 0.06):
                last_check = now
                raw = np.frombuffer(audio_buf, dtype=np.int16).astype(np.float32) / 32768.0
                raw = raw - np.mean(raw)
                tensor = torch.from_numpy(raw).unsqueeze(0).to(device_torch)

                with torch.no_grad():
                    mels = feat_extractor(tensor)
                    logits = kws_model(mels)
                    kw_conf = float(F.softmax(logits, dim=-1)[0][0])

                rms = np.sqrt(np.mean(raw ** 2))
                bars = int(min(12, rms * 60))
                meter = '#' * bars + '-' * (12 - bars)
                print(f"  [Standby - Say 'Khus Khus'] Mic: [{meter}] Wake Conf: {kw_conf*100:4.1f}%", end="\r", flush=True)

                if kw_conf >= 0.40:
                    print('\n\n' + '#' * 65)
                    print(f'  ⚡ [WAKE DETECTED!] "KHUS KHUS" (Confidence: {kw_conf*100:.1f}%)')
                    print('#' * 65)
                    assistant.tts.speak("Yes, how can I help you?")
                    time.sleep(1.2)

                    print("\n[*] 🔴 RECORDING QUESTION (Speak now for 4.0 seconds)...")
                    q_pcm = record_seconds(duration_sec=4.0, device=device, rate=rate, decimate=decimate)

                    t_stt_0 = time.perf_counter()
                    print("\n[*] Transcribing your question with Moonshine...")
                    q_text = asr_engine.transcribe_audio(q_pcm)
                    stt_ms = (time.perf_counter() - t_stt_0) * 1000.0
                    print(f"[*] Transcribed in {stt_ms:.1f} ms!")

                    if q_text:
                        print(f"\n[🎙️ Question]: \"{q_text}\"")
                        assistant.process_command_text(q_text)
                    else:
                        print("[-] No question heard.")

                    print("\n[+] Resuming standby listening for 'Khus Khus'...\n")
                    audio_buf.clear()


def main():
    parser = argparse.ArgumentParser(description="Test Voice Assistant on Laptop Microphone")
    parser.add_argument("--mode", choices=["wake", "direct", "diag"], default="wake",
                        help="'wake' (first wake word 'Khus Khus', then voice assistant), 'direct' (press enter and speak), or 'diag' (quick diagnostic test)")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device index")
    args = parser.parse_args()

    # Detect laptop mic
    if args.device is None:
        device, rate = pick_input_device()
        if device is None:
            print("[-] No working microphone found. Check sound settings.")
            return
    else:
        device, rate = args.device, supported_rate(args.device)

    dev_name = sd.query_devices(device)['name']
    decimate = Decimator(rate // TARGET_RATE)
    print("=" * 65)
    print("  🎤 LAPTOP MICROPHONE INITIALIZED")
    print("=" * 65)
    print(f"  Device   : [{device}] {dev_name}")
    print(f"  Rate     : {rate} Hz -> Resampled to {TARGET_RATE} Hz")
    print(f"  Test Mode: {args.mode.upper()}")
    print("=" * 65)

    assistant = VoiceAssistant()

    if args.mode == "diag":
        run_quick_diagnostics(assistant)
    elif args.mode == "wake":
        run_wake_word_test(assistant, device, rate, decimate)
    else:
        run_direct_voice_test(assistant, device, rate, decimate)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTest terminated.")
