"""
ESP32 Audio Streaming Server with Dual ASR: Moonshine and Whisper Large-v3-Turbo.
Receives 16 kHz PCM audio from ESP32, detects wake phrase 'khus khus',
and offers interactive selection between Moonshine and Whisper Large-v3-Turbo.
"""

import argparse
import asyncio
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import time

# Ensure UTF-8 output on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass
if hasattr(sys.stderr, 'reconfigure'):
    try:
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from websockets.asyncio.server import broadcast, serve

# Add project paths to sys.path
sys.path.insert(0, 'E:/New folder (4)')
sys.path.insert(0, 'E:/kwas')
sys.path.insert(0, 'E:/whisper_turbo_experiment')
sys.path.insert(0, 'E:/whisper_turbo_experiment/src')

from kws.config import AUDIO_CFG, MODEL_CFG, AudioConfig, ModelConfig
from kws.features import LogMelFeatureExtractor
from kws.model import KhusKhusKWS
from moonshine_streaming_engine import MoonshineStreamingEngine

LOG_FILE = 'live_transcript_log.txt'
KWS_THRESHOLD = 0.40

# ASR model used after the wake word: 'ask' prompts in the terminal, or force one model
ASR_MODE = 'ask'

# =====================================================================
# 1. Model Initialization
# =====================================================================

print('=' * 65)
print('  Initializing ESP32 AI Voice Assistant (Dual ASR: Moonshine and Whisper)')
print('=' * 65)

# 1. KWS Model
print('[1/3] Loading Trained "KhusKhus-KWS" Wake-Word Model...')
device = torch.device('cpu')
kws_model = KhusKhusKWS(MODEL_CFG).to(device)
feat_extractor = LogMelFeatureExtractor(AUDIO_CFG).to(device)
feat_extractor.eval()

kws_paths = [
    'models/kws/best_model.pth',
    'best_model.pth',
    'E:/kwas/models/KhusKhus-KWS/best_model.pth'
]
model_loaded = False
for path in kws_paths:
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
        kws_model.load_state_dict(state_dict)
        kws_model.eval()
        val_acc = checkpoint.get('val_acc', 0.9977) * 100
        val_f1 = checkpoint.get('keyword_f1', 1.0)
        print(f'      Loaded from: {path}')
        print(f'      Validation Accuracy: {val_acc:.2f}% | Keyword F1: {val_f1:.2f}')
        print('      Wake Phrase: "khus khus" (Class Index 0) -> ACTIVATED!')
        model_loaded = True
        break

if not model_loaded:
    print('      [Warning] No KWS checkpoint found. Running in passthrough mode.')

# 2. Moonshine Engine
print('[2/3] Loading Pretrained Moonshine Streaming ASR Engine...')
moonshine_engine = MoonshineStreamingEngine(language='en')
print('      Moonshine Edge ASR Ready!')

# 3. Whisper Large-v3-Turbo Engine (Lazy loader)
class WhisperASRAdapter:
    """Embedded ASR Adapter for Whisper Large-v3-Turbo with performance telemetry."""

    def __init__(self, model_id: str = "openai/whisper-large-v3-turbo", device: str = "cpu", language: str = "en"):
        self.model_id = model_id
        self.device = device
        self.language = language
        self.processor = None
        self.model = None

    def _ensure_loaded(self):
        if self.model is None:
            import importlib
            try:
                transformers = importlib.import_module("transformers")
                AutoModelForSpeechSeq2Seq = getattr(transformers, "AutoModelForSpeechSeq2Seq")
                AutoProcessor = getattr(transformers, "AutoProcessor")
            except ImportError:
                raise ImportError(
                    "The 'transformers' package is required for Whisper Large-v3-Turbo. "
                    "Run 'pip install transformers' or choose option [1] Moonshine Streaming ASR."
                )

            print(f"[*] Loading Whisper model ({self.model_id}) on {self.device}...")
            torch_dtype = torch.float16 if str(self.device).lower().startswith("cuda") else torch.float32
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True
            ).to(self.device)
            self.model.eval()
            print("[+] Whisper model loaded successfully.")

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
        self._ensure_loaded()
        t0 = time.perf_counter()
        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_dur = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0

        if len(audio_array) == 0:
            return {
                'transcript': '',
                'total_asr_latency_ms': 0.0,
                'audio_duration_s': 0.0,
                'rtf': 0.0,
                'metrics': {}
            }

        t_ext_0 = time.perf_counter()
        inputs = self.processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
        torch_dtype = torch.float16 if str(self.device).lower().startswith("cuda") else torch.float32
        input_features = inputs.input_features.to(self.device, dtype=torch_dtype)
        t_ext_1 = time.perf_counter()

        t_inf_0 = time.perf_counter()
        with torch.inference_mode():
            predicted_ids = self.model.generate(input_features)
        t_inf_1 = time.perf_counter()

        t_dec_0 = time.perf_counter()
        transcript = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()
        t_dec_1 = time.perf_counter()

        t_end = time.perf_counter()
        total_lat_ms = (t_end - t0) * 1000.0
        ext_ms = (t_ext_1 - t_ext_0) * 1000.0
        inf_ms = (t_inf_1 - t_inf_0) * 1000.0
        dec_ms = (t_dec_1 - t_dec_0) * 1000.0
        rtf = (total_lat_ms / 1000.0) / max(0.01, audio_dur)

        return {
            'transcript': transcript,
            'total_asr_latency_ms': total_lat_ms,
            'audio_duration_s': audio_dur,
            'rtf': rtf,
            'metrics': {
                'feature_extraction_ms': ext_ms,
                'inference_ms': inf_ms,
                'decode_ms': dec_ms
            }
        }

whisper_adapter = None

def get_whisper_adapter():
    global whisper_adapter
    if whisper_adapter is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        whisper_adapter = WhisperASRAdapter(device=device, language='en')
    return whisper_adapter

print('[3/3] Whisper Large-v3-Turbo Adapter configured (On-Demand Loading Ready).\n')

def log_transcript(text: str, wake_detected: bool = False, model_name: str = 'Moonshine'):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    tag = '[WAKE: KHUS KHUS]' if wake_detected else f'[{model_name.upper()}]'
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(f'[{timestamp}] {tag} {text}\n')

# =====================================================================
# 2. Live Dashboard Event Feed (WebSocket, consumed by dashboard/*.html)
# =====================================================================

DASHBOARD_HOST = '127.0.0.1'
DASHBOARD_PORT = 8767
dashboard_clients = set()
event_loop = None
live_state = {'device': None, 'stage': 'waiting', 'model': None}

def emit(event_type: str, **data):
    """Broadcasts one JSON event to every open dashboard window (safe from any thread)."""
    if event_loop is None or not dashboard_clients:
        return
    data['type'] = event_type
    data['ts'] = time.time()
    event_loop.call_soon_threadsafe(broadcast, dashboard_clients, json.dumps(data))

def set_stage(stage: str, model: str = None):
    live_state['stage'] = stage
    live_state['model'] = model
    emit('stage', stage=stage, model=model)

async def dashboard_handler(ws):
    dashboard_clients.add(ws)
    try:
        await ws.send(json.dumps({'type': 'state', 'ts': time.time(), 'threshold': KWS_THRESHOLD, 'asr_mode': ASR_MODE, **live_state}))
        await ws.wait_closed()
    finally:
        dashboard_clients.discard(ws)

# =====================================================================
# 3. Interactive Asynchronous TCP Socket Server
# =====================================================================

async def prompt_for_asr_choice() -> int:
    def _input_sync():
        while True:
            print('\n' + '=' * 50)
            print('  ⚡ WAKE WORD DETECTED: "Khus Khus"')
            print('=' * 50)
            print('Select ASR Model for Command Transcription:')
            print('  [1] Moonshine Streaming Tiny')
            print('  [2] Whisper Large-v3-Turbo')
            print('-' * 50)
            try:
                c = input('Enter choice (1 or 2): ').strip()
                if c in ['1', '2']:
                    return int(c)
                print('Invalid choice. Please enter 1 or 2.')
            except (EOFError, KeyboardInterrupt):
                return 1
    return await asyncio.to_thread(_input_sync)


async def tcp_handler(reader, writer):
    peer = writer.get_extra_info('peername')
    client_ip = peer[0] if peer else 'ESP32 Device'
    
    print('=' * 65)
    print(f'[+] ESP32-S3 Connected from: {client_ip}')
    print('[+] Status: STANDBY -> Waiting for Wake-Word ("Khus Khus")')
    print('=' * 65)
    print('\n[*] Audio stream active. Say "Khus Khus" into the ESP32 microphone...\n')
    live_state['device'] = client_ip
    emit('device_connected', client=client_ip)
    set_stage('standby')

    moonshine_engine.reset()

    def on_partial(partial_text, model_lat_ms):
        if partial_text:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f'  [{ts}] 🎙️ [Hearing]: "{partial_text}" (Model Latency: {model_lat_ms:.1f}ms)          ', end='\r', flush=True)
            emit('partial', text=partial_text, latency_ms=model_lat_ms)

    def on_final(final_text, model_lat_ms, duration_sec, rtf):
        if final_text:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f'\r[{ts}] 🎯 [MOONSHINE TRANSCRIPT]: "{final_text}" | Latency: {model_lat_ms:.1f}ms | Audio Dur: {duration_sec:.2f}s | RTF: {rtf:.2f}    ')
            log_transcript(f'{final_text} (Latency: {model_lat_ms:.1f}ms, RTF: {rtf:.2f})', wake_detected=False, model_name='Moonshine')
            emit('final', text=final_text, model='Moonshine', latency_ms=model_lat_ms, duration_s=duration_sec, rtf=rtf)

    moonshine_engine.on_partial_callback = on_partial
    moonshine_engine.on_final_callback = on_final

    # 1.0 second rolling audio buffer for KWS
    audio_buffer = bytearray()
    MAX_BUFFER_BYTES = 16000 * 2
    last_kws_check = 0
    
    is_awake = False
    active_model_choice = 1 # 1: Moonshine, 2: Whisper
    whisper_post_wake_buffer = bytearray()
    is_collecting_whisper = False
    whisper_collect_start = 0.0
    WHISPER_CAPTURE_SEC = 3.5

    # Dashboard telemetry: incoming byte rate and throttled level updates
    stats_window_start = time.time()
    stats_window_bytes = 0
    total_bytes = 0
    last_level_emit = 0.0

    try:
        while True:
            chunk = await reader.read(1024)
            if not chunk or len(chunk) == 0:
                break

            current_time = time.time()

            # Visual Audio Level Meter (Safe ASCII)
            samples_np = np.frombuffer(chunk, dtype=np.int16)
            rms = np.sqrt(np.mean((samples_np.astype(np.float32) / 32768.0)**2)) if len(samples_np) > 0 else 0
            meter_bars = int(min(10, rms * 50))
            level_str = '#' * meter_bars + '-' * (10 - meter_bars)

            stats_window_bytes += len(chunk)
            total_bytes += len(chunk)
            if current_time - stats_window_start >= 1.0:
                emit('stats', bytes_per_sec=stats_window_bytes / (current_time - stats_window_start), total_bytes=total_bytes)
                stats_window_start = current_time
                stats_window_bytes = 0
            if current_time - last_level_emit >= 0.05:
                last_level_emit = current_time
                emit('level', rms=float(rms))
            
            # --- STAGE 1: Wake-Word Detection (Standby) ---
            if not is_awake:
                audio_buffer.extend(chunk)
                if len(audio_buffer) > MAX_BUFFER_BYTES:
                    audio_buffer = audio_buffer[-MAX_BUFFER_BYTES:]

                if len(audio_buffer) == MAX_BUFFER_BYTES and (current_time - last_kws_check >= 0.06):
                    last_kws_check = current_time
                    
                    raw_float = np.frombuffer(audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    audio_proc = raw_float - np.mean(raw_float)
                    
                    audio_tensor = torch.from_numpy(audio_proc).unsqueeze(0).to(device)
                    
                    t_kws_0 = time.perf_counter()
                    with torch.no_grad():
                        mels = feat_extractor(audio_tensor)
                        logits = kws_model(mels)
                        probs = F.softmax(logits, dim=-1)[0]
                        kw_conf = float(probs[0])
                    t_kws_1 = time.perf_counter()
                    kws_latency_ms = (t_kws_1 - t_kws_0) * 1000.0
                    
                    print(f'  [Standby - Say "Khus Khus"] Mic: [{level_str}]  Wake: {kw_conf*100:4.1f}% | Latency: {kws_latency_ms:4.1f}ms', end='\r', flush=True)
                    emit('kws', conf=kw_conf, latency_ms=kws_latency_ms)

                    if kw_conf >= KWS_THRESHOLD:
                        is_awake = True
                        wake_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        print('\n\n' + '#' * 65)
                        print(f'  ⚡ [WAKE WORD DETECTED!] "KHUS KHUS"')
                        print(f'  📅 Detection Timestamp  : {wake_ts}')
                        print(f'  ⏱️ KWS Inference Latency : {kws_latency_ms:.2f} ms')
                        print(f'  📊 Confidence Score     : {kw_conf*100:.1f}%')
                        print('#' * 65)
                        log_transcript(f'Wake word detected at {wake_ts} (Latency: {kws_latency_ms:.2f}ms, Conf: {kw_conf*100:.1f}%)', wake_detected=True)
                        emit('wake', conf=kw_conf, latency_ms=kws_latency_ms, at=wake_ts)

                        # Prompt user for ASR choice (unless a model was fixed with --asr)
                        if ASR_MODE == 'ask':
                            set_stage('choosing')
                            choice = await prompt_for_asr_choice()
                        else:
                            choice = 1 if ASR_MODE == 'moonshine' else 2
                        active_model_choice = choice

                        if choice == 1:
                            print('\n[+] Moonshine Streaming ASR Activated! Speak your command continuously...')
                            set_stage('listening', model='Moonshine')
                            moonshine_engine.reset()
                            # Feed pre-roll audio buffer into Moonshine
                            moonshine_engine.process_pcm_chunk(bytes(audio_buffer))
                        else:
                            print(f'\n[+] Whisper Large-v3-Turbo Selected! Buffering command speech ({WHISPER_CAPTURE_SEC}s)...')
                            set_stage('listening', model='Whisper')
                            whisper_post_wake_buffer.clear()
                            whisper_post_wake_buffer.extend(audio_buffer) # preserve pre-roll
                            is_collecting_whisper = True
                            whisper_collect_start = time.time()
                            
                        audio_buffer.clear()

            # --- STAGE 2: Moonshine Streaming Mode ---
            elif is_awake and active_model_choice == 1:
                moonshine_engine.process_pcm_chunk(chunk)

            # --- STAGE 2: Whisper Large-v3-Turbo Buffering Mode ---
            elif is_awake and active_model_choice == 2 and is_collecting_whisper:
                whisper_post_wake_buffer.extend(chunk)
                dur = len(whisper_post_wake_buffer) / 32000.0
                print(f'  🎙️ [Buffering Whisper Audio] Mic: [{level_str}] Captured: {dur:4.2f}s / {WHISPER_CAPTURE_SEC:.1f}s', end='\r', flush=True)
                emit('whisper_progress', captured_s=dur, total_s=WHISPER_CAPTURE_SEC)

                if (current_time - whisper_collect_start) >= WHISPER_CAPTURE_SEC:
                    is_collecting_whisper = False
                    audio_snap = bytes(whisper_post_wake_buffer)
                    print(f'\n[*] Running Whisper Large-v3-Turbo on {len(audio_snap)/32000.0:.2f}s audio...')
                    set_stage('transcribing', model='Whisper')

                    adapter = get_whisper_adapter()
                    res = adapter.transcribe(audio_snap, sample_rate=16000)
                    
                    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
                    transcript_text = res.get('transcript', '')
                    total_lat = res.get('total_asr_latency_ms', 0.0)
                    audio_dur = res.get('audio_duration_s', len(audio_snap)/32000.0)
                    rtf = res.get('rtf', 0.0)
                    metrics = res.get('metrics', {})
                    
                    print('\n' + '=' * 65)
                    print('            🎯 WHISPER LARGE-V3-TURBO TRANSCRIPT')
                    print('=' * 65)
                    print(f'  Transcript     : "{transcript_text}"')
                    print(f'  ASR Latency    : {total_lat:.2f} ms')
                    if 'feature_extraction_ms' in metrics:
                        print(f'  - Extraction   : {metrics["feature_extraction_ms"]:.2f} ms')
                        print(f'  - Inference    : {metrics["inference_ms"]:.2f} ms')
                        print(f'  - Decode       : {metrics["decode_ms"]:.2f} ms')
                    print(f'  Audio Duration : {audio_dur:.2f} s')
                    print(f'  RTF            : {rtf:.3f}x')
                    print('=' * 65 + '\n')
                    
                    log_transcript(f'{transcript_text} (Latency: {total_lat:.1f}ms, RTF: {rtf:.2f})', wake_detected=False, model_name='Whisper')
                    emit('final', text=transcript_text, model='Whisper', latency_ms=total_lat, duration_s=audio_dur, rtf=rtf)

                    # Return to standby
                    is_awake = False
                    print('[+] Standby -> Listening for next wake word ("Khus Khus") from ESP32...\n')
                    set_stage('standby')

    except asyncio.CancelledError:
        pass
    except Exception as e:
        print(f'\n[!] Connection exception: {e}')
        emit('error', message=str(e))
    finally:
        if is_awake and active_model_choice == 1 and moonshine_engine.last_partial_text:
            ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            print(f'\r[{ts}] 🎯 [TRANSCRIPT]: "{moonshine_engine.last_partial_text}" (Flushed on Disconnect)           ')
            log_transcript(moonshine_engine.last_partial_text, wake_detected=False, model_name='Moonshine')
            emit('final', text=moonshine_engine.last_partial_text, model='Moonshine', flushed=True)
        print(f'\n[-] ESP32 Disconnected from {client_ip}.\n')
        live_state['device'] = None
        emit('device_disconnected', client=client_ip)
        set_stage('waiting')
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

async def main():
    global event_loop
    event_loop = asyncio.get_running_loop()
    host = '0.0.0.0'
    port = 8766
    server = await asyncio.start_server(tcp_handler, host, port)

    async with serve(dashboard_handler, DASHBOARD_HOST, DASHBOARD_PORT):
        print('=' * 65)
        print(f'   ASR & KWS Server Running on {host}:{port}')
        print(f'   Live Dashboard Feed: ws://{DASHBOARD_HOST}:{DASHBOARD_PORT} (open dashboard/server.html)')
        print(f'   Trained Wake Model : KhusKhus-KWS (best_model.pth)')
        print(f'   Dual ASR Engines   : [1] Moonshine Streaming Tiny  [2] Whisper Large-v3-Turbo')
        print(f'   Post-Wake ASR Mode : {ASR_MODE}')
        print(f'   Ready and waiting for ESP32 connection...')
        print('=' * 65 + '\n')

        async with server:
            await server.serve_forever()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ESP32 KWS + ASR streaming server')
    parser.add_argument('--asr', choices=['ask', 'moonshine', 'whisper'], default='ask',
                        help='ASR model after the wake word (default: ask in the terminal)')
    ASR_MODE = parser.parse_args().asr
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print('\nServer stopped.')
