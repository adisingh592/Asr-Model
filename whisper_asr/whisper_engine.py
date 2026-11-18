"""
Whisper Large-v3-Turbo Edge & PyTorch ASR Engine for ESP32 Audio Streams.
Lives in whisper_asr/ and integrates with server.py.
"""

import os
import sys
import time
from pathlib import Path
import numpy as np
import torch

# Ensure whisper_asr is in sys.path
WHISPER_DIR = Path(__file__).resolve().parent
if str(WHISPER_DIR) not in sys.path:
    sys.path.insert(0, str(WHISPER_DIR))

import config
from model import load_whisper_turbo
from live_streaming import AudioDSPFilter, HALLUCINATION_PATTERNS


class WhisperTurboASREngine:
    """ASR Engine for Whisper Large-v3-Turbo using the models in whisper_asr."""

    def __init__(self, backend: str = "onnx", language: str = "en"):
        self.backend = backend
        self.language = language
        self.processor = None
        self.model = None
        self.device_str = ""
        self.dtype = torch.float32
        self.dsp_filter = AudioDSPFilter(sample_rate=16000)
        self._is_loaded = False

    def _ensure_loaded(self):
        if not self._is_loaded:
            print("\n" + "=" * 65)
            print("  [*] Loading Whisper Large-v3-Turbo from whisper_asr/ ...")
            print("=" * 65)
            
            # Load using unified loader in whisper_asr/model.py
            self.processor, self.model, self.device_str, self.dtype = load_whisper_turbo(
                model_name=config.MODEL_NAME,
                backend=self.backend,
                verbose=True
            )
            self._is_loaded = True
            print("[+] Whisper Large-v3-Turbo is Ready in whisper_asr!\n")

    def transcribe(self, pcm_bytes: bytes, sample_rate: int = 16000) -> dict:
        """
        Transcribes 16-bit 16kHz PCM audio bytes streamed from the ESP32.
        
        Returns:
            dict with:
                - transcript: str
                - total_asr_latency_ms: float
                - audio_duration_s: float
                - rtf: float
                - metrics: dict (feature_extraction_ms, inference_ms, decode_ms)
        """
        self._ensure_loaded()
        t_start = time.perf_counter()

        # 1. Convert PCM16 to float32 normalized [-1.0, 1.0]
        audio_array = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        audio_dur = len(audio_array) / float(sample_rate) if sample_rate > 0 else 0.0

        if len(audio_array) < int(sample_rate * 0.3):
            return {
                'transcript': '',
                'total_asr_latency_ms': 0.0,
                'audio_duration_s': audio_dur,
                'rtf': 0.0,
                'metrics': {}
            }

        # 2. Apply DSP bandpass noise filtering
        filtered_audio = self.dsp_filter.filter_chunk(audio_array)

        # 3. Audio feature extraction
        t_ext_0 = time.perf_counter()
        inputs = self.processor(
            filtered_audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        input_features = inputs.input_features
        t_ext_1 = time.perf_counter()

        # 4. Model inference
        t_inf_0 = time.perf_counter()
        if hasattr(self.model, "generate"):
            # Edge / ONNX / TFLite engine
            predicted_ids = self.model.generate(
                input_features,
                language=self.language,
                task="transcribe",
                max_new_tokens=config.MAX_NEW_TOKENS,
                no_repeat_ngram_size=config.NO_REPEAT_NGRAM_SIZE,
                repetition_penalty=config.REPETITION_PENALTY
            )
        else:
            with torch.inference_mode():
                predicted_ids = self.model.generate(
                    input_features.to(self.device_str, dtype=self.dtype)
                )
        t_inf_1 = time.perf_counter()

        # 5. Decode predicted token IDs to clean text
        t_dec_0 = time.perf_counter()
        transcript = self.processor.batch_decode(
            predicted_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0].strip()

        # Filter hallucinations / noise artifacts
        if transcript.lower() in HALLUCINATION_PATTERNS:
            transcript = ""

        t_dec_1 = time.perf_counter()
        t_end = time.perf_counter()

        total_lat_ms = (t_end - t_start) * 1000.0
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
