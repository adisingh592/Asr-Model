"""
Configuration settings for Whisper Large-v3-Turbo experiment.
Provides centralized parameters for model loading, audio recording, and inference.
"""

from pathlib import Path
import torch

# ==========================================
# Base Directories
# ==========================================
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT
RESULTS_DIR = PROJECT_ROOT / "results"

# Ensure results directory exists
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# TFLite Model Directory & Paths
MODELS_DIR = PROJECT_ROOT / "models"
TFLITE_DIR = MODELS_DIR / "tflite"
if not TFLITE_DIR.exists():
    TFLITE_DIR = Path("E:/whisper_turbo_experiment/models/tflite")
TFLITE_DIR.mkdir(parents=True, exist_ok=True)

TFLITE_ENCODER_PATH = TFLITE_DIR / "whisper_turbo_encoder.tflite"
TFLITE_DECODER_PATH = TFLITE_DIR / "whisper_turbo_decoder.tflite"

# ==========================================
# Pretrained Model Configuration & Backend
# ==========================================
# Pretrained weights identifier from Hugging Face Hub (No training / weights only)
MODEL_NAME = "openai/whisper-large-v3-turbo"

# Model Execution Backend: "tflite" or "pytorch"
MODEL_BACKEND = "tflite"

# Hardware Device & Precision Selection (PyTorch mode)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TORCH_DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32


# ==========================================
# Audio & Microphone Settings
# ==========================================
SAMPLE_RATE = 16000  # 16 kHz required by Whisper
CHANNELS = 1         # Mono audio
AUDIO_DTYPE = "float32"

# Audio input device selection:
# "laptop" (or None / "default"): Automatically detects and uses the built-in laptop microphone.
# Integer (e.g., 1): Uses a specific sounddevice device index.
AUDIO_DEVICE = "laptop"

# ==========================================
# Live Streaming & Voice Activity Detection (VAD) Settings
# ==========================================
# Audio chunk size per callback (1024 samples = 64 ms at 16 kHz)
STREAM_BLOCK_SIZE = 1024

# Energy-based VAD & Noise Filtering parameters
VAD_ENERGY_THRESHOLD = 0.025    # Higher threshold to completely reject laptop fan / chassis noise
VAD_SILENCE_DURATION = 0.60     # Seconds of silence after speech before committing segment
VAD_MIN_SPEECH_DURATION = 0.45  # Minimum speech duration (seconds) before transcribing
VAD_PRE_SPEECH_BUFFER = 0.25    # Seconds of audio buffered prior to speech trigger
LIVE_UPDATE_INTERVAL = 0.35     # Interval (seconds) between real-time partial transcript updates
MAX_UTTERANCE_SECONDS = 20.0    # Maximum duration before auto-committing segment

# Real-Time DSP Noise Reduction
ENABLE_AUDIO_FILTER = True      # Bandpass filter (85Hz - 7500Hz) & soft noise gating
FILTER_LOWCUT = 90.0            # Low frequency cutoff in Hz (eliminates fan rumble / table thumps)
FILTER_HIGHCUT = 7500.0         # High frequency cutoff in Hz (eliminates electrical hiss)

# ==========================================
# Transcription & Generation Settings (English-Only & Anti-Loop)
# ==========================================
DEFAULT_LANGUAGE = "en"
DEFAULT_TASK = "transcribe"
MAX_NEW_TOKENS = 64
NO_REPEAT_NGRAM_SIZE = 3        # Prevents 3-token phrases from repeating in an utterance
REPETITION_PENALTY = 1.25       # Penalizes autoregressive loop traps
COMPRESSION_RATIO_THRESHOLD = 1.8  # Rejects repetitive hallucinated loops

# Flash Attention option (set to True if FlashAttention-2 is installed on CUDA)
USE_FLASH_ATTENTION_2 = False





