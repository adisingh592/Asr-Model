"""
Live Real-Time Streaming Transcription for Whisper Large-v3-Turbo.
Features:
- English-Only locked decoding
- Anti-Loop Protection: no_repeat_ngram_size=3 & repetition_penalty=1.25
- Real-Time DSP Filtering: Butterworth Bandpass (90Hz-7500Hz) & Adaptive Noise Gate
- Multi-tier Loop & Hallucination Guard (rejects phantom text, "men men men", "bit of bit of" loops)
- Silent output: Displays ONLY the clean transcribed words
"""

import argparse
import collections
import os
import queue
import re
import sys
import time
import warnings
import zlib
from pathlib import Path

# Silence all third-party logging, warnings, and environment noise
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import numpy as np
from scipy import signal
import sounddevice as sd
import torch

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from model import load_whisper_turbo


# Known Whisper hallucinations on low energy / background noise
HALLUCINATION_PATTERNS = {
    "", ".", "..", "...", ",", "?", "!", "'", '"', "-", " ",
    "you", "thank you", "thank you.", "thank you very much.",
    "thanks for watching", "thanks for watching!", "thanks for watching.",
    "subtitles by", "subtitles by community", "subtitles",
    "bye", "bye.", "goodbye", "goodbye.", "see you next time",
    "[blank_audio]", "[applause]", "[laughter]", "[music]",
    "so", "the", "a", "i", "and", "it's", "he's", "she's",
    "a man", "a bit of", "bit of"
}


def clean_device_name(name: str) -> str:
    """Sanitizes device names to avoid terminal encoding issues on Windows."""
    return name.encode("ascii", "replace").decode("ascii").replace("?", " ").strip()


def get_laptop_microphone(config_device=config.AUDIO_DEVICE):
    """
    Automatically detects and selects the built-in laptop microphone.
    
    Returns:
        tuple (int, str): (device_index, clean_device_name)
    """
    devices = sd.query_devices()
    input_devices = [idx for idx, d in enumerate(devices) if d.get("max_input_channels", 0) > 0]
    
    if not input_devices:
        return None, "None"
        
    if isinstance(config_device, int) and config_device in input_devices:
        return config_device, clean_device_name(devices[config_device]["name"])
        
    default_dev_idx = sd.default.device[0]
    if default_dev_idx is not None and default_dev_idx in input_devices:
        return default_dev_idx, clean_device_name(devices[default_dev_idx]["name"])
        
    for idx in input_devices:
        name_lower = devices[idx]["name"].lower()
        if any(term in name_lower for term in ["microphone array", "smart sound", "realtek", "internal", "built-in"]):
            return idx, clean_device_name(devices[idx]["name"])
            
    first_idx = input_devices[0]
    return first_idx, clean_device_name(devices[first_idx]["name"])


class AudioDSPFilter:
    """
    Real-Time Digital Signal Processing (DSP) filter:
    1. 4th-order Butterworth Bandpass Filter (90 Hz - 7500 Hz) to eliminate fan hum & rumble.
    2. Adaptive Noise Gating to suppress low-level ambient hiss.
    """
    def __init__(self, sample_rate: int = config.SAMPLE_RATE, lowcut: float = config.FILTER_LOWCUT, highcut: float = config.FILTER_HIGHCUT):
        self.sample_rate = sample_rate
        nyq = 0.5 * sample_rate
        low = max(0.001, lowcut / nyq)
        high = min(0.999, highcut / nyq)
        self.b, self.a = signal.butter(4, [low, high], btype="bandpass")
        
    def filter_chunk(self, audio: np.ndarray, noise_floor: float = 0.01) -> np.ndarray:
        """Applies bandpass filtering and soft noise gating."""
        if len(audio) < 32:
            return audio
            
        try:
            filtered = signal.filtfilt(self.b, self.a, audio).astype(np.float32)
        except Exception:
            filtered = audio.astype(np.float32)
            
        rms = np.sqrt(np.mean(filtered**2))
        gate_threshold = noise_floor * 1.5
        if rms < gate_threshold:
            filtered = filtered * 0.10  # Strong attenuation on quiet hiss
            
        return filtered


class WhisperLiveStreamer:
    """
    Clean real-time streaming speech recognizer.
    Outputs ONLY the transcribed words with anti-loop protection.
    """
    
    def __init__(
        self,
        processor,
        model,
        device: str,
        torch_dtype: torch.dtype,
        device_index: int = None,
        sample_rate: int = config.SAMPLE_RATE,
        block_size: int = config.STREAM_BLOCK_SIZE,
    ):
        self.processor = processor
        self.model = model
        self.device = device
        self.torch_dtype = torch_dtype
        self.device_index = device_index
        self.sample_rate = sample_rate
        self.block_size = block_size
        
        # Audio DSP Filter
        self.dsp_filter = AudioDSPFilter(sample_rate=sample_rate) if config.ENABLE_AUDIO_FILTER else None
        
        # Audio frame queue
        self.audio_queue = queue.Queue()
        
        # VAD & Buffer Management
        self.block_duration = self.block_size / self.sample_rate
        self.pre_buffer_len = max(2, int(config.VAD_PRE_SPEECH_BUFFER / self.block_duration))
        self.pre_buffer = collections.deque(maxlen=self.pre_buffer_len)
        self.speech_buffer = []
        
        # Adaptive noise floor tracking
        self.noise_floor = 0.015
        self.consecutive_speech_blocks = 0
        self.required_speech_blocks = 3  # ~192ms of continuous speech before triggering
        
        # State tracking
        self.is_speaking = False
        self.speech_start_time = 0.0
        self.silence_start_time = 0.0
        self.last_live_infer_time = 0.0
        
        # Low-latency English-only generation arguments with Anti-Loop constraints
        self.generate_kwargs = {
            "language": config.DEFAULT_LANGUAGE,
            "task": config.DEFAULT_TASK,
            "num_beams": 1,
            "do_sample": False,
            "max_new_tokens": config.MAX_NEW_TOKENS,
            "no_repeat_ngram_size": config.NO_REPEAT_NGRAM_SIZE,  # Prevents 3-gram repetition
            "repetition_penalty": config.REPETITION_PENALTY,      # Strongly penalizes loops
            "use_cache": True,
            "condition_on_prev_tokens": False,
            "return_timestamps": False,
        }

    def _audio_callback(self, indata, frames, time_info, status):
        """Audio callback from sounddevice."""
        block = indata[:, 0].copy()
        
        if self.dsp_filter is not None:
            filtered_block = self.dsp_filter.filter_chunk(block, noise_floor=self.noise_floor)
        else:
            filtered_block = block
            
        rms = float(np.sqrt(np.mean(filtered_block**2)))
        self.audio_queue.put((filtered_block, rms))

    def _is_hallucination_or_loop(self, text: str) -> bool:
        """
        Determines if the transcribed text is a phantom artifact or an autoregressive repetition loop
        (e.g., 'a man, a man, a man', 'a bit of a bit of a bit of').
        """
        cleaned = text.strip()
        if not cleaned or len(cleaned) < 2:
            return True
            
        cleaned_lower = cleaned.lower()
        cleaned_no_punct = re.sub(r"[^\w\s]", "", cleaned_lower).strip()
        
        if cleaned_lower in HALLUCINATION_PATTERNS or cleaned_no_punct in HALLUCINATION_PATTERNS:
            return True
            
        words = re.findall(r"\b\w+\b", cleaned_lower)
        if not words:
            return True
            
        # Check 1: Unique words density (catches low diversity repetition loops)
        if len(words) >= 4 and (len(set(words)) / len(words)) < 0.48:
            return True
            
        # Check 2: Repeated n-grams (catches repeated 1-word, 2-word, or 3-word loops)
        for n in [1, 2, 3]:
            if len(words) >= n:
                ngrams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
                counts = collections.Counter(ngrams)
                for ng, count in counts.items():
                    if count >= 3:
                        return True
                        
        # Check 3: Compression ratio (repetitive text compresses unnaturally high)
        if len(cleaned) > 25:
            compressed = zlib.compress(cleaned.encode("utf-8"))
            comp_ratio = len(cleaned.encode("utf-8")) / max(1, len(compressed))
            if comp_ratio > config.COMPRESSION_RATIO_THRESHOLD:
                return True
                
        return False

    def _transcribe_audio_array(self, audio_array: np.ndarray) -> str:
        """Runs Whisper inference on 1D audio array."""
        if len(audio_array) < int(self.sample_rate * 0.40):
            return ""
            
        # Reject audio if overall energy is barely above noise floor
        overall_rms = np.sqrt(np.mean(audio_array**2))
        if overall_rms < (self.noise_floor * 1.35):
            return ""
            
        try:
            inputs = self.processor(
                audio_array,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            if str(self.device).lower().startswith("cuda") or str(self.device).lower().startswith("cpu"):
                input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)
            else:
                input_features = inputs.input_features
            
            with torch.inference_mode():
                predicted_ids = self.model.generate(input_features, **self.generate_kwargs)
                
            transcription = self.processor.batch_decode(
                predicted_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )[0].strip()
            
            if self._is_hallucination_or_loop(transcription):
                return ""
                
            return transcription
        except Exception:
            return ""

    def start_streaming(self):
        """Starts live microphone listening and outputs ONLY transcribed words."""
        # Warmup the model
        dummy = np.zeros(int(self.sample_rate * 0.5), dtype=np.float32)
        _ = self._transcribe_audio_array(dummy)

        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                device=self.device_index,
                channels=config.CHANNELS,
                dtype=config.AUDIO_DTYPE,
                blocksize=self.block_size,
                callback=self._audio_callback,
            ):
                while True:
                    try:
                        block, rms = self.audio_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    current_time = time.time()
                    
                    # Adaptive speech threshold
                    active_threshold = max(config.VAD_ENERGY_THRESHOLD, self.noise_floor * 2.8)
                    is_block_speech = rms > active_threshold

                    if not self.is_speaking:
                        # Smoothly track ambient room noise
                        self.noise_floor = 0.96 * self.noise_floor + 0.04 * rms
                        self.pre_buffer.append(block)

                        if is_block_speech:
                            self.consecutive_speech_blocks += 1
                        else:
                            self.consecutive_speech_blocks = 0

                        # Trigger speech once 3 consecutive speech blocks detected (~192ms)
                        if self.consecutive_speech_blocks >= self.required_speech_blocks:
                            self.is_speaking = True
                            self.speech_start_time = current_time
                            self.silence_start_time = 0.0
                            self.last_live_infer_time = current_time
                            self.speech_buffer = list(self.pre_buffer) + [block]
                            self.pre_buffer.clear()
                            self.consecutive_speech_blocks = 0

                    else:
                        # Active speech accumulation
                        self.speech_buffer.append(block)
                        speech_duration = (len(self.speech_buffer) * self.block_size) / self.sample_rate

                        if is_block_speech:
                            self.silence_start_time = 0.0
                        else:
                            if self.silence_start_time == 0.0:
                                self.silence_start_time = current_time

                        silence_duration = (current_time - self.silence_start_time) if self.silence_start_time > 0.0 else 0.0
                        should_commit = (silence_duration >= config.VAD_SILENCE_DURATION) or (speech_duration >= config.MAX_UTTERANCE_SECONDS)

                        if should_commit:
                            # Phrase finished -> Output ONLY finalized text if valid
                            full_audio = np.concatenate(self.speech_buffer, axis=0) if self.speech_buffer else np.array([], dtype=np.float32)
                            
                            if speech_duration >= config.VAD_MIN_SPEECH_DURATION and len(full_audio) > 0:
                                final_transcript = self._transcribe_audio_array(full_audio)
                                if final_transcript and len(final_transcript) > 1:
                                    # Output ONLY the transcribed words on their own line
                                    sys.stdout.write(f"\r\033[K{final_transcript}\n")
                                    sys.stdout.flush()
                                else:
                                    sys.stdout.write("\r\033[K")
                                    sys.stdout.flush()
                            else:
                                sys.stdout.write("\r\033[K")
                                sys.stdout.flush()

                            self.speech_buffer = []
                            self.is_speaking = False
                            self.silence_start_time = 0.0

                        else:
                            # Real-time live text preview at cursor
                            if (current_time - self.last_live_infer_time) >= config.LIVE_UPDATE_INTERVAL and speech_duration >= config.VAD_MIN_SPEECH_DURATION:
                                self.last_live_infer_time = current_time
                                partial_audio = np.concatenate(self.speech_buffer, axis=0)
                                partial_transcript = self._transcribe_audio_array(partial_audio)
                                if partial_transcript:
                                    sys.stdout.write(f"\r\033[K{partial_transcript}")
                                    sys.stdout.flush()

        except KeyboardInterrupt:
            sys.stdout.write("\r\033[K\n")
            sys.stdout.flush()


def run_live_streaming(device_index: int = None):
    """Entry point for starting clean live streaming transcription."""
    if device_index is None:
        dev_idx, dev_name = get_laptop_microphone()
    else:
        dev_idx = device_index
        dev_name = clean_device_name(sd.query_devices(dev_idx)["name"])
        
    if dev_idx is None:
        return

    # Load Whisper Large-v3-Turbo silently
    processor, model, device, torch_dtype = load_whisper_turbo(verbose=False)

    # Start Live Streamer
    streamer = WhisperLiveStreamer(
        processor=processor,
        model=model,
        device=device,
        torch_dtype=torch_dtype,
        device_index=dev_idx,
    )
    streamer.start_streaming()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Whisper Large-v3-Turbo Live Streaming Transcription")
    parser.add_argument("--device-index", "-d", type=int, default=None, help="Audio input device index")
    args = parser.parse_args()

    run_live_streaming(device_index=args.device_index)
