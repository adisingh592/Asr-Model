"""
Model loading utility for Whisper Large-v3-Turbo.
Loads pretrained weights strictly in inference mode (no training, no gradients).
"""

import sys
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

try:
    from . import config
except ImportError:
    import config


def load_whisper_turbo(
    model_name: str = config.MODEL_NAME,
    device: str = config.DEVICE,
    torch_dtype: torch.dtype = config.TORCH_DTYPE,
    backend: str = config.MODEL_BACKEND,
    verbose: bool = True,
):
    """
    Unified model loader supporting both TensorFlow Lite (TFLite) and PyTorch backends.
    
    Args:
        model_name (str): Hugging Face model identifier (for processor).
        device (str): PyTorch target device ('cuda' or 'cpu').
        torch_dtype (torch.dtype): PyTorch weight precision.
        backend (str): Model backend ('tflite' or 'pytorch').
        verbose (bool): Whether to print loading diagnostics.
        
    Returns:
        tuple: (processor, model_or_engine, device_str, dtype)
    """
    # 1. Load Hugging Face Processor / Feature Extractor
    if verbose:
        print(f"[*] Initializing Audio Feature Processor for {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name)
    
    # 2. TFLite / Edge Engine Backend
    if backend.lower() in ["tflite", "onnx", "edge"]:
        try:
            from .tflite_engine import WhisperTFLiteEngine
        except ImportError:
            from tflite_engine import WhisperTFLiteEngine
        
        engine = WhisperTFLiteEngine(
            processor=processor,
        )
        if verbose:
            print("=" * 60)
            print("Loaded Whisper Large-v3-Turbo Edge Runtime Engine")
            print("=" * 60)
            print(f"Model Info    : {engine.model_info}")
            print(f"Backend       : {engine.backend_type.upper()} Multi-Threaded Engine")
            print("=" * 60 + "\n")
            
        return processor, engine, f"Edge ({engine.backend_type.upper()})", torch.float32

    # 3. PyTorch Backend (Fallback / Explicit)
    if verbose:
        print("=" * 60)
        print("Loading Pretrained Whisper Large-v3-Turbo Model (PyTorch)")
        print("=" * 60)
        print(f"Model ID       : {model_name}")
        print(f"Target Device  : {device.upper()}")
        print(f"Torch Dtype    : {torch_dtype}")
        print("[1/2] Loading Pretrained PyTorch Weights...")
        
    model_kwargs = {
        "dtype": torch_dtype,
        "low_cpu_mem_usage": True,
    }
    
    if config.USE_FLASH_ATTENTION_2 and device == "cuda":
        model_kwargs["attn_implementation"] = "flash_attention_2"
        
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name,
        **model_kwargs
    )
    
    model.to(device)
    
    if hasattr(model, "generation_config") and model.generation_config is not None:
        model.generation_config.max_length = None
        model.generation_config.forced_decoder_ids = None
        
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
        
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if device == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        device_display = f"CUDA ({gpu_name})"
    else:
        device_display = "CPU"
        
    if verbose:
        print("\n" + "-" * 60)
        print("MODEL LOADED SUCCESSFULLY (PyTorch Inference Mode)")
        print("-" * 60)
        print(f"Model Name           : {model_name}")
        print(f"Active Device        : {device_display}")
        print(f"Active Dtype         : {next(model.parameters()).dtype}")
        print(f"Total Parameters     : {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"Trainable Parameters : {trainable_params:,} (Weights Frozen)")
        print("=" * 60 + "\n")
        
    return processor, model, device_display, torch_dtype



if __name__ == "__main__":
    print("Whisper Large-v3-Turbo Model Loader Module.")
    print("This module defines load_whisper_turbo().")
    print("To run tests, manually execute: python src/live_mic_test.py")
