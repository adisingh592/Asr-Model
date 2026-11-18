"""
TensorFlow Lite (TFLite) Inference Engine for Whisper Large-v3-Turbo.
Provides unified generate() method and autoregressive decoding loop on top of
exported TFLite Encoder and Decoder models.
"""

import collections
import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import torch

# Ensure src module is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config

# Try importing TFLite Runtime or TensorFlow Lite
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tflite_runtime.interpreter as tflite
        Interpreter = tflite.Interpreter
    except ImportError:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter


class WhisperTFLiteEngine:
    """
    Unified Edge Inference Engine for Whisper Large-v3-Turbo.
    Supports both TensorFlow Lite / LiteRT runtime and ONNX Runtime backends.
    Implements greedy autoregressive generation with anti-loop penalties.
    """
    
    def __init__(
        self,
        encoder_path: Path = None,
        decoder_path: Path = None,
        processor=None,
        num_threads: int = 4,
    ):
        self.processor = processor
        self.num_threads = num_threads
        
        # Check available model formats
        tflite_enc = Path(encoder_path) if encoder_path else config.TFLITE_ENCODER_PATH
        tflite_dec = Path(decoder_path) if decoder_path else config.TFLITE_DECODER_PATH
        
        onnx_enc = config.TFLITE_DIR / "whisper_turbo_encoder.onnx"
        onnx_dec = config.TFLITE_DIR / "whisper_turbo_decoder.onnx"
        
        if tflite_enc.exists() and tflite_dec.exists():
            self.backend_type = "tflite"
            self.encoder_interp = Interpreter(model_path=str(tflite_enc), num_threads=self.num_threads)
            self.encoder_interp.allocate_tensors()
            self.enc_input_details = self.encoder_interp.get_input_details()
            self.enc_output_details = self.encoder_interp.get_output_details()
            
            self.decoder_interp = Interpreter(model_path=str(tflite_dec), num_threads=self.num_threads)
            self.decoder_interp.allocate_tensors()
            self.dec_input_details = self.decoder_interp.get_input_details()
            self.dec_output_details = self.decoder_interp.get_output_details()
            self.model_info = f"TFLite ({tflite_enc.name})"
        elif onnx_enc.exists() and onnx_dec.exists():
            self.backend_type = "onnx"
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            opts.intra_op_num_threads = self.num_threads
            opts.inter_op_num_threads = self.num_threads
            
            self.encoder_sess = ort.InferenceSession(str(onnx_enc), sess_options=opts, providers=["CPUExecutionProvider"])
            self.decoder_sess = ort.InferenceSession(str(onnx_dec), sess_options=opts, providers=["CPUExecutionProvider"])
            self.model_info = f"ONNX Runtime Edge ({onnx_enc.name})"
        else:
            raise FileNotFoundError(
                f"No exported edge model files found in {config.TFLITE_DIR}.\n"
                f"Please run 'python src/export_tflite.py' to export the models."
            )
            
        # Special Tokens for Whisper Large-v3-Turbo
        self.sot_token = 50258        # <|startoftranscript|>
        self.eot_token = 50257        # <|endoftranscript|>
        self.en_token = 50259         # <|en|>
        self.transcribe_token = 50360 # <|transcribe|>
        self.notimestamps_token = 50364 # <|notimestamps|>

    def encode(self, input_features: np.ndarray) -> np.ndarray:
        """
        Runs Whisper Encoder on Log-Mel spectrogram features [1, 128, 3000].
        
        Returns:
            np.ndarray: Encoder hidden states [1, 1500, 1280].
        """
        if isinstance(input_features, torch.Tensor):
            input_features = input_features.cpu().numpy()
            
        if input_features.dtype != np.float32:
            input_features = input_features.astype(np.float32)
            
        if self.backend_type == "tflite":
            self.encoder_interp.set_tensor(self.enc_input_details[0]["index"], input_features)
            self.encoder_interp.invoke()
            return self.encoder_interp.get_tensor(self.enc_output_details[0]["index"])
        else:
            outputs = self.encoder_sess.run(None, {"input_features": input_features})
            return outputs[0]

    def decode_step(self, input_ids: np.ndarray, encoder_hidden_states: np.ndarray) -> np.ndarray:
        """
        Runs Whisper Decoder on token sequence and encoder hidden states.
        
        Returns:
            np.ndarray: Next token logits [vocab_size].
        """
        if self.backend_type == "tflite":
            self.decoder_interp.resize_tensor_input(self.dec_input_details[0]["index"], input_ids.shape)
            self.decoder_interp.allocate_tensors()
            self.decoder_interp.set_tensor(self.dec_input_details[0]["index"], input_ids.astype(np.int64))
            self.decoder_interp.set_tensor(self.dec_input_details[1]["index"], encoder_hidden_states.astype(np.float32))
            self.decoder_interp.invoke()
            logits = self.decoder_interp.get_tensor(self.dec_output_details[0]["index"])
            return logits[0, -1, :]
        else:
            outputs = self.decoder_sess.run(None, {
                "input_ids": input_ids.astype(np.int64),
                "encoder_hidden_states": encoder_hidden_states.astype(np.float32)
            })
            return outputs[0][0, -1, :]

    def generate(
        self,
        input_features,
        language: str = "en",
        task: str = "transcribe",
        max_new_tokens: int = config.MAX_NEW_TOKENS,
        no_repeat_ngram_size: int = config.NO_REPEAT_NGRAM_SIZE,
        repetition_penalty: float = config.REPETITION_PENALTY,
        **kwargs
    ):
        """
        Simulates Transformers model.generate() using the Edge runtime.
        
        Args:
            input_features: 1D or 3D Log-Mel spectrogram (PyTorch Tensor or NumPy array).
            language (str): Language code (default "en").
            task (str): Task ("transcribe").
            max_new_tokens (int): Maximum tokens to generate.
            no_repeat_ngram_size (int): Anti-repetition n-gram block size.
            repetition_penalty (float): Multiplier penalizing token repetition.
            
        Returns:
            torch.Tensor: Generated token IDs of shape [1, seq_len].
        """
        # 1. Run Encoder
        encoder_hidden_states = self.encode(input_features)
        
        # 2. Prepare initial prompt tokens: [<|startoftranscript|>, <|en|>, <|transcribe|>, <|notimestamps|>]
        tokens = [self.sot_token, self.en_token, self.transcribe_token, self.notimestamps_token]
        generated_tokens = []
        
        # 3. Autoregressive Greedy Decoding Loop
        for _ in range(max_new_tokens):
            input_ids = np.array([tokens], dtype=np.int64)
            logits = self.decode_step(input_ids, encoder_hidden_states)
            
            # Apply repetition penalty to previously generated tokens
            if repetition_penalty != 1.0 and generated_tokens:
                for tok in set(generated_tokens):
                    if logits[tok] > 0:
                        logits[tok] /= repetition_penalty
                    else:
                        logits[tok] *= repetition_penalty
                        
            # Apply no_repeat_ngram_size blocking
            if no_repeat_ngram_size > 0 and len(generated_tokens) >= no_repeat_ngram_size:
                ngram_prefix = tuple(generated_tokens[-(no_repeat_ngram_size - 1):])
                for i in range(len(generated_tokens) - no_repeat_ngram_size + 1):
                    if tuple(generated_tokens[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                        blocked_tok = generated_tokens[i + no_repeat_ngram_size - 1]
                        logits[blocked_tok] = -float("inf")
                        
            # Select most probable token
            next_token = int(np.argmax(logits))
            
            # Check for End of Transcript
            if next_token == self.eot_token:
                break
                
            tokens.append(next_token)
            generated_tokens.append(next_token)
            
        # Return as 2D torch.Tensor for full compatibility with processor.batch_decode()
        return torch.tensor([tokens], dtype=torch.long)


def load_whisper_tflite_engine(verbose: bool = True):
    """
    Factory function to initialize and return the Whisper TFLite / Edge engine.
    """
    engine = WhisperTFLiteEngine()
    if verbose:
        print("=" * 60)
        print("Loaded Whisper Large-v3-Turbo Edge Runtime Engine")
        print("=" * 60)
        print(f"Model Info    : {engine.model_info}")
        print(f"Backend       : {engine.backend_type.upper()} Multi-Threaded Engine")
        print("=" * 60 + "\n")
    return engine
