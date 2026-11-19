"""
Laptop microphone client that stands in for the ESP32-S3.
Streams 16 kHz 16-bit mono PCM over TCP to server.py, exactly like the firmware does.

Many laptop sound cards cannot capture at 16 kHz directly, so the mic is opened at a
supported multiple of 16 kHz (e.g. 48 kHz) and low-pass filtered + decimated here.
"""

import argparse
import asyncio
import queue
import sys

import numpy as np
from scipy.signal import firwin, lfilter, lfilter_zi
import sounddevice as sd

TARGET_RATE = 16000
CHUNK_SAMPLES = 512          # 1024 bytes per TCP write, same as the server's read size
CANDIDATE_RATES = (16000, 48000, 32000, 96000)

# Thread-safe queue for audio chunks from the sounddevice callback
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
    """Streaming anti-alias FIR + integer decimation (filter state carried across blocks)."""

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
    if status:
        print(f"[Audio Status] {status}", file=sys.stderr)
    audio_queue.put(indata[:, 0].copy())


async def main(args):
    if args.device is None:
        device, rate = pick_input_device()
        if device is None:
            print("[Error] No working microphone found. Run with --list and pick one using --device.")
            return
    else:
        device, rate = args.device, supported_rate(args.device)
        if rate is None:
            print(f"[Error] Device {args.device} supports none of {CANDIDATE_RATES} Hz.")
            return

    decimate = Decimator(rate // TARGET_RATE)
    print(f"[*] Microphone: [{device}] {sd.query_devices(device)['name']} @ {rate} Hz -> {TARGET_RATE} Hz")
    print(f"Connecting to TCP ASR server at {args.host}:{args.port}...")
    writer = None
    try:
        reader, writer = await asyncio.open_connection(args.host, args.port)
        print("[+] Connected! Start speaking into your laptop microphone (Say 'Khus Khus')...")
        print("[*] Streaming 16kHz 16-bit PCM audio (Press Ctrl+C to stop)...")

        with sd.InputStream(samplerate=rate, blocksize=CHUNK_SAMPLES * decimate.factor, dtype='int16',
                            channels=1, device=device, callback=audio_callback):
            while True:
                # Fetch audio chunk asynchronously, resample, and send over TCP
                block = await asyncio.to_thread(audio_queue.get)
                writer.write(decimate(block).tobytes())
                await writer.drain()

    except ConnectionRefusedError:
        print(f"\n[Error] Could not connect to {args.host}:{args.port}. Make sure 'python server.py' is running first!")
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nStopping client...")
    except Exception as e:
        print(f"\n[Notice] Connection closed: {e}")
    finally:
        if writer is not None:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream the laptop microphone to server.py like the ESP32 does")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--device", type=int, default=None, help="sounddevice input index (default: auto-detect)")
    parser.add_argument("--list", action="store_true", help="list audio devices and exit")
    args = parser.parse_args()
    if args.list:
        print(sd.query_devices())
        sys.exit(0)
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nExited.")
