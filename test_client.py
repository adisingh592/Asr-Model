import asyncio
import socket
import sounddevice as sd
import queue
import sys

# Thread-safe queue for audio chunks from the sounddevice callback
audio_queue = queue.Queue()

def audio_callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status] {status}", file=sys.stderr)
    audio_queue.put(bytes(indata))

async def main():
    host = "127.0.0.1"
    port = 8766
    print(f"Connecting to TCP ASR server at {host}:{port}...")
    try:
        reader, writer = await asyncio.open_connection(host, port)
        print("[+] Connected! Start speaking into your laptop microphone (Say 'Khus Khus')...")
        print("[*] Streaming 16kHz 16-bit PCM audio (Press Ctrl+C to stop)...")
        
        # Open a 16kHz, 16-bit mono microphone stream matching ESP32
        with sd.RawInputStream(samplerate=16000, blocksize=1024, dtype='int16',
                               channels=1, callback=audio_callback):
            while True:
                # Fetch audio chunk asynchronously and send over TCP
                data = await asyncio.to_thread(audio_queue.get)
                writer.write(data)
                await writer.drain()

    except ConnectionRefusedError:
        print(f"\n[Error] Could not connect to {host}:{port}. Make sure 'python server.py' is running first!")
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nStopping client...")
    except Exception as e:
        print(f"\n[Notice] Connection closed: {e}")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nExited.")

