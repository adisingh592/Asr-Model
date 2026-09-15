import asyncio
import socket
import sounddevice as sd
import queue
import time
import sys

# Create a queue to hold audio chunks
q = queue.Queue()

# This function grabs audio from your laptop mic
def callback(indata, frames, time_info, status):
    if status:
        print(f"[Audio Status] {status}", file=sys.stderr)
    q.put(bytes(indata))

async def test():
    host = "127.0.0.1"
    port = 8766
    print(f"Connecting to TCP ASR server at {host}:{port}...")
    try:
        reader, writer = await asyncio.open_connection(host, port)
        print("[+] Connected to Server!")
        
        while True:
            await asyncio.to_thread(input, "\nPress ENTER to stream 5 seconds of laptop microphone audio (or Ctrl+C to quit)...")
            
            print("Recording and streaming... Speak now (Say 'Khus Khus' to wake)!")
            
            # Clear any old audio in the queue
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
            
            # Open a 16kHz microphone stream
            with sd.RawInputStream(samplerate=16000, blocksize=1024, dtype='int16',
                                   channels=1, callback=callback):
                start_time = time.time()
                while time.time() - start_time < 5.0: # Record for 5 seconds
                    data = await asyncio.to_thread(q.get)
                    writer.write(data)
                    await writer.drain()
                    
            print("[+] Finished sending audio chunk. Check server.py terminal for transcripts!")
            
    except ConnectionRefusedError:
        print(f"[Error] Could not connect to {host}:{port}. Make sure 'python server.py' is running!")
    except (asyncio.CancelledError, KeyboardInterrupt):
        print("\nExiting.")
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:
            pass

if __name__ == "__main__":
    try:
        asyncio.run(test())
    except KeyboardInterrupt:
        print("\nExited.")

