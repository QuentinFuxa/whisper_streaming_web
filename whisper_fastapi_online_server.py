import io
import argparse
import asyncio
import numpy as np
import ffmpeg
from time import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisper_online import asr_factory, add_shared_args

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument("--host", type=str, default='localhost', help="The host address to bind the server to.")
parser.add_argument("--port", type=int, default=8000, help="The port number to bind the server to.")
parser.add_argument("--warmup-file", type=str, dest="warmup_file", 
        help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .")
add_shared_args(parser)
args = parser.parse_args()

asr, online = asr_factory(args)

# Load demo HTML for the root endpoint
with open("src/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()

@app.get("/")
async def get():
    return HTMLResponse(html)

SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2               # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE

async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg
        .input('pipe:0', format='webm')
        .output('pipe:1', format='s16le', acodec='pcm_s16le', ac=CHANNELS, ar=str(SAMPLE_RATE))
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connection opened.")

    ffmpeg_process = await start_ffmpeg_decoder()
    pcm_buffer = bytearray()
    # Continuously read decoded PCM from ffmpeg stdout in a background task
    async def ffmpeg_stdout_reader():
        nonlocal pcm_buffer
        loop = asyncio.get_event_loop()
        full_transcription = ""
        beg = time()
        while True:
            try:
                elapsed_time = int(time() - beg)
                beg = time()
                chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, 32000*elapsed_time)
                if not chunk: # The first chunk will be almost empty, FFmpeg is still starting up
                    chunk = await loop.run_in_executor(None, ffmpeg_process.stdout.read, 4096)
                    if not chunk: # FFmpeg might have closed
                        print("FFmpeg stdout closed.")
                        break

                pcm_buffer.extend(chunk)

                if len(pcm_buffer) >= BYTES_PER_SEC:
                    # Convert int16 -> float32
                    pcm_array = np.frombuffer(pcm_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                    pcm_buffer = bytearray()
                    online.insert_audio_chunk(pcm_array)
                    transcription = online.process_iter()[2]
                    if args.vac:
                        buffer = online.online.to_flush(online.online.transcript_buffer.buffer)[2] # We need to access the underlying online object to get the buffer
                    else:
                        buffer = online.to_flush(online.transcript_buffer.buffer)[2]
                    if buffer in full_transcription: # With VAC, the buffer is not updated until the next chunk is processed
                        buffer = ""
                    await websocket.send_json({
                        "transcription": transcription,
                        "buffer": buffer
                    })
            except Exception as e:
                print(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        print("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())

    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            # Pass them to ffmpeg via stdin
            ffmpeg_process.stdin.write(message)
            ffmpeg_process.stdin.flush()

    except WebSocketDisconnect:
        print("WebSocket connection closed.")
    except Exception as e:
        print(f"Error in websocket loop: {e}")
    finally:
        # Clean up ffmpeg and the reader task
        try:
            ffmpeg_process.stdin.close()
        except:
            pass
        stdout_reader_task.cancel()

        try:
            ffmpeg_process.stdout.close()
        except:
            pass

        ffmpeg_process.wait()
        
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True)
