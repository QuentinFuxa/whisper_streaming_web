import io
import argparse
import asyncio
import numpy as np
import ffmpeg
from time import time, sleep
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from whisper_streaming_custom.whisper_online import backend_factory, online_factory, add_shared_args
from timed_objects import ASRToken

import math
import logging
from datetime import timedelta
import traceback

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


# Configure logging for all modules
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Set root logger level
logging.getLogger().setLevel(logging.WARNING)




# Configure main module logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)




##### LOAD ARGS #####

parser = argparse.ArgumentParser(description="Whisper FastAPI Online Server")
parser.add_argument(
    "--host",
    type=str,
    default="localhost",
    help="The host address to bind the server to.",
)
parser.add_argument(
    "--port", type=int, default=8000, help="The port number to bind the server to."
)
parser.add_argument(
    "--warmup-file",
    type=str,
    dest="warmup_file",
    help="The path to a speech audio wav file to warm up Whisper so that the very first chunk processing is fast. It can be e.g. https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav .",
)

parser.add_argument(
    "--confidence-validation",
    type=bool,
    default=True,
    help="Accelerates validation of tokens using confidence scores. Transcription will be faster but punctuation might be less accurate.",
)

parser.add_argument(
    "--diarization",
    type=bool,
    default=False,
    help="Whether to enable speaker diarization.",
)

parser.add_argument(
    "--transcription",
    type=bool,
    default=True,
    help="To disable to only see live diarization results.",
)

add_shared_args(parser)
args = parser.parse_args()


# Configure specific loggers for different modules
logging.getLogger("whisper_streaming_custom").setLevel(logging.getLevelName(args.log_level))
logging.getLogger("diarization").setLevel(logging.getLevelName(args.log_level))



SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLES_PER_SEC = SAMPLE_RATE * int(args.min_chunk_size)
BYTES_PER_SAMPLE = 2  # s16le = 2 bytes per sample
BYTES_PER_SEC = SAMPLES_PER_SEC * BYTES_PER_SAMPLE
MAX_BYTES_PER_SEC = 32000 * 5  # 5 seconds of audio at 32 kHz


class SharedState:
    def __init__(self):
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""  # To track changes in response
        
    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        async with self.lock:
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep
            
    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization
            
    async def add_dummy_token(self):
        async with self.lock:
            current_time = time() - self.beg_loop
            dummy_token = ASRToken(
                start=current_time,
                end=current_time + 0.5,
                text="",
                speaker=-1
            )
            self.tokens.append(dummy_token)
            
    async def get_current_state(self):
        async with self.lock:
            current_time = time()
            remaining_time_transcription = 0
            remaining_time_diarization = 0
            
            # Calculate remaining time for transcription buffer
            if self.end_buffer > 0:
                remaining_time_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))
                
            # Calculate remaining time for diarization
            remaining_time_diarization = max(0, round(max(self.end_buffer, self.tokens[-1].end if self.tokens else 0) - self.end_attributed_speaker, 2))
                
            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_time_transcription,
                "remaining_time_diarization": remaining_time_diarization
            }
            
    async def reset(self):
        """Reset the state."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = ""
            self.buffer_diarization = ""
            self.end_buffer = 0
            self.end_attributed_speaker = 0
            self.full_transcription = ""
            self.beg_loop = time()
            self.last_response_content = ""

##### LOAD APP #####

@asynccontextmanager
async def lifespan(app: FastAPI):
    global asr, tokenizer, diarization
    if args.transcription:
        asr, tokenizer = backend_factory(args)
    else:
        asr, tokenizer = None, None

    if args.diarization:
        from diarization.diarization_online import DiartDiarization
        diarization = DiartDiarization(SAMPLE_RATE)
    else :
        diarization = None
    yield

app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load demo HTML for the root endpoint
with open("web/live_transcription.html", "r", encoding="utf-8") as f:
    html = f.read()

async def start_ffmpeg_decoder():
    """
    Start an FFmpeg process in async streaming mode that reads WebM from stdin
    and outputs raw s16le PCM on stdout. Returns the process object.
    """
    process = (
        ffmpeg.input("pipe:0", format="webm")
        .output(
            "pipe:1",
            format="s16le",
            acodec="pcm_s16le",
            ac=CHANNELS,
            ar=str(SAMPLE_RATE),
        )
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )
    return process

async def transcription_processor(shared_state, pcm_queue, online):
    full_transcription = ""
    sep = online.asr.sep
    
    while True:
        try:
            pcm_array = await pcm_queue.get()
            
            logger.info(f"{len(online.audio_buffer) / online.SAMPLING_RATE} seconds of audio will be processed by the model.")
            
            # Process transcription
            online.insert_audio_chunk(pcm_array)
            new_tokens = online.process_iter()
            
            if new_tokens:
                full_transcription += new_tokens.get_text(sep=sep)
                
            _buffer = online.get_buffer()
            buffer = _buffer.text
            end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)
            
            if buffer in full_transcription:
                buffer = ""
                
            await shared_state.update_transcription(
                new_tokens, buffer, end_buffer, full_transcription, sep)
            
        except Exception as e:
            logger.warning(f"Exception in transcription_processor: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            pcm_queue.task_done()

async def diarization_processor(shared_state, pcm_queue, diarization_obj):
    buffer_diarization = ""
    
    while True:
        try:
            pcm_array = await pcm_queue.get()
            
            # Process diarization
            await diarization_obj.diarize(pcm_array)
            
            # Get current state
            state = await shared_state.get_current_state()
            tokens = state["tokens"]
            end_attributed_speaker = state["end_attributed_speaker"]
            
            # Update speaker information
            new_end_attributed_speaker = diarization_obj.assign_speakers_to_tokens(
                end_attributed_speaker, tokens)
            
            await shared_state.update_diarization(new_end_attributed_speaker, buffer_diarization)
            
        except Exception as e:
            logger.warning(f"Exception in diarization_processor: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
        finally:
            pcm_queue.task_done()

async def results_formatter(shared_state, websocket):
    while True:
        try:
            # Get the current state
            state = await shared_state.get_current_state()
            tokens = state["tokens"]
            buffer_transcription = state["buffer_transcription"]
            buffer_diarization = state["buffer_diarization"]
            end_attributed_speaker = state["end_attributed_speaker"]
            remaining_time_transcription = state["remaining_time_transcription"]
            remaining_time_diarization = state["remaining_time_diarization"]
            sep = state["sep"]
            
            # If diarization is enabled but no transcription, add dummy tokens periodically
            if not tokens and not args.transcription and args.diarization:
                await shared_state.add_dummy_token()
                # Re-fetch tokens after adding dummy
                state = await shared_state.get_current_state()
                tokens = state["tokens"]
            
            # Process tokens to create response
            previous_speaker = -10
            lines = []
            last_end_diarized = 0
            undiarized_text = []
            
            for token in tokens:
                speaker = token.speaker
                if args.diarization:
                    if (speaker == -1 or speaker == 0) and token.end >= end_attributed_speaker:
                        undiarized_text.append(token.text)
                        continue
                    elif (speaker == -1 or speaker == 0) and token.end < end_attributed_speaker:
                        speaker = previous_speaker
                    if speaker not in [-1, 0]:
                        last_end_diarized = max(token.end, last_end_diarized)

                if speaker != previous_speaker or not lines:
                    lines.append(
                        {
                            "speaker": speaker,
                            "text": token.text,
                            "beg": format_time(token.start),
                            "end": format_time(token.end),
                            "diff": round(token.end - last_end_diarized, 2)
                        }
                    )
                    previous_speaker = speaker
                elif token.text:  # Only append if text isn't empty
                    lines[-1]["text"] += sep + token.text
                    lines[-1]["end"] = format_time(token.end)
                    lines[-1]["diff"] = round(token.end - last_end_diarized, 2)
            
            if undiarized_text:
                combined_buffer_diarization = sep.join(undiarized_text)
                if buffer_transcription:
                    combined_buffer_diarization += sep
                await shared_state.update_diarization(end_attributed_speaker, combined_buffer_diarization)
                buffer_diarization = combined_buffer_diarization
                
            if lines:
                response = {
                    "lines": lines, 
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "remaining_time_transcription": remaining_time_transcription,
                    "remaining_time_diarization": remaining_time_diarization
                }
            else:
                response = {
                    "lines": [{
                        "speaker": 1,
                        "text": "",
                        "beg": format_time(0),
                        "end": format_time(tokens[-1].end) if tokens else format_time(0),
                        "diff": 0
                }],
                    "buffer_transcription": buffer_transcription,
                    "buffer_diarization": buffer_diarization,
                    "remaining_time_transcription": remaining_time_transcription,
                    "remaining_time_diarization": remaining_time_diarization

                }
            
            response_content = ' '.join([str(line['speaker']) + ' ' + line["text"] for line in lines]) + ' | ' + buffer_transcription + ' | ' + buffer_diarization
            
            if response_content != shared_state.last_response_content:
                if lines or buffer_transcription or buffer_diarization:
                    await websocket.send_json(response)
                    shared_state.last_response_content = response_content
            
            # Add a small delay to avoid overwhelming the client
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.warning(f"Exception in results_formatter: {e}")
            logger.warning(f"Traceback: {traceback.format_exc()}")
            await asyncio.sleep(0.5)  # Back off on error

##### ENDPOINTS #####

@app.get("/")
async def get():
    return HTMLResponse(html)

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket connection opened.")

    ffmpeg_process = None
    pcm_buffer = bytearray()
    shared_state = SharedState()
    
    transcription_queue = asyncio.Queue() if args.transcription else None
    diarization_queue = asyncio.Queue() if args.diarization else None
    
    online = None

    async def restart_ffmpeg():
        nonlocal ffmpeg_process, online, pcm_buffer
        if ffmpeg_process:
            try:
                ffmpeg_process.kill()
                await asyncio.get_event_loop().run_in_executor(None, ffmpeg_process.wait)
            except Exception as e:
                logger.warning(f"Error killing FFmpeg process: {e}")
        ffmpeg_process = await start_ffmpeg_decoder()
        pcm_buffer = bytearray()
        
        if args.transcription:
            online = online_factory(args, asr, tokenizer)
        
        await shared_state.reset()
        logger.info("FFmpeg process started.")

    await restart_ffmpeg()

    tasks = []    
    if args.transcription and online:
        tasks.append(asyncio.create_task(
            transcription_processor(shared_state, transcription_queue, online)))    
    if args.diarization and diarization:
        tasks.append(asyncio.create_task(
            diarization_processor(shared_state, diarization_queue, diarization)))
    formatter_task = asyncio.create_task(results_formatter(shared_state, websocket))
    tasks.append(formatter_task)

    async def ffmpeg_stdout_reader():
        nonlocal ffmpeg_process, pcm_buffer
        loop = asyncio.get_event_loop()
        beg = time()
        
        while True:
            try:
                elapsed_time = math.floor((time() - beg) * 10) / 10 # Round to 0.1 sec
                ffmpeg_buffer_from_duration = max(int(32000 * elapsed_time), 4096)
                beg = time()

                # Read chunk with timeout
                try:
                    chunk = await asyncio.wait_for(
                        loop.run_in_executor(
                            None, ffmpeg_process.stdout.read, ffmpeg_buffer_from_duration
                        ),
                        timeout=15.0
                    )
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg read timeout. Restarting...")
                    await restart_ffmpeg()
                    beg = time()
                    continue  # Skip processing and read from new process

                if not chunk:
                    logger.info("FFmpeg stdout closed.")
                    break
                pcm_buffer.extend(chunk)
                if len(pcm_buffer) >= BYTES_PER_SEC:
                    if len(pcm_buffer) > MAX_BYTES_PER_SEC:
                        logger.warning(
                            f"""Audio buffer is too large: {len(pcm_buffer) / BYTES_PER_SEC:.2f} seconds.
                            The model probably struggles to keep up. Consider using a smaller model.
                            """)
                    # Convert int16 -> float32
                    pcm_array = (
                        np.frombuffer(pcm_buffer[:MAX_BYTES_PER_SEC], dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    pcm_buffer = pcm_buffer[MAX_BYTES_PER_SEC:]
                    
                    if args.transcription and transcription_queue:
                        await transcription_queue.put(pcm_array.copy())
                    
                    if args.diarization and diarization_queue:
                        await diarization_queue.put(pcm_array.copy())
                    
                    if not args.transcription and not args.diarization:
                        await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                break

        logger.info("Exiting ffmpeg_stdout_reader...")

    stdout_reader_task = asyncio.create_task(ffmpeg_stdout_reader())
    tasks.append(stdout_reader_task)    
    try:
        while True:
            # Receive incoming WebM audio chunks from the client
            message = await websocket.receive_bytes()
            try:
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
            except (BrokenPipeError, AttributeError) as e:
                logger.warning(f"Error writing to FFmpeg: {e}. Restarting...")
                await restart_ffmpeg()
                ffmpeg_process.stdin.write(message)
                ffmpeg_process.stdin.flush()
    except WebSocketDisconnect:
        logger.warning("WebSocket disconnected.")
    finally:
        for task in tasks:
            task.cancel()
            
        try:
            await asyncio.gather(*tasks, return_exceptions=True)
            ffmpeg_process.stdin.close()
            ffmpeg_process.wait()
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
        
        if args.diarization and diarization:
            diarization.close()
        
        logger.info("WebSocket endpoint cleaned up.")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "whisper_fastapi_online_server:app", host=args.host, port=args.port, reload=True,
        log_level="info"
    )