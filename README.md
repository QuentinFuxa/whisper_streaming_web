# Whisper Streaming with FastAPI and WebSocket Integration

This project extends the [Whisper Streaming](https://github.com/ufal/whisper_streaming) implementation by incorporating few extras. The enhancements include:

1. **FastAPI Server with WebSocket Endpoint**: Real-time STT in browsers. Audio chunks processed via FFmpeg async streaming process.

2. **Buffering preview**: Enhances streaming feedback by displaying the unvalidated buffer content.

3. **Multiple users**: The backend can support multiple users simultaneously without conflicts.

4. **Javascript Client implementation**: MediaRecorder implementation that can be copied on your client side.

5. **MLX Whisper backend**: Integrates the alternative backend option MLX Whisper, optimized for efficient speech recognition on Apple silicon.

6. **Diarization (beta)**: Adds speaker labeling in real-time alongside transcription using the [Diart](https://github.com/juanmc2005/diart) library. Each transcription segment is tagged with a speaker.

![Demo Screenshot](src/web/demo.png)

##  Code Origins

This project reuses and extends code from the original Whisper Streaming repository:
- whisper_online.py, backends.py and online_asr.py: Contains code from whisper_streaming
- silero_vad_iterator.py: Originally from the Silero VAD repository, included in the whisper_streaming project.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/QuentinFuxa/whisper_streaming_web
   cd whisper_streaming_web
   ```
2. **Install dependencies using uv**:

    uv is a fast rust-based package manager and drop in replacement for pip. It is recommended to use it to install whisper_streaming dependencies.

    ```bash
    uv sync
    ```

It creates a virtual environment in .venv with all whisper backends.



### **Manual instalation of dependencies**:

- Install required dependences :

    ```bash
    # Whisper streaming required dependencies
    pip install librosa soundfile

    # Whisper streaming web required dependencies
    pip install fastapi ffmpeg-python
    ```
- Install at least one whisper backend among:

    ```
   whisper
   whisper-timestamped
   faster-whisper (faster backend on NVIDIA GPU)
   mlx-whisper (faster backend on Apple Silicon)
   ```
- Optionnal dependencies

    ```
    # If you want to use VAC (Voice Activity Controller). Useful for preventing hallucinations
    torch
   
    # If you choose sentences as buffer trimming strategy
    mosestokenizer
    wtpsplit
    tokenize_uk # If you work with Ukrainian text

    # If you want to run the server using uvicorn (recommended)
    uvicorn

    # If you want to use diarization
    diart
    ```

## How to Launch the Server

1. **Run the FastAPI Server**:

    ```bash
    uv run python whisper_fastapi_online_server.py --host 0.0.0.0 --port 8000
    ```

    - `--host` and `--port` let you specify the server’s IP/port. 
    - `-min-chunk-size` sets the minimum chunk size for audio processing. Make sure this value aligns with the chunk size selected in the frontend. If not aligned, the system will work but may unnecessarily over-process audio data.
    - For a full list of configurable options, run `python whisper_fastapi_online_server.py -h`
    - `--diarization`, default to False, let you choose whether or not you want to run diarization in parallel
    - For other parameters, look at [whisper streaming](https://github.com/ufal/whisper_streaming) readme.

2. **Open the Provided HTML**:

    - By default, the server root endpoint `/` serves a simple `live_transcription.html` page.  
    - Open your browser at `http://localhost:8000` (or replace `localhost` and `8000` with whatever you specified).  
    - The page uses vanilla JavaScript and the WebSocket API to capture your microphone and stream audio to the server in real time.

### How the Live Interface Works

- Once you **allow microphone access**, the page records small chunks of audio using the **MediaRecorder** API in **webm/opus** format.  
- These chunks are sent over a **WebSocket** to the FastAPI endpoint at `/asr`.  
- The Python server decodes `.webm` chunks on the fly using **FFmpeg** and streams them into the **whisper streaming** implementation for transcription.  
- **Partial transcription** appears as soon as enough audio is processed. The “unvalidated” text is shown in **lighter or grey color** (i.e., an ‘aperçu’) to indicate it’s still buffered partial output. Once Whisper finalizes that segment, it’s displayed in normal text.  
- You can watch the transcription update in near real time, ideal for demos, prototyping, or quick debugging.

## Deploying to a Remote Server

If you want to **deploy** this setup:

1. **Host the FastAPI app** behind a production-grade HTTP(S) server (like **Uvicorn + Nginx** or Docker). If you use HTTPS, use "wss" instead of "ws" in WebSocket URL.
2. The **HTML/JS page** can be served by the same FastAPI app or a separate static host.  
3. Users open the page in **Chrome/Firefox** (any modern browser that supports MediaRecorder + WebSocket).  

No additional front-end libraries or frameworks are required. The WebSocket logic in `live_transcription.html` is minimal enough to adapt for your own custom UI or embed in other pages.

## Acknowledgments

This project builds upon the foundational work of the Whisper Streaming project. We extend our gratitude to the original authors for their contributions.

