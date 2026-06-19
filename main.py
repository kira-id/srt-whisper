import os
import tempfile
import datetime
import asyncio
import threading
from pathlib import Path

import ffmpeg
import torch
from fastapi import FastAPI, Form, File, UploadFile, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")
templates.env.cache = None

MODEL_CACHE_DIR = "./model_cache"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

model_cache = {}
processing_lock = threading.Lock()
cancel_event = threading.Event()
is_processing = False
processing_filename = None
processing_model_size = None
processing_language = None
processing_start_time = None
last_srt_path = None
last_original_filename = None
last_detected_lang = None
last_srt_lock = threading.Lock()


def format_timestamp(seconds: float) -> str:
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    milliseconds = int(secs * 1000) % 1000
    return f"{hours:02d}:{minutes:02d}:{int(secs):02d},{milliseconds:03d}"


def segments_to_srt(segments) -> str:
    srt_content = []
    entry_idx = 1

    for segment in segments:
        if segment.get("words"):
            for word, w_start, w_end in segment["words"]:
                clean_word = word.strip()
                if clean_word and len(clean_word) > 1:
                    srt_content.append(f"{entry_idx}")
                    srt_content.append(f"{format_timestamp(w_start)} --> {format_timestamp(w_end)}")
                    srt_content.append(f"{clean_word}")
                    srt_content.append("")
                    entry_idx += 1

    return "\n".join(srt_content)


def get_model(model_size: str, device: str = "cpu"):
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    compute_type = (
        "float16" if device == "cuda" else "int8" if device == "cpu" else "float32"
    )
    print(f"[srt-whisper] Using device: {device}, compute_type: {compute_type}, model: {model_size}")
    cache_key = f"{model_size}_{device}"
    if cache_key not in model_cache:
        model_cache[cache_key] = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root=MODEL_CACHE_DIR,
            local_files_only=False,
        )
    return model_cache[cache_key]


def do_transcribe(input_path, file_ext, audio_path, srt_path, model_size, language, cancel_event):
    audio_file = input_path
    if file_ext in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        ffmpeg.input(input_path).output(
            audio_path,
            ac=1,
            ar=16000,
            vn=None,
        ).run(quiet=True, overwrite_output=True)
        audio_file = audio_path

    model = get_model(model_size)

    segments, info = model.transcribe(
        audio_file,
        language=language,
        task="transcribe",
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 500},
        word_timestamps=True,
    )

    all_segments = []
    for segment in segments:
        if cancel_event.is_set():
            return None, "cancelled", info
        words = []
        if hasattr(segment, "words") and segment.words:
            words = [(word.word, word.start, word.end) for word in segment.words]
        all_segments.append(
            {
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "words": words,
            }
        )

    srt_content = segments_to_srt(all_segments)

    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt_content)

    return srt_path, "success", info


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.get("/status")
async def status():
    with processing_lock:
        status_data = {
            "is_processing": is_processing,
            "filename": processing_filename,
            "model_size": processing_model_size,
            "language": processing_language,
            "start_time": processing_start_time,
        }
    with last_srt_lock:
        if not is_processing and last_srt_path:
            status_data["download_available"] = True
            status_data["original_filename"] = last_original_filename
            status_data["detected_language"] = last_detected_lang
        else:
            status_data["download_available"] = False
    return JSONResponse(status_data)


@app.post("/cancel")
async def cancel():
    global is_processing, processing_filename, processing_model_size, processing_language, processing_start_time
    cancel_event.set()
    with processing_lock:
        is_processing = False
        processing_filename = None
        processing_model_size = None
        processing_language = None
        processing_start_time = None
    return JSONResponse({"status": "cancelled"})


@app.get("/download")
async def download_result():
    global last_srt_path, last_original_filename, last_detected_lang
    with last_srt_lock:
        if not last_srt_path or not os.path.exists(last_srt_path):
            return JSONResponse({"error": "No completed result available."}, status_code=404)
        filename = f"{Path(last_original_filename).stem}.srt"
        detected_lang = last_detected_lang or "unknown"
    return FileResponse(
        last_srt_path,
        media_type="application/octet-stream",
        filename=filename,
        headers={"X-Detected-Language": detected_lang},
    )


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    model_size: str = Form(default="medium"),
    language: str = Form(default="auto"),
):
    global is_processing, processing_filename, processing_model_size, processing_language, processing_start_time

    if language == "auto":
        language = None

    allowed_extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}
    file_ext = Path(file.filename).suffix.lower()

    if file_ext not in allowed_extensions:
        return JSONResponse(
            {"error": f"Unsupported file format: {file_ext}"},
            status_code=400,
        )

    with processing_lock:
        if is_processing:
            return JSONResponse(
                {"error": "Another transcription is currently in progress. Please wait."},
                status_code=409,
            )
        is_processing = True
        processing_filename = file.filename
        processing_model_size = model_size
        processing_language = language
        processing_start_time = datetime.datetime.now().isoformat()
        cancel_event.clear()

    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, f"input{file_ext}")
    audio_path = os.path.join(temp_dir, "audio.wav")
    srt_path = os.path.join(temp_dir, "output.srt")

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    try:
        loop = asyncio.get_running_loop()
        srt_file, status, info = await loop.run_in_executor(
            None,
            do_transcribe,
            input_path,
            file_ext,
            audio_path,
            srt_path,
            model_size,
            language,
            cancel_event,
        )

        if status == "cancelled":
            return JSONResponse({"status": "cancelled"}, status_code=200)

        detected_lang = info.language if info.language else "unknown"

        with last_srt_lock:
            global last_srt_path, last_original_filename, last_detected_lang
            last_srt_path = srt_path
            last_original_filename = file.filename
            last_detected_lang = detected_lang

        return FileResponse(
            srt_path,
            media_type="application/octet-stream",
            filename=f"{Path(file.filename).stem}.srt",
            headers={"X-Detected-Language": detected_lang},
        )
    finally:
        with processing_lock:
            is_processing = False
            processing_filename = None
            processing_model_size = None
            processing_language = None
            processing_start_time = None
            cancel_event.clear()


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="SRT Whisper API Server")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port)
