# Video to SRT Subtitle Generator

A web application for generating detailed SRT subtitles from video or audio files using Faster Whisper.

## Features

- Upload video (MP4, AVI, MOV, MKV, WEBM) or audio (MP3, WAV, FLAC, OGG, M4A, AAC) files
- Word-by-word and sentence-by-sentence transcription
- Multiple model size options (tiny, base, small, medium, large)
- Language selection (Indonesian default, auto-detect, and 8 other languages)
- Real-time progress indicator during processing
- SRT file download

## Setup

```bash
pip install -r requirements.txt
python main.py
```

## Usage

1. Open `http://localhost:8000` in your browser
2. Upload a video or audio file
3. Select model size and language
4. Click "Generate SRT Subtitles"
5. Wait for processing (model is downloaded on first run)
6. Download the generated `.srt` file

## Notes

- First run will download the Whisper model (1-4GB depending on size)
- Models are cached in `./model_cache/` directory (see .gitignore)
- CUDA is used automatically if available, otherwise CPU
- VAD (Voice Activity Detection) is enabled for better segmentation
- Word timestamps enabled for detailed, word-per-word subtitles

## Model Cache

Downloaded models are stored in `model_cache/` and excluded from git via `.gitignore`.