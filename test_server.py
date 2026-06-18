from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from faster_whisper import WhisperModel
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory='templates')
templates.env.cache = None

model_cache = {}

def get_model(model_size, device):
    cache_key = f"{model_size}_{device}"
    if cache_key not in model_cache:
        model_cache[cache_key] = WhisperModel(model_size, device=device, local_files_only=True)
    return model_cache[cache_key]

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="error")
