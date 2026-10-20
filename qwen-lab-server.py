from __future__ import annotations

import gc
import io
import json
import os
import threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

HOST = "127.0.0.1"
PORT = int(os.environ.get("QWEN_LAB_PORT", "8765"))
ROOT = Path(__file__).resolve().parent

MODEL_IDS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
}
ALLOWED_SPEAKERS = {
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"
}

if torch.cuda.is_available():
    DEVICE = "cuda:0"
    DTYPE = torch.bfloat16
else:
    DEVICE = "cpu"
    DTYPE = torch.float32

_model = None
_loaded_key: str | None = None
_lock = threading.Lock()


def json_bytes(data: dict) -> bytes:
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def unload_model() -> None:
    global _model, _loaded_key
    _model = None
    _loaded_key = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_model(key: str):
    global _model, _loaded_key
    if key not in MODEL_IDS:
        raise ValueError("Unsupported model selection")
    if _model is not None and _loaded_key == key:
        return _model

    unload_model()
    model_id = MODEL_IDS[key]
    print(f"Loading {model_id} on {DEVICE} ({DTYPE}) ...")
    _model = Qwen3TTSModel.from_pretrained(
        model_id,
        device_map=DEVICE,
        dtype=DTYPE,
    )
    _loaded_key = key
    print(f"Loaded {model_id}")
    return _model


class Handler(BaseHTTPRequestHandler):
    server_version = "QwenPronunciationLab/1.0"

    def send_json(self, status: int, payload: dict) -> None:
        body = json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/qwen-lab.html"):
            html = (ROOT / "qwen-lab.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(html)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(html)
            return

        if self.path == "/api/status":
            self.send_json(200, {
                "ok": True,
                "device": DEVICE,
                "loaded_model": _loaded_key,
            })
            return

        self.send_json(404, {"error": "Not found"})

    def do_POST(self):
        if self.path != "/api/tts":
            self.send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 100_000:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(length).decode("utf-8"))

            text = str(data.get("text", "")).strip()
            model_key = str(data.get("model", "0.6B"))
            speaker = str(data.get("speaker", "Aiden"))
            instruct = str(data.get("instruct", "")).strip()

            if not text or len(text) > 1000:
                raise ValueError("Text must be between 1 and 1000 characters")
            if model_key not in MODEL_IDS:
                raise ValueError("Unsupported model selection")
            if speaker not in ALLOWED_SPEAKERS:
                raise ValueError("Unsupported speaker")

            with _lock:
                model = load_model(model_key)
                wavs, sr = model.generate_custom_voice(
                    text=text,
                    language="English",
                    speaker=speaker,
                    instruct=instruct or None,
                    max_new_tokens=1024,
                )

            output = io.BytesIO()
            sf.write(output, wavs[0], sr, format="WAV", subtype="PCM_16")
            audio = output.getvalue()

            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(audio)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(audio)

        except Exception as exc:
            print(f"Generation error: {type(exc).__name__}: {exc}")
            self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt, *args):
        print(f"[{self.log_date_time_string()}] {fmt % args}")


if __name__ == "__main__":
    print(f"Qwen3-TTS Lab: http://{HOST}:{PORT}")
    print(f"Device: {DEVICE}")
    if DEVICE == "cpu":
        print("CUDA GPU not detected. Generation will work but may be slow on CPU.")
    print("First generation downloads the selected model from Hugging Face and then caches it locally.")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()
