from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT_DIR = Path("audio")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate two spelling variants so they can be compared directly:
# 1) "Yijun Leo"
# 2) "Yi-Jun Leo"
# The surname is intentionally rendered as the English-friendly approximation "Leo".
VARIANTS = {
    "yijun-leo.wav": "Yijun Leo",
    "yi-jun-leo.wav": "Yi-Jun Leo",
}

pipeline = KPipeline(lang_code="a")

for filename, text in VARIANTS.items():
    parts = []
    # Generate somewhat slowly at the model level to keep the voice natural.
    for result in pipeline(text, voice="af_heart", speed=0.82):
        if result.audio is not None:
            audio = result.audio.detach().cpu().numpy() if hasattr(result.audio, "detach") else np.asarray(result.audio)
            parts.append(audio)

    if not parts:
        raise RuntimeError(f"Kokoro produced no audio for {text}")

    audio = np.concatenate(parts)
    out = OUT_DIR / filename
    sf.write(out, audio, 24000)
    print(f"Wrote {out} ({len(audio)/24000:.2f}s) from: {text}")
