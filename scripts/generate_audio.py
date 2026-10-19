from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT = Path("audio/yijun-liu.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Use an English-friendly approximation for the name in the public page/audio.
# Yijun -> "ee-June"; Liu -> "Leo" for immediate readability to English speakers.
text = "[Yijun](/iː dʒuːn/) [Liu](/ˈliːoʊ/)"

pipeline = KPipeline(lang_code="a")
parts = []
for result in pipeline(text, voice="af_heart", speed=0.92):
    if result.audio is not None:
        audio = result.audio.detach().cpu().numpy() if hasattr(result.audio, "detach") else np.asarray(result.audio)
        parts.append(audio)

if not parts:
    raise RuntimeError("Kokoro produced no audio")

audio = np.concatenate(parts)
sf.write(OUT, audio, 24000)
print(f"Wrote {OUT} ({len(audio)/24000:.2f}s)")
