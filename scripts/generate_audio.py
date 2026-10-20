from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT = Path("audio/yijun-liu.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

# American-English approximation intended to be easy for English speakers to copy.
# Yijun: /iˈdʒuːn/  -> stress the SECOND syllable ("June")
# Liu:   /ˈliːoʊ/   -> "Leo"
TEXT = "[Yijun](/iˈdʒuːn/) [Liu](/ˈliːoʊ/)"

pipeline = KPipeline(lang_code="a")
parts = []

# Generate slightly slower than normal at the model level so 1.0x playback
# remains clear and natural without post-processing slowdown.
for result in pipeline(TEXT, voice="af_heart", speed=0.90):
    if result.audio is not None:
        audio = result.audio.detach().cpu().numpy() if hasattr(result.audio, "detach") else np.asarray(result.audio)
        parts.append(audio)

if not parts:
    raise RuntimeError("Kokoro produced no audio")

audio = np.concatenate(parts)
sf.write(OUT, audio, 24000)
print(f"Wrote {OUT} ({len(audio)/24000:.2f}s)")
