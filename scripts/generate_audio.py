from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT = Path("audio/yijun-liu.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

# American-English voice, with explicit phoneme hints so the pronunciation is stable.
# Yijun is intentionally approximated as "ee-June" for English-speaking listeners.
# Mandarin Liu is closer to one syllable /ljoʊ/ than to English "Leo" /ˈliːoʊ/.
text = "[Yijun](/iː dʒuːn/) [Liu](/ljoʊ/)"

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
