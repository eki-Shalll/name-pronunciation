from pathlib import Path
import numpy as np
import soundfile as sf
from kokoro import KPipeline

OUT = Path("audio/yijun-liu.wav")
OUT.parent.mkdir(parents=True, exist_ok=True)

# Tone-stripped Mandarin segmental pronunciation for pinyin "Yi Jun Liu".
# Yi  ≈ /i/
# Jun ≈ /tɕyn/   (pinyin j is /tɕ/; u after j represents ü /y/)
# Liu ≈ /ljou/   (pinyin -iu is the contracted spelling of -iou)
# No lexical tone marks are included on purpose.
PHONEMES = "i tɕyn ljou"

# Use a Mandarin Kokoro voice so the Mandarin segments are realized naturally.
pipeline = KPipeline(lang_code="z")
voice_pack = pipeline.load_voice("zf_xiaoxiao")

# Generate slightly slowly at the model level. The webpage can still adjust playback
# speed, but 1.0x playback is the intended natural baseline for this file.
result = KPipeline.infer(pipeline.model, PHONEMES, voice_pack, speed=0.88)
audio = result.audio.detach().cpu().numpy() if hasattr(result.audio, "detach") else np.asarray(result.audio)

sf.write(OUT, audio, 24000)
print(f"Wrote {OUT} ({len(audio)/24000:.2f}s) from phonemes: {PHONEMES}")
