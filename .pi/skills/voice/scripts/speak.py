#!/usr/bin/env python3
"""Speak text aloud using edge-tts. Usage: python3 speak.py "text" [voice]"""

import asyncio
import os
import sys
import tempfile

import edge_tts

# ── Available English voices ────────────────────────────────────────────
# US voices
#   en-US-AndrewNeural                  Male      ← current default
#   en-US-AndrewMultilingualNeural      Male
#   en-US-AvaNeural                     Female
#   en-US-AvaMultilingualNeural         Female
#   en-US-EmmaNeural                    Female
#   en-US-EmmaMultilingualNeural        Female
#   en-US-BrianNeural                   Male
#   en-US-BrianMultilingualNeural       Male
#   en-US-AriaNeural                    Female
#   en-US-JennyNeural                   Female
#   en-US-MichelleNeural                Female
#   en-US-ChristopherNeural             Male
#   en-US-EricNeural                    Male
#   en-US-GuyNeural                     Male
#   en-US-RogerNeural                   Male
#   en-US-SteffanNeural                 Male
#   en-US-AnaNeural                     Female
# UK voices
#   en-GB-RyanNeural                    Male
#   en-GB-SoniaNeural                   Female
#   en-GB-ThomasNeural                  Male
#   en-GB-LibbyNeural                   Female
#   en-GB-MaisieNeural                  Female
# Australian
#   en-AU-WilliamMultilingualNeural     Male
#   en-AU-NatashaNeural                 Female
# Canadian
#   en-CA-LiamNeural                    Male
#   en-CA-ClaraNeural                   Female
# Irish
#   en-IE-ConnorNeural                  Male
#   en-IE-EmilyNeural                   Female
# Indian
#   en-IN-PrabhatNeural                 Male
#   en-IN-NeerjaNeural                  Female
#   en-IN-NeerjaExpressiveNeural        Female
# Other
#   en-NZ-MitchellNeural               Male  (NZ)
#   en-NZ-MollyNeural                  Female (NZ)
#   en-SG-WayneNeural                  Male  (SG)
#   en-SG-LunaNeural                   Female (SG)
#   en-ZA-LukeNeural                   Male  (ZA)
#   en-ZA-LeahNeural                   Female (ZA)
#   en-HK-SamNeural                    Male  (HK)
#   en-HK-YanNeural                    Female (HK)
#   en-KE-ChilembaNeural               Male  (KE)
#   en-KE-AsiliaNeural                 Female (KE)
#   en-NG-AbeoNeural                   Male  (NG)
#   en-NG-EzinneNeural                 Female (NG)
#   en-PH-JamesNeural                  Male  (PH)
#   en-PH-RosaNeural                   Female (PH)
#   en-TZ-ElimuNeural                  Male  (TZ)
#   en-TZ-ImaniNeural                  Female (TZ)
# ────────────────────────────────────────────────────────────────────────
DEFAULT_VOICE = "en-GB-RyanNeural"


async def speak(text: str, voice: str = DEFAULT_VOICE) -> None:
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        tmp = f.name
    try:
        await edge_tts.Communicate(text, voice).save(tmp)
        os.system(f"afplay {tmp}")
    finally:
        os.unlink(tmp)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: speak.py <text> [voice]")
        sys.exit(1)
    text = sys.argv[1]
    voice = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_VOICE
    asyncio.run(speak(text, voice))
