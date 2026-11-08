"""음성 입출력: Whisper STT, Piper TTS (06-C 수업 코드)"""
import re
import subprocess

from config import *


def speech_to_text():
    subprocess.run(
        [
            "pasuspender", "--",
            "arecord",
            "-D", MIC_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "1",
            "-d", str(RECORD_SECONDS),
            AUDIO_FILE,
        ],
        check=True,
    )

    result = subprocess.run(
        [
            WHISPER_PATH,
            "-m", WHISPER_MODEL,
            "-f", AUDIO_FILE,
            "-l", "ko",
            "--no-timestamps",
        ],
        text=True,
        capture_output=True,
        check=True,
    )

    return re.sub(r"\[.*?\]", "", result.stdout).strip()     # [BLANK_AUDIO] 같은 태그 제거


def text_to_speech(text):
    print(f"\n[응답] {text}\n", flush=True)
    if not USE_TTS:
        return

    subprocess.run(
        [
            PIPER_PYTHON,
            "-m",
            "piper",

            "-m",
            PIPER_MODEL,

            "-f",
            OUTPUT_FILE,

            "--",
            text,
        ],
        check=True,
    )

    subprocess.run(
        [
            "aplay",
            OUTPUT_FILE,
        ],
        check=True,
    )
