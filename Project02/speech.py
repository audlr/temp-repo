"""음성 입출력: Whisper STT, Piper TTS (06-C)"""
import re
import subprocess

from config import *


_mic = None


def find_mic():
    """MIC_DEVICE가 "auto"면 arecord -l 목록에서 USB 마이크를 찾아 "plughw:카드,장치"로 반환"""
    global _mic
    if MIC_DEVICE != "auto":
        return MIC_DEVICE
    if _mic:
        return _mic

    listing = subprocess.run(["arecord", "-l"], text=True, capture_output=True).stdout
    devices = re.findall(r"^card (\d+): .*?, device (\d+): (.*)$", listing, re.M)
    if not devices:
        raise RuntimeError("녹음 장치를 찾을 수 없습니다. 마이크 연결과 'arecord -l' 결과를 확인하세요.")
    usb = [d for d in devices if "USB" in d[2]] or devices
    card, device, name = usb[0]
    _mic = f"plughw:{card},{device}"
    print(f"[마이크] {_mic} ({name.strip()})")
    return _mic


def speech_to_text():
    subprocess.run(
        ["pasuspender", "--", "arecord", "-D", find_mic(), "-f", "S16_LE", "-r", "16000",
         "-c", "1", "-d", str(RECORD_SECONDS), AUDIO_FILE],
        check=True,
    )
    result = subprocess.run(
        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", AUDIO_FILE, "-l", "ko", "--no-timestamps"],
        text=True, capture_output=True, check=True,
    )
    return re.sub(r"\[.*?\]", "", result.stdout).strip()


def text_to_speech(text):
    print(f"\n[응답] {text}\n")
    if not USE_TTS:
        return
    subprocess.run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", OUTPUT_FILE, "--", text], check=True)
    subprocess.run(["aplay", OUTPUT_FILE], check=True)
