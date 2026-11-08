"""음성 입출력: Whisper STT, Piper TTS (06-C)
실패하면 원인(명령의 오류 메시지)을 그대로 보여 줌"""
import os
import re
import shutil
import subprocess

from config import *


_mic = None


def run(command):
    """명령 실행. 실패하면 그 명령이 낸 오류 메시지를 담아 RuntimeError"""
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout).strip().splitlines()
        raise RuntimeError(f"{os.path.basename(command[0])} 실패: {' / '.join(message[-3:])}")
    return result.stdout


def find_mic():
    """MIC_DEVICE가 "auto"면 arecord -l 목록에서 USB 마이크를 찾아 "plughw:카드,장치"로 반환"""
    global _mic
    if MIC_DEVICE != "auto":
        return MIC_DEVICE
    if _mic:
        return _mic

    listing = run(["arecord", "-l"])
    devices = re.findall(r"^card (\d+): .*?, device (\d+): (.*)$", listing, re.M)
    if not devices:
        raise RuntimeError("녹음 장치를 찾을 수 없습니다. 마이크 연결과 'arecord -l' 결과를 확인하세요.")
    usb = [d for d in devices if "USB" in d[2]] or devices
    card, device, name = usb[0]
    _mic = f"plughw:{card},{device}"
    print(f"[마이크] {_mic} ({name.strip()})")
    return _mic


def check_audio():
    """시작할 때 음성 기능에 필요한 파일·명령이 있는지 점검해서 문제 목록 반환"""
    problems = []
    if USE_STT:
        for path in (WHISPER_PATH, WHISPER_MODEL):
            if not os.path.exists(path):
                problems.append(f"STT 파일 없음: {path}")
        if not shutil.which("arecord"):
            problems.append("arecord 명령 없음 (alsa-utils)")
    if USE_TTS:
        for path in (PIPER_PYTHON, PIPER_MODEL):
            if not os.path.exists(path):
                problems.append(f"TTS 파일 없음: {path}")
        if not shutil.which("aplay"):
            problems.append("aplay 명령 없음 (alsa-utils)")
    return problems


def record():
    """마이크로 RECORD_SECONDS초 녹음. 수업처럼 pasuspender로 감싸고, 그게 안 되면 arecord만 실행"""
    os.makedirs(os.path.dirname(AUDIO_FILE), exist_ok=True)
    command = ["arecord", "-D", find_mic(), "-f", "S16_LE", "-r", "16000",
               "-c", "1", "-d", str(RECORD_SECONDS), AUDIO_FILE]
    if shutil.which("pasuspender"):
        try:
            run(["pasuspender", "--"] + command)
            return
        except RuntimeError as error:
            print(f"[마이크] pasuspender 실패, arecord로 다시 시도 ({error})")
    run(command)


def speech_to_text():
    record()
    text = run([WHISPER_PATH, "-m", WHISPER_MODEL, "-f", AUDIO_FILE, "-l", "ko", "--no-timestamps"])
    return re.sub(r"\[.*?\]", "", text).strip()     # [BLANK_AUDIO] 같은 태그 제거


def text_to_speech(text):
    print(f"\n[응답] {text}\n", flush=True)
    if not USE_TTS:
        return
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", OUTPUT_FILE, "--", text])
    run(["aplay", OUTPUT_FILE])
