"""명령 하나로 실행: vision.py는 별도 프로세스로 띄우고, assistant는 이 터미널에서 질문을 받음
(내부적으로는 두 프로세스라 Gemma가 답을 생각하는 동안에도 카메라는 멈추지 않음)

실행: 저장소 루트(src 폴더가 있는 위치)에서 python Project02/run.py
종료: assistant 터미널에서 q (vision도 같이 종료)
"""
import os
import sys
import time
import threading
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
stopping = threading.Event()                     # run.py가 직접 끄는 중인지 (종료 안내 중복 방지)


def start_vision():
    """vision.py를 같은 파이썬, 같은 작업 폴더(src 경로 기준)로 실행"""
    return subprocess.Popen([sys.executable, os.path.join(HERE, "vision.py")])


def watch_vision(proc):
    """vision이 먼저 꺼지면(카메라 창에서 q, 카메라 오류 등) 알려 줌"""
    proc.wait()
    if stopping.is_set():
        return
    print(f"\n[run] vision이 종료되었습니다 (코드 {proc.returncode}). assistant도 끄려면 q를 입력하세요.")


def stop_vision(proc):
    stopping.set()
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def main():
    proc = start_vision()
    threading.Thread(target=watch_vision, args=(proc,), daemon=True).start()
    try:
        time.sleep(1.0)                          # 카메라를 먼저 열고 나서 Gemma 로드 시작
        sys.path.insert(0, HERE)
        import assistant                         # 이 import에서 Gemma를 메모리에 올림
        assistant.main()
    except KeyboardInterrupt:
        pass
    finally:
        stop_vision(proc)
        print("[run] 종료", flush=True)
        os._exit(0)                              # Gemma(llama.cpp)·CUDA 정리 중 멈추는 경우가 있어 바로 종료


if __name__ == "__main__":
    main()
