"""명령 하나로 실행: vision.py는 별도 프로세스로 띄우고, assistant는 이 터미널에서 질문을 받음
(내부적으로는 두 프로세스라 Gemma가 답을 생각하는 동안에도 카메라는 멈추지 않음)

실행: 저장소 루트(src 폴더가 있는 위치)에서 python Project02/run.py
종료: assistant 터미널에서 q (vision도 같이 종료)
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    vision = subprocess.Popen([sys.executable, os.path.join(HERE, "vision.py")])
    time.sleep(1.0)                          # 카메라를 먼저 열고 나서 Gemma 로드 시작

    sys.path.insert(0, HERE)
    import assistant                         # 이 import에서 Gemma를 메모리에 올림
    assistant.main()

    vision.terminate()
    vision.wait()
    print("[run] 종료", flush=True)
    os._exit(0)                              # Gemma(llama.cpp)·CUDA 정리 중 멈추는 경우가 있어 바로 종료


if __name__ == "__main__":
    main()
