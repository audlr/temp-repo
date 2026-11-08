import numpy as np


# 수업(04 섹션)에서 만든 INT8 TensorRT 엔진 사용
YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"

STATE_JSON = "src/json/checkout_state.json"      # vision → assistant : 현재 물건 상태
EVENT_JSON = "src/json/checkout_event.json"      # vision → assistant : 체크아웃 결과
CONTROL_JSON = "src/json/checkout_control.json"  # assistant → vision : 예외 목록, 명령

# STT / TTS (06 섹션 설정 그대로)
MIC_DEVICE = "plughw:0,0"   # arecord -l 의 "card 번호, device 번호"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-base.bin"   # 수업 모델. tiny는 가볍지만 한국어 인식률이 많이 떨어짐
RECORD_SECONDS = 5

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"


# 수업 코드와 같은 카메라 설정
PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)


# Jetson 없이 PC에서 테스트할 때 False로 두면 키보드 입력 / 화면 출력으로 대체
USE_STT = True          # False: 음성 대신 키보드로 질문 입력
USE_TTS = True          # False: 음성 대신 터미널 출력만
USE_LLM = True          # False: Gemma 없이 키워드 규칙으로 의도 분석


YOLO_CONF = 0.3         # 사진 한 장 기준. 오탐은 STABLE_HITS(4장 중 3장)로 거르므로 높게 잡지 않음 (0.4는 침대 위 휴대폰 등을 놓침)
YOLO_IOU = 0.5          # YOLO 내부 NMS IoU Threshold
YOLO_IMGSZ = 640        # 입력 크기. TensorRT 엔진은 만들 때의 imgsz로 고정되므로 같은 값이어야 함 (수업 엔진: 기본값 640)
NMS_IOU = 0.6           # YOLO + 색 마커 결과를 합친 뒤의 클래스 무관 NMS

# 소지품으로 다룰 COCO 클래스
YOLO_ITEMS = [
    "cell phone", "laptop", "book", "bottle", "umbrella", "handbag", "backpack", "suitcase",
    "cup", "remote", "mouse", "toothbrush", "scissors",
]
# 작고 생김새가 단순해 오탐이 잦은 클래스는 더 높은 기준 적용 (실제 화면을 보며 조정)
CLASS_CONF = {"mouse": 0.7, "remote": 0.6, "toothbrush": 0.6, "scissors": 0.6}


# COCO에 없는 물건은 초록 스티커를 붙여 색으로 찾음. "name"에 스티커를 붙인 물건 이름을 적음
# LAB 범위는 Project01 초록 스티커 값. 조명에 맞게 튜닝 필요
MARKERS = {
    "green_marker": {"name": "여권",
                     "low": np.array([30, 60, 90], dtype=np.uint8),
                     "high": np.array([230, 115, 180], dtype=np.uint8)},
}
MARKER_MIN_AREA = 150   # 마커로 인정할 최소 윤곽선 면적


ITEM_NAMES_KO = {
    "cell phone": "휴대폰", "laptop": "노트북", "book": "책", "bottle": "물병",
    "umbrella": "우산", "handbag": "가방", "backpack": "배낭", "suitcase": "캐리어",
    "cup": "컵", "remote": "리모컨", "mouse": "마우스", "toothbrush": "칫솔", "scissors": "가위",
}
ITEM_NAMES_KO.update({key: marker["name"] for key, marker in MARKERS.items()})   # 마커 물건 이름


# 위치 설명: 가장 가까운 주변 물건을 기준으로 말함 ("노트북 왼쪽", "컵 위")
ON_TOP_RATIO = 0.6       # 물건 면적의 60% 이상이 기준 물건과 겹치면 "위" (예: 노트북 위의 휴대폰)
MIN_REF_RATIO = 0.5      # 크기가 절반도 안 되는 물건은 기준으로 쓰지 않음
NEAR_RATIO = 1.0         # 두 물건 사이 간격이 둘 중 큰 물건 크기 이내면 "왼쪽/오른쪽/위쪽/아래쪽", 더 멀면 화면 기준 위치


# 매 프레임 YOLO를 돌리지 않고, 일정 간격으로 찍은 사진만 처리 (물건은 거의 움직이지 않으므로 충분)
SNAPSHOT_SEC = 2.0      # 자동 사진 처리 간격(초). 질문/명령이 오면 즉시 한 장 더 처리
POLL_SEC = 0.2          # control.json 확인 간격(초)


MATCH_IOU = 0.3         # 이전 사진 물체와 같은 물체로 볼 최소 IoU
STABLE_WINDOW = 4       # 최근 N장 중
STABLE_HITS = 3         # M장 이상 보이면 "있음" 확정 (손에 잠깐 가려져도 유지)
MISSING_SEC = 4.5       # 확정된 물건이 이 시간(사진 2장 이상) 넘게 안 보이면 "챙김"
MOVE_PX = 40            # 물건 중심이 이 거리(px) 넘게 움직이면 "이동"으로 기록
ALERT_SEC = 6           # 남은 물건 경고 화면 표시 시간


BLUR_K, MORPH_K = 7, 5
OPEN_ITER, CLOSE_ITER = 2, 3


CONTEXT_WINDOW = 1024   # Nano: KV 캐시 메모리 절약
MAX_TOKENS_INTENT = 80
SNAPSHOT_WAIT_SEC = 4.0  # 질문 후 vision이 새 사진을 처리할 때까지 기다리는 최대 시간


GREEN, YELLOW, RED, WHITE, BLACK = (0, 200, 0), (0, 200, 255), (0, 0, 255), (255, 255, 255), (0, 0, 0)
GRAY, BLUE = (150, 150, 150), (255, 140, 0)
LINE, TEXT, TEXT_L, PAD = 2, 0.5, 0.6, 12
WIN = "Checkout Checker"
