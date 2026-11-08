import numpy as np


# 수업(04 섹션)에서 만든 INT8 TensorRT 엔진 사용
YOLO_MODEL_PATH = "src/models/YOLO/yolo11n_int8.engine"
GEMMA_MODEL_PATH = "src/models/Gemma4/google_gemma-4-E2B-it-Q4_K_M.gguf"
MMPROJ_PATH = "src/models/Gemma4/mmproj-google_gemma-4-E2B-it-f16.gguf"
ROI_DIR = "src/output/checkout"                  # 모르는 물건 ROI 이미지 저장 위치
AWAY_DIR = "src/output/away"                     # 부재 모드 변화 사진 저장 위치

STATE_JSON = "src/json/checkout_state.json"      # vision → assistant : 현재 물건 상태
EVENT_JSON = "src/json/checkout_event.json"      # vision → assistant : 출발 감지, 새 물건 이벤트
CONTROL_JSON = "src/json/checkout_control.json"  # assistant → vision : 예외 목록, 명령, 물건 이름

# STT / TTS (06 섹션 설정 그대로)
MIC_DEVICE = "plughw:2,0"
AUDIO_FILE = "src/audio/input.wav"
WHISPER_PATH = "whisper.cpp/build-cpu/bin/whisper-cli"
WHISPER_MODEL = "whisper.cpp/models/ggml-tiny.bin"   # Nano: base 대신 tiny (메모리·속도). 인식률이 아쉬우면 base
RECORD_SECONDS = 5

PIPER_PYTHON = ".piper_venv/bin/python"
PIPER_MODEL = "src/models/Piper/ko_KR-kss-medium.onnx"
OUTPUT_FILE = "src/audio/response.wav"


PIPELINE = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw,format=YUY2,width=640,height=480,framerate=30/1 ! "
    "videoconvert ! video/x-raw,format=BGR ! "
    "queue leaky=downstream max-size-buffers=1 "
    "max-size-bytes=0 max-size-time=0 ! "
    "appsink drop=true max-buffers=1 sync=false"
)
USE_GSTREAMER = True    # False면 cv2.VideoCapture(0) (노트북 웹캠)
CAMERA_RETRY = 10       # 프레임을 이 횟수만큼 연속으로 못 읽으면 카메라를 다시 엶
CAMERA_REOPEN_MAX = 5   # 다시 열기를 이 횟수만큼 실패하면 종료


# Jetson 없이 PC에서 테스트할 때 False로 두면 키보드 입력 / 화면 출력으로 대체
USE_STT = True          # False: 음성 대신 키보드로 질문 입력
USE_TTS = True          # False: 음성 대신 터미널 출력만
USE_LLM = True          # False: Gemma 없이 키워드 규칙으로 의도 분석
USE_VISION_LLM = False  # 모르는 물건을 Gemma 이미지 입력(mmproj)으로 이름 붙이기. Nano는 메모리 측정 후 켜기. False: "물건 N번"


YOLO_CONF = 0.25        # Confidence Threshold
YOLO_IOU = 0.5          # YOLO 내부 NMS IoU Threshold
YOLO_IMGSZ = 640        # 입력 크기. TensorRT 엔진은 만들 때의 imgsz로 고정되므로 같은 값이어야 함 (수업 엔진: 기본값 640)
NMS_IOU = 0.6           # YOLO + 색 마커 결과를 합친 뒤의 클래스 무관 NMS

# 소지품으로 다룰 COCO 클래스
YOLO_ITEMS = [
    "cell phone", "laptop", "book", "bottle", "umbrella",
    "handbag", "cup", "remote", "mouse", "toothbrush", "scissors",
]
# 출발 판단용 (들고 나가는 가방)
CARRIERS = ["backpack", "suitcase"]


# 스티커 색 자체에는 의미가 없고, 스티커를 붙인 물건의 이름을 "name"에 적어 사용
# (COCO에 없으면서 두고 가면 가장 곤란한 물건에 붙이는 것을 추천: 여권, 지갑, 충전기 등)
# LAB 범위는 Project01 초록 스티커 값. 조명에 맞게 튜닝 필요
MARKERS = {
    "green_marker": {"name": "여권",
                     "low": np.array([30, 60, 90], dtype=np.uint8),
                     "high": np.array([230, 115, 180], dtype=np.uint8)},
}

MARKER_MIN_AREA = 150   # 마커로 인정할 최소 윤곽선 면적


ITEM_NAMES_KO = {
    "cell phone": "휴대폰", "laptop": "노트북", "book": "책", "bottle": "물병",
    "umbrella": "우산", "handbag": "가방", "cup": "컵", "remote": "리모컨",
    "mouse": "마우스", "toothbrush": "칫솔", "scissors": "가위",
    "backpack": "배낭", "suitcase": "캐리어",
    "unknown": "확인 중인 물건",
}
ITEM_NAMES_KO.update({key: marker["name"] for key, marker in MARKERS.items()})   # 마커 물건 이름


# 체크인 때 찍은 사진을 배경으로 저장하고, 이후 구역 안에 생긴 변화 중
# YOLO와 색 마커가 설명하지 못하는 영역을 "모르는 물건"으로 등록
USE_UNKNOWN = True
UNKNOWN_CLASS = "unknown"
DIFF_BLUR = 21          # 배경 차분 전 블러 커널 (작은 노이즈 무시)
DIFF_THRESHOLD = 35     # 배경과 밝기 차이가 이 값 이상이면 변화로 판단
UNKNOWN_MIN_AREA = 1500 # 모르는 물건으로 볼 최소 면적(px)
UNKNOWN_MAX_RATIO = 0.4 # 화면의 40% 이상이 변하면 조명 변화나 사람으로 보고 무시
KNOWN_MARGIN = 15       # 이미 아는 물건 BBox를 이만큼 넓혀서 제외 (그림자 등)
BG_ALPHA = 0.05         # 물체가 없는 영역의 배경 갱신 비율 (사진마다 5%, 약 40초에 걸쳐 조명 변화를 따라감)
ROI_PAD = 10            # ROI를 자를 때 여유


# 외출 중 내 물건이 사라지거나 새 물체가 생긴 순간만 사진으로 저장 (영상 녹화·녹음 없음)
AWAY_BUFFER = 4          # 변화 직전 사진도 남기기 위해 최근 N장을 메모리에 보관 (챙김 판정까지 사진 2~3장 지연)
KEEP_SEC = 24 * 3600     # 저장한 사진(부재 사진, 모르는 물건 ROI)은 24시간 뒤 자동 삭제
CLEANUP_SEC = 600        # 오래된 사진 삭제 확인 주기(초)


ZONES = [
    {"key": "bed_left", "name": "침대 왼쪽", "rect": (0, 140, 320, 480)},
    {"key": "bed_right", "name": "침대 오른쪽", "rect": (320, 140, 640, 480)},
    {"key": "nightstand", "name": "협탁", "rect": (0, 0, 640, 140)},
]
OUTSIDE_ZONE = {"key": "outside", "name": "구역 밖"}


# 매 프레임 YOLO를 돌리지 않고, 일정 간격으로 찍은 사진만 처리 (물건은 거의 움직이지 않으므로 충분)
SNAPSHOT_SEC = 2.0      # 자동 사진 처리 간격(초). 질문/명령이 오면 즉시 한 장 더 처리
POLL_SEC = 0.2          # control.json 확인 간격(초)


MATCH_IOU = 0.3         # 이전 사진 물체와 같은 물체로 볼 최소 IoU
STABLE_WINDOW = 4       # 최근 N장 중
STABLE_HITS = 3         # M장 이상 보이면 "있음" 확정 (손에 잠깐 가려져도 유지)
MISSING_SEC = 4.5       # 확정된 물건이 이 시간(사진 2장 이상) 넘게 안 보이면 "챙김"
ALERT_SEC = 6           # 남은 물건 경고 화면 표시 시간


BLUR_K, MORPH_K = 7, 5
OPEN_ITER, CLOSE_ITER = 2, 3


CONTEXT_WINDOW = 1024   # Nano: KV 캐시 메모리 절약 (의도 분석 프롬프트 + 이미지 토큰이 들어가는 크기)
MAX_TOKENS_INTENT = 80
MAX_TOKENS_NAME = 20
SNAPSHOT_WAIT_SEC = 4.0  # 질문 후 vision이 새 사진을 처리할 때까지 기다리는 최대 시간


GREEN, YELLOW, RED, WHITE, BLACK = (0, 200, 0), (0, 200, 255), (0, 0, 255), (255, 255, 255), (0, 0, 0)
GRAY, BLUE, ORANGE = (150, 150, 150), (255, 140, 0), (0, 140, 255)
LINE, TEXT, TEXT_L, PAD = 2, 0.5, 0.6, 12
WIN = "Checkout Checker"
