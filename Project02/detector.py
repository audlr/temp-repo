"""탐지: YOLO(COCO), 초록 색 마커"""
import cv2
import numpy as np

from config import *


kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))


def load_yolo():
    """YOLO 엔진 로드 + 워밍업 + 사용할 클래스 이름을 COCO 클래스 ID로 변환 (04-D-6 클래스 지정 탐지)
    yolo.names를 바로 읽으면 엔진이 한 번 더 로드되므로(GPU 메모리 2배), 워밍업 결과에서 클래스 이름을 얻음"""
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL_PATH, task="detect")
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    result = yolo.predict(source=dummy, imgsz=YOLO_IMGSZ, verbose=False)[0]   # 첫 추론이 느린 것을 미리 처리
    name_to_id = {name: class_id for class_id, name in result.names.items()}
    class_ids = [name_to_id[name] for name in YOLO_ITEMS]
    return yolo, class_ids


def detect_yolo(yolo, class_ids, frame):
    """YOLO 탐지 결과를 dict 리스트로 변환 (06-A-2 JSON 구조와 같은 필드)"""
    results = yolo.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, classes=class_ids,
                           imgsz=YOLO_IMGSZ, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_name = result.names[int(box.cls[0].item())]
        conf = float(box.conf[0].item())
        if conf < CLASS_CONF.get(class_name, YOLO_CONF):     # 오탐이 잦은 클래스는 기준을 높게
            continue
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        detections.append({
            "class": class_name,
            "conf": round(conf, 3),
            "bbox": (int(x1), int(y1), int(x2), int(y2)),
        })
    return detections


def detect_markers(frame):
    """COCO에 없는 물건: 색 스티커를 LAB 변환 → 블러 → 색 분할 → 모폴로지 → 가장 큰 윤곽선의 BBox (Project01)
    LAB의 L(밝기) 범위를 넓게 잡아 두어 밝기 보정(CLAHE) 없이도 조명 변화에 강함"""
    blur = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2LAB), (BLUR_K, BLUR_K), 0)
    detections = []

    for marker_class, marker in MARKERS.items():
        mask = cv2.inRange(blur, marker["low"], marker["high"])
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contours = [c for c in contours if cv2.contourArea(c) > MARKER_MIN_AREA]
        if len(contours) == 0:
            continue

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)        # (x, y, w, h) → (x1, y1, x2, y2) 변환 (04-A)
        detections.append({"class": marker_class, "conf": 1.0, "bbox": (x, y, x + w, y + h)})

    return detections
