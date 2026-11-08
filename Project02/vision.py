import os
import cv2
import time
import numpy as np
from collections import deque

import away
from config import *
from common import (write_json, read_json, apply_nms, calculate_iou, track_name,
                    is_exception, is_my_item, is_left, cleanup_old_files)


# ======================== 초기화 ========================
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_K, MORPH_K))


def load_yolo():
    """YOLO 엔진 로드 + 사용할 클래스 이름을 COCO 클래스 ID로 변환 (04-D-6 클래스 지정 탐지)"""
    from ultralytics import YOLO
    yolo = YOLO(YOLO_MODEL_PATH)
    name_to_id = {name: class_id for class_id, name in yolo.names.items()}
    class_ids = [name_to_id[name] for name in YOLO_ITEMS + CARRIERS]
    return yolo, class_ids


def open_camera():
    if USE_GSTREAMER:
        return cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)   # Jetson
    return cv2.VideoCapture(0)                                 # 노트북 웹캠


# ======================== 1. 탐지 ========================
def detect_yolo(yolo, class_ids, frame):
    """YOLO 탐지 결과를 dict 리스트로 변환 (06-A-2 JSON 구조와 같은 필드)"""
    results = yolo.predict(source=frame, conf=YOLO_CONF, iou=YOLO_IOU, classes=class_ids,
                           imgsz=YOLO_IMGSZ, verbose=False)
    result = results[0]

    detections = []
    for box in result.boxes:
        class_id = int(box.cls[0].item())
        x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
        detections.append({
            "class": result.names[class_id],
            "conf": round(float(box.conf[0].item()), 3),
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


def to_gray(frame):
    """배경 차분용: 흑백 + 큰 가우시안 블러 (작은 노이즈, 미세한 흔들림 무시)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (DIFF_BLUR, DIFF_BLUR), 0)


def detect_unknown(S, gray, known_bboxes):
    """체크인 배경과 비교해 생긴 변화 중, 이미 아는 물건이 아닌 영역 = 모르는 물건
    배경 차분 → 이진화 → 아는 물건 영역 제거 → 모폴로지 → Contour (02 섹션 기법)"""
    if S["background"] is None:
        return []

    diff = cv2.absdiff(gray, S["background"])
    _, mask = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)

    for x1, y1, x2, y2 in known_bboxes:
        cv2.rectangle(mask, (x1 - KNOWN_MARGIN, y1 - KNOWN_MARGIN), (x2 + KNOWN_MARGIN, y2 + KNOWN_MARGIN), 0, -1)

    if cv2.countNonZero(mask) > UNKNOWN_MAX_RATIO * mask.size:
        return []                                       # 조명이 바뀌었거나 사람이 화면을 가림

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=OPEN_ITER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=CLOSE_ITER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    for c in contours:
        if cv2.contourArea(c) < UNKNOWN_MIN_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        bbox = (x, y, x + w, y + h)
        if zone_of(bbox) is not OUTSIDE_ZONE:            # 구역 안에 놓인 것만
            detections.append({"class": UNKNOWN_CLASS, "conf": 0.5, "bbox": bbox})
    return detections


def update_background(S, gray, object_bboxes):
    """물체가 없는 영역만 배경을 천천히 갱신 (햇빛 변화 등으로 생기는 가짜 변화 방지)
    물체가 있는 영역은 갱신하지 않으므로, 새로 놓인 물건이 배경에 묻히지 않음"""
    if S["background"] is None:
        return
    mask = np.full(gray.shape, 255, dtype=np.uint8)
    for x1, y1, x2, y2 in object_bboxes:
        cv2.rectangle(mask, (x1 - KNOWN_MARGIN, y1 - KNOWN_MARGIN), (x2 + KNOWN_MARGIN, y2 + KNOWN_MARGIN), 0, -1)
    background = S["background"].astype(np.float32)
    cv2.accumulateWeighted(gray.astype(np.float32), background, BG_ALPHA, mask=mask)
    S["background"] = background.astype(np.uint8)


def save_roi(frame, bbox, path):
    """BBox 영역을 잘라 저장 (06-B ROI 저장 방식: 임시 파일 → os.replace)"""
    h, w = frame.shape[:2]
    x1, y1 = max(0, bbox[0] - ROI_PAD), max(0, bbox[1] - ROI_PAD)
    x2, y2 = min(w, bbox[2] + ROI_PAD), min(h, bbox[3] + ROI_PAD)
    roi = frame[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return False
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = path.replace(".jpg", "_tmp.jpg")
    if not cv2.imwrite(tmp_path, roi):
        return False
    os.replace(tmp_path, path)
    return True


# ======================== 2. 구역 판정 ========================
def zone_of(bbox):
    """BBox 중심점이 들어 있는 구역"""
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    for zone in ZONES:
        x1, y1, x2, y2 = zone["rect"]
        if x1 <= cx < x2 and y1 <= cy < y2:
            return zone
    return OUTSIDE_ZONE


# ======================== 3. 추적 ========================
# track 상태: candidate(확인 중) → present(있음) → taken(챙김) → (다시 보이면) present
# owner: unknown(체크인 전) / amenity(숙소 비품) / mine(내 소지품) / ignored(Gemma가 물건이 아니라고 판단)
def new_track(S, det, now):
    S["next_id"] += 1
    return {"id": S["next_id"], "class": det["class"], "bbox": det["bbox"],
            "hits": deque([1], maxlen=STABLE_WINDOW), "status": "candidate",
            "owner": "mine" if S["mode"] == "WATCH" else "unknown",
            "zone": zone_of(det["bbox"]), "first_seen": now, "last_seen": now,
            "taken_at": None, "origin": None, "label": None, "label_requested": False}


def match_detections(tracks, detections):
    """같은 클래스끼리 IoU가 가장 큰 쌍부터 매칭 (Greedy). (track, det) 쌍과 남은 탐지 반환"""
    pairs = []
    for ti, t in enumerate(tracks):
        for di, d in enumerate(detections):
            if t["class"] == d["class"]:
                iou = calculate_iou(t["bbox"], d["bbox"])
                if iou >= MATCH_IOU:
                    pairs.append((iou, ti, di))

    matched, used_t, used_d = [], set(), set()
    for iou, ti, di in sorted(pairs, reverse=True):
        if ti not in used_t and di not in used_d:
            matched.append((tracks[ti], detections[di]))
            used_t.add(ti)
            used_d.add(di)

    unmatched = [d for di, d in enumerate(detections) if di not in used_d]
    return matched, unmatched


def confirm(S, t):
    """candidate가 안정적으로 보이면 present로 확정. 같은 클래스의 '챙김' 기록이 있으면 그 기록을 되살림
    (물건을 옮겨 놓은 경우 → IoU 매칭이 끊겨도 같은 물건으로 취급)"""
    taken = [o for o in S["tracks"] if o["class"] == t["class"] and o["status"] == "taken" and o is not t]
    if taken and t["class"] != UNKNOWN_CLASS:          # 모르는 물건끼리는 같은 물건인지 알 수 없으므로 제외
        old = max(taken, key=lambda o: o["taken_at"])
        old.update(bbox=t["bbox"], hits=t["hits"], status="present", zone=t["zone"],
                   last_seen=t["last_seen"], taken_at=None)
        S["tracks"].remove(t)
        return old
    t["status"] = "present"
    return t


def update_tracks(S, detections, now):
    """탐지 결과로 track 갱신. 가방이 사라지면(출발) True 반환"""
    matched, unmatched = match_detections(S["tracks"], detections)
    seen = set()

    for t, d in matched:
        t.update(bbox=d["bbox"], zone=zone_of(d["bbox"]), last_seen=now)
        t["hits"].append(1)
        seen.add(t["id"])

    departed = False
    for t in list(S["tracks"]):
        if t["id"] not in seen:
            t["hits"].append(0)

        if t["status"] == "candidate":
            if sum(t["hits"]) >= STABLE_HITS:
                S["changes"].append(("appeared", confirm(S, t)))
            elif now - t["last_seen"] > MISSING_SEC:
                S["tracks"].remove(t)                           # 잠깐 보였던 오탐지 제거

        elif t["status"] == "present" and now - t["last_seen"] > MISSING_SEC:
            t["status"], t["taken_at"] = "taken", t["last_seen"]
            S["changes"].append(("taken", t))
            if t["class"] in CARRIERS:
                departed = True

    for d in unmatched:
        S["tracks"].append(new_track(S, d, now))

    return departed


# ======================== 4. 상태 / 이벤트 ========================
def summarize(t):
    return {"id": t["id"], "class": t["class"], "name": track_name(t),
            "zone": t["zone"]["name"], "status": t["status"], "owner": t["owner"],
            "first_seen": t["first_seen"], "last_seen": t["last_seen"], "taken_at": t["taken_at"]}


def screen_name(t):
    """화면 표시용 영어 이름 (cv2.putText는 한글을 그리지 못함)"""
    if t["class"] == UNKNOWN_CLASS:
        return f"item {t['id']}" + ("" if t["label"] else " ?")
    return t["class"]


def items_left(S):
    """출발 시 남아 있으면 안 되는 물건 (판단 기준은 common.is_left)"""
    return [t for t in S["tracks"] if is_left(t, S["exceptions"])]


def save_state(S, now):
    """사진 처리 후 저장. control_seq: 어느 요청까지 반영된 상태인지 (assistant가 최신 사진 결과를 기다릴 때 사용)"""
    items = [summarize(t) for t in S["tracks"] if t["status"] in ("present", "taken")]
    write_json(STATE_JSON, {"session": S["session"], "mode": S["mode"], "control_seq": S["control_seq"],
                            "exceptions": S["exceptions"], "items": items,
                            "away": S["away"], "away_log": S["away_log"]})


def push_event(S, event):
    """이벤트를 최근 20개까지 목록으로 저장 (출발 이벤트와 새 물건 이벤트가 연달아 와도 덮어쓰지 않음)"""
    S["event_seq"] += 1
    event["seq"], event["session"] = S["event_seq"], S["session"]
    S["events"].append(event)
    write_json(EVENT_JSON, {"events": list(S["events"])})


def fire_depart(S, now, reason):
    """출발 이벤트: 남은 물건 목록을 event.json으로 전달, 남은 물건이 있으면 화면 경고"""
    left = items_left(S)
    push_event(S, {"time": now, "type": "depart", "reason": reason,
                   "items_left": [summarize(t) for t in left]})
    S["alert_until"] = now + ALERT_SEC if left else 0
    S["alert_ids"] = [t["id"] for t in left]
    print(f"[출발 감지: {reason}] 남은 물건 {len(left)}개", [track_name(t) for t in left])


def request_label(S, t, frame, now):
    """새로 확정된 모르는 물건의 ROI를 저장하고, assistant에게 이름 붙이기 요청"""
    path = os.path.join(ROI_DIR, f"item_{t['id']}.jpg")
    if save_roi(frame, t["bbox"], path):
        push_event(S, {"time": now, "type": "unknown_item", "track_id": t["id"],
                       "zone": t["zone"]["name"], "image": path})
        print(f"[새 물건] id {t['id']} ({t['zone']['name']}) → 이름 요청")
    t["label_requested"] = True


def checkin(S):
    """체크인: 지금 보이는 물건은 숙소 비품으로 등록, 이후 새로 나타나는 물건만 내 소지품"""
    for t in S["tracks"]:
        if t["class"] not in CARRIERS:
            t["owner"] = "amenity"
            t["origin"] = t["bbox"]           # 비품을 옮기면 원래 자리가 배경과 달라 보이므로 기억해 두고 제외
    S["mode"] = "WATCH"
    S["background"] = S["last_gray"]          # 모르는 물건 탐지용 배경 (비품만 있는 상태)
    print(f"[체크인] 숙소 비품 {sum(t['owner'] == 'amenity' for t in S['tracks'])}개 등록")


def poll_control(S):
    """assistant가 보낸 예외 목록과 명령 확인 (seq가 바뀌었을 때만). 명령은 새 사진을 찍은 뒤 실행"""
    control = read_json(CONTROL_JSON)
    if not control or control.get("seq", 0) == S["control_seq"]:
        return
    S["control_seq"] = control["seq"]
    S["exceptions"] = control.get("exceptions", [])
    S["pending"] = control.get("command") or S["pending"] or "snapshot"

    labels = control.get("labels", {})                # assistant(Gemma)가 붙인 모르는 물건 이름
    ignored = control.get("ignored", [])              # Gemma가 물건이 아니라고 판단한 영역
    for t in S["tracks"]:
        if str(t["id"]) in labels:
            t["label"] = labels[str(t["id"])]
        if t["id"] in ignored:
            t["owner"] = "ignored"


def run_command(S, now):
    """사진 처리가 끝난 뒤 대기 중인 명령 실행 (최신 사진 기준으로 판단하기 위해)"""
    command, S["pending"] = S["pending"], None
    if command == "checkin":
        checkin(S)
    elif command == "depart":
        fire_depart(S, now, "voice")
    elif command == "depart_key":
        fire_depart(S, now, "key")
    elif command == "away_on":
        away.set_away(S, True)
    elif command == "away_off":
        away.set_away(S, False)
    elif command == "reset":
        reset(S)


# ======================== 5. 화면 표시 ========================
def put_text(img, text, org, color=WHITE, scale=TEXT):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def track_color(S, t):
    if t["class"] in CARRIERS:
        return YELLOW
    if t["status"] == "candidate":
        return GRAY
    if t["owner"] == "amenity":
        return GRAY
    if is_exception(t, S["exceptions"]):
        return BLUE
    if t["id"] in S["alert_ids"] and S["alert_until"] > time.time():
        return RED
    return GREEN


def draw(S, frame, now):
    # 구역
    for zone in ZONES:
        x1, y1, x2, y2 = zone["rect"]
        cv2.rectangle(frame, (x1, y1), (x2 - 1, y2 - 1), ORANGE, 1)
        put_text(frame, zone["key"], (x1 + 5, y2 - 8), ORANGE)

    # 물건 (화면 글자는 영어: cv2.putText가 한글을 지원하지 않음)
    for t in S["tracks"]:
        if t["status"] == "taken" or t["owner"] == "ignored":
            continue
        color = track_color(S, t)
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE)
        tag = {"amenity": " (room)", "unknown": ""}.get(t["owner"], "")
        put_text(frame, f"{screen_name(t)}{tag}", (x1, max(y1 - 6, 14)), color)

    # 상단 상태
    mine = [t for t in S["tracks"] if is_my_item(t)]
    present = sum(t["status"] == "present" for t in mine)
    taken = sum(t["status"] == "taken" for t in mine)
    if S["mode"] == "READY":
        put_text(frame, "READY - press i (or say check-in) to register room amenities", (PAD, 22), YELLOW, TEXT_L)
    else:
        put_text(frame, f"WATCH  mine: {present} on table / {taken} packed", (PAD, 22), GREEN, TEXT_L)
    if S["exceptions"]:
        put_text(frame, "leave OK: " + ", ".join(S["exceptions"]), (PAD, 44), BLUE)
    away.draw_status(S, frame)

    # 출발 경고
    if S["alert_until"] > now:
        cv2.addWeighted(np.full_like(frame, RED), 0.25, frame, 0.75, 0, frame)
        names = [screen_name(t) for t in S["tracks"] if t["id"] in S["alert_ids"]]
        put_text(frame, "WAIT! LEFT: " + ", ".join(names), (PAD, 70), WHITE, TEXT_L)

    h = frame.shape[0]
    put_text(frame, "i: check-in  a: away on/off  d: depart check  c: reset  q: quit", (PAD, h - 10), WHITE)


# ======================== 6. 상태 관리 ========================
def new_state():
    S = {"mode": "READY", "tracks": [], "next_id": 0, "exceptions": [],
         "session": int(time.time()), "control_seq": 0, "event_seq": 0, "alert_until": 0, "alert_ids": [],
         "pending": None, "last_snap": 0, "last_poll": 0, "last_cleanup": 0,
         "background": None, "last_gray": None, "events": deque(maxlen=20), "changes": []}
    S.update(away.new_away_state())
    return S


def reset(S):
    keep = {k: S[k] for k in ("session", "control_seq", "event_seq", "next_id", "events")}
    S.clear()
    S.update(new_state(), **keep)
    print("[초기화]")


def handle_key(S, key, now):
    """q 종료(False 반환), i 체크인, d 수동 출발 점검, c 초기화"""
    if key == ord("q"):
        return False
    if key == ord("i"):
        S["pending"] = "checkin"         # 새 사진을 찍은 뒤 체크인
    if key == ord("a"):
        S["pending"] = "away_off" if S["away"] else "away_on"
    if key == ord("d"):
        S["pending"] = "depart_key"      # 새 사진을 찍은 뒤 출발 점검
    if key == ord("c"):
        reset(S)
    return True


# ======================== 7. 사진 처리 ========================
def snapshot(S, yolo, class_ids, frame, now):
    """사진 한 장: 탐지(YOLO + 색 마커) → NMS → 모르는 물건 → 추적 → 출발 판정 → 명령 실행 → 저장"""
    gray = to_gray(frame)                     # 배경 차분용
    S["last_gray"] = gray

    # YOLO와 색 마커가 같은 물체를 동시에 잡을 수 있어 출처를 합친 뒤 NMS (YOLO 내부 NMS로는 불가)
    detections = detect_yolo(yolo, class_ids, frame) + detect_markers(frame)
    detections = apply_nms(detections, NMS_IOU)

    # 모르는 물건: 아는 물건(이번 탐지 + 있다고 기록된 물건 + 비품 원래 자리) 영역을 뺀 나머지 변화
    if USE_UNKNOWN and S["mode"] == "WATCH":
        known = [d["bbox"] for d in detections]
        known += [t["bbox"] for t in S["tracks"] if t["status"] == "present" and t["class"] != UNKNOWN_CLASS]
        known += [t["origin"] for t in S["tracks"] if t["origin"]]
        detections += detect_unknown(S, gray, known)

    departed = update_tracks(S, detections, now)

    objects = [d["bbox"] for d in detections] + [t["bbox"] for t in S["tracks"] if t["status"] != "taken"]
    update_background(S, gray, objects)
    if departed and S["mode"] == "WATCH" and not S["away"]:   # 외출 중 가방이 사라진 건 출발이 아니라 방범 기록
        fire_depart(S, now, "carrier left")

    away.record_changes(S, frame, now)
    if now - S["last_cleanup"] >= CLEANUP_SEC:
        cleanup_old_files([AWAY_DIR, ROI_DIR], KEEP_SEC, now)
        S["last_cleanup"] = now

    for t in S["tracks"]:
        if (t["class"] == UNKNOWN_CLASS and t["status"] == "present"
                and t["owner"] == "mine" and not t["label_requested"]):
            request_label(S, t, frame, now)

    if S["pending"]:
        run_command(S, now)

    save_state(S, now)
    S["last_snap"] = now


def process(S, yolo, class_ids, frame, now):
    """매 프레임은 화면 표시만 하고, SNAPSHOT_SEC마다 또는 요청이 있을 때만 사진 처리"""
    if now - S["last_poll"] >= POLL_SEC:
        poll_control(S)
        S["last_poll"] = now

    if S["pending"] or now - S["last_snap"] >= SNAPSHOT_SEC:
        snapshot(S, yolo, class_ids, frame, now)

    draw(S, frame, now)
    return frame


def main():
    yolo, class_ids = load_yolo()
    cap = open_camera()
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    S = new_state()
    control = read_json(CONTROL_JSON)
    if control:                                # 이전 실행 때 남은 명령은 실행하지 않음 (예외 목록만 이어받음)
        S["control_seq"] = control.get("seq", 0)
        S["exceptions"] = [e for e in control.get("exceptions", []) if not e.startswith("id:")]
    print("Checkout Checker 시작 - 숙소 비품만 있는 상태에서 i 키(또는 '체크인')")

    while True:
        ret, frame = cap.read()
        now = time.time()
        if not ret:
            break
        cv2.imshow(WIN, process(S, yolo, class_ids, frame, now))
        if not handle_key(S, cv2.waitKey(1) & 0xFF, now):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
