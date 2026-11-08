"""프로세스 1: 카메라 → 사진 처리 → state.json / event.json, control.json 명령 반영, 화면 표시"""
import os
import time
from collections import deque

import cv2
import numpy as np

from config import *
from common import (write_json, read_json, apply_nms, track_name,
                    is_exception, is_my_item, is_left, cleanup_old_files)
from detector import (load_yolo, detect_yolo, detect_markers, to_gray,
                      detect_unknown, update_background, save_roi)
from tracker import update_tracks, update_locations, summarize


def open_camera():
    """GStreamer(수업 방식)로 열고, 실패하면 V4L2로 직접 열기"""
    if USE_GSTREAMER:
        cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
        if cap.isOpened():
            return cap
        print("[카메라] GStreamer로 열지 못해 V4L2로 다시 시도합니다.")
    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # 오래된 프레임이 쌓이지 않도록
    return cap


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
                            "exceptions": S["exceptions"], "items": items})


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
                       "where": t["where"], "image": path})
        print(f"[새 물건] id {t['id']} ({t['where']}) → 이름 요청")
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
    elif command == "reset":
        reset(S)


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
    # 물건 (화면 글자는 영어: cv2.putText가 한글을 지원하지 않음)
    for t in S["tracks"]:
        if t["status"] == "taken" or t["owner"] == "ignored":
            continue
        if t["class"] == UNKNOWN_CLASS and t["status"] == "candidate":
            continue                                    # 확인 중인 모르는 물건은 화면에 그리지 않음 (깜빡임 방지)
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

    # 출발 경고
    if S["alert_until"] > now:
        cv2.addWeighted(np.full_like(frame, RED), 0.25, frame, 0.75, 0, frame)
        names = [screen_name(t) for t in S["tracks"] if t["id"] in S["alert_ids"]]
        put_text(frame, "WAIT! LEFT: " + ", ".join(names), (PAD, 70), WHITE, TEXT_L)

    h = frame.shape[0]
    put_text(frame, "i: check-in  d: depart check  c: reset  q: quit", (PAD, h - 10), WHITE)


def new_state():
    S = {"mode": "READY", "tracks": [], "next_id": 0, "exceptions": [],
         "session": int(time.time()), "control_seq": 0, "event_seq": 0, "alert_until": 0, "alert_ids": [],
         "pending": None, "last_snap": 0, "last_poll": 0, "last_cleanup": 0,
         "background": None, "last_gray": None, "events": deque(maxlen=20)}
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
    if key == ord("d"):
        S["pending"] = "depart_key"      # 새 사진을 찍은 뒤 출발 점검
    if key == ord("c"):
        reset(S)
    return True


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
    update_locations(S, frame.shape)

    objects = [d["bbox"] for d in detections] + [t["bbox"] for t in S["tracks"] if t["status"] != "taken"]
    update_background(S, gray, objects)
    if departed and S["mode"] == "WATCH":
        fire_depart(S, now, "carrier left")

    if now - S["last_cleanup"] >= CLEANUP_SEC:
        cleanup_old_files([ROI_DIR], KEEP_SEC, now)
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

    fails = 0
    while True:
        ret, frame = cap.read()
        now = time.time()
        if not ret:
            # USB 카메라가 잠깐 끊기거나 읽기에 실패해도 바로 종료하지 않고 다시 연결 시도
            fails += 1
            if fails == 1:
                print("[카메라] 프레임을 읽지 못했습니다. 다시 연결을 시도합니다...")
            if fails % CAMERA_RETRY == 0:
                cap.release()
                time.sleep(1.0)
                cap = open_camera()
            if fails >= CAMERA_RETRY * CAMERA_REOPEN_MAX:
                print("[카메라] 연결이 끊겼습니다. 케이블과 /dev/video 장치를 확인하세요.")
                break
            time.sleep(0.1)
            continue
        if fails:
            print("[카메라] 다시 연결됨")
            fails = 0
        cv2.imshow(WIN, process(S, yolo, class_ids, frame, now))
        if not handle_key(S, cv2.waitKey(1) & 0xFF, now):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
