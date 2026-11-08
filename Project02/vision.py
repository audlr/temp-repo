"""프로세스 1: 카메라 → 2초마다 사진 처리 → state.json / event.json, control.json 명령 반영, 화면 표시

흐름: 체크인(i 또는 "체크인") 때 보이는 물건 = 숙소 비품
      → 2초마다 물건이 생겼는지, 옮겨졌는지, 사라졌는지 확인
      → 체크아웃(o 또는 "체크아웃") 때 체크인 상태와 비교해 남은 물건·없어진 비품 안내
"""
import time
from collections import deque

import cv2
import numpy as np

from config import *
from common import write_json, read_json, apply_nms, name_ko, is_left, is_missing_amenity
from detector import load_yolo, detect_yolo, detect_markers
from tracker import update_tracks, update_locations, summarize


def log_changes(S, changes, now):
    """이번 사진에서 생긴 변화를 터미널에 출력하고 최근 20개를 기록 (체크인 후에만)"""
    if S["mode"] != "WATCH":
        return
    words = {"appeared": "생김", "moved": "이동", "missing": "화면에서 사라짐",
             "returned": "다시 탐지됨", "taken": "사라짐(이전 버전)"}
    for kind, t in changes:
        owner = "비품" if t["owner"] == "amenity" else "내 물건"
        print(f"[변화] {t['name']}({owner}) {words[kind]} - {t['where']}")
        S["changes"].append({"time": now, "type": kind, "class": t["class"],
                             "name": t["name"], "owner": t["owner"], "where": t["where"]})


def save_state(S, now):
    """사진 처리 후 저장. control_seq: 어느 요청까지 반영된 상태인지 (assistant가 최신 사진 결과를 기다릴 때 사용)"""
    items = [summarize(t) for t in S["tracks"] if t["status"] in ("present", "missing", "taken")]
    write_json(STATE_JSON, {"mode": S["mode"], "time": now, "control_seq": S["control_seq"],
                            "exceptions": S["exceptions"], "items": items, "changes": list(S["changes"])})


def checkin(S):
    """체크인: 지금 보이는 물건은 숙소 비품으로 기록, 이후 새로 생기는 물건만 내 물건"""
    for t in S["tracks"]:
        t["owner"] = "amenity"
    S["mode"] = "WATCH"
    S["changes"].clear()
    names = [t["name"] for t in S["tracks"] if t["status"] == "present"]
    print(f"[체크인] 숙소 비품 {len(names)}개: {', '.join(names) or '없음'}")


def checkout(S, now, reason):
    """체크아웃: 체크인 상태와 비교 → 남은 내 물건, 사라진 숙소 비품을 event.json으로 전달 + 화면 경고"""
    if S["mode"] != "WATCH":
        print("[체크아웃] 체크인 전이라 비교할 기준이 없습니다. 먼저 i 키(또는 '체크인')를 눌러 주세요.")
        return
    left = [t for t in S["tracks"] if is_left(t, S["exceptions"])]
    missing = [t for t in S["tracks"] if is_missing_amenity(t)]
    uncertain = [t for t in S["tracks"] if t["status"] == "missing" and t["owner"] == "mine"]

    S["event_seq"] += 1
    write_json(EVENT_JSON, {"seq": S["event_seq"], "time": now, "type": "checkout", "reason": reason,
                            "items_left": [summarize(t) for t in left],
                            "missing_amenities": [summarize(t) for t in missing],
                            "items_uncertain": [summarize(t) for t in uncertain]})

    S["alert_until"] = now + ALERT_SEC if left else 0
    S["alert_ids"] = [t["id"] for t in left]
    print(f"[체크아웃] 남은 물건 {len(left)}개 {[t['name'] for t in left]}, "
          f"확인되지 않는 내 물건 {len(uncertain)}개 {[t['name'] for t in uncertain]}, "
          f"화면에서 사라진 비품 {len(missing)}개 {[t['name'] for t in missing]}")


def poll_control(S):
    """assistant가 보낸 예외 목록과 명령 확인 (seq가 바뀌었을 때만). 명령은 새 사진을 찍은 뒤 실행"""
    control = read_json(CONTROL_JSON)
    if not control or control.get("seq", 0) == S["control_seq"]:
        return
    S["control_seq"] = control["seq"]
    S["exceptions"] = control.get("exceptions", [])
    S["pending"] = control.get("command") or S["pending"] or "snapshot"


def run_command(S, now):
    """사진 처리가 끝난 뒤 대기 중인 명령 실행 (최신 사진 기준으로 판단하기 위해)"""
    command, S["pending"] = S["pending"], None
    if command == "checkin":
        checkin(S)
    elif command == "checkout":
        checkout(S, now, "voice")
    elif command == "checkout_key":
        checkout(S, now, "key")
    elif command == "reset":
        reset(S)


def put_text(img, text, org, color=WHITE, scale=TEXT):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, BLACK, 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def track_color(S, t):
    if t["status"] == "candidate" or t["owner"] == "amenity":
        return GRAY
    if t["class"] in S["exceptions"]:
        return BLUE
    if t["id"] in S["alert_ids"] and S["alert_until"] > time.time():
        return RED
    return GREEN


def draw(S, frame, now):
    # 물건 (화면 글자는 영어: cv2.putText가 한글을 지원하지 않음)
    for t in S["tracks"]:
        if t["status"] in ("taken", "missing"):
            continue
        color = track_color(S, t)
        x1, y1, x2, y2 = t["bbox"]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, LINE)
        tag = " (room)" if t["owner"] == "amenity" else ""
        num = f" {t['num']}" if t["name"] != name_ko(t["class"]) else ""   # 같은 종류가 여러 개면 번호 표시
        put_text(frame, f"{t['class']}{num}{tag}", (x1, max(y1 - 6, 14)), color)

    # 상단 상태
    mine = [t for t in S["tracks"] if t["owner"] == "mine"]
    present = sum(t["status"] == "present" for t in mine)
    taken = sum(t["status"] == "taken" for t in mine)
    uncertain = sum(t["status"] == "missing" for t in mine)
    if S["mode"] == "READY":
        put_text(frame, "READY - press i (or say check-in) to register room amenities", (PAD, 22), YELLOW, TEXT_L)
    else:
        put_text(frame, f"WATCH  mine: {present} out / {taken} packed / {uncertain} unconfirmed", (PAD, 22), GREEN, TEXT_L)
    if S["exceptions"]:
        put_text(frame, "leave OK: " + ", ".join(S["exceptions"]), (PAD, 44), BLUE)

    # 체크아웃 경고
    if S["alert_until"] > now:
        cv2.addWeighted(np.full_like(frame, RED), 0.25, frame, 0.75, 0, frame)
        names = [t["class"] for t in S["tracks"] if t["id"] in S["alert_ids"]]
        put_text(frame, "WAIT! LEFT: " + ", ".join(names), (PAD, 70), WHITE, TEXT_L)

    h = frame.shape[0]
    put_text(frame, "i: check-in  o: check-out  c: reset  q: quit", (PAD, h - 10), WHITE)


def new_state():
    return {"mode": "READY", "tracks": [], "next_id": 0, "exceptions": [],
            "control_seq": 0, "event_seq": 0, "alert_until": 0, "alert_ids": [], "class_nums": {},
            "pending": None, "last_snap": 0, "last_poll": 0, "changes": deque(maxlen=20)}


def reset(S):
    keep = {k: S[k] for k in ("control_seq", "event_seq", "next_id")}
    S.clear()
    S.update(new_state(), **keep)
    print("[초기화]")


def handle_key(S, key):
    """q 종료(False 반환), i 체크인, o 체크아웃, c 초기화"""
    if key == ord("q"):
        return False
    if key == ord("i"):
        S["pending"] = "checkin"          # 새 사진을 찍은 뒤 체크인
    if key == ord("o"):
        S["pending"] = "checkout_key"     # 새 사진을 찍은 뒤 체크아웃
    if key == ord("c"):
        reset(S)
    return True


def snapshot(S, yolo, class_ids, frame, now):
    """사진 한 장: 탐지(YOLO + 초록 마커) → NMS → 추적·변화 기록 → 위치 설명 → 명령 실행 → 저장"""
    # YOLO와 색 마커가 같은 물체를 동시에 잡을 수 있어 출처를 합친 뒤 NMS (YOLO 내부 NMS로는 불가)
    detections = detect_yolo(yolo, class_ids, frame) + detect_markers(frame)
    detections = apply_nms(detections, NMS_IOU)

    changes = update_tracks(S, detections, now)
    update_locations(S, frame.shape)
    log_changes(S, changes, now)

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
    cap = cv2.VideoCapture(PIPELINE, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return

    S = new_state()
    control = read_json(CONTROL_JSON)
    if control:                                # 이전 실행 때 남은 명령은 실행하지 않음 (예외 목록만 이어받음)
        S["control_seq"] = control.get("seq", 0)
        S["exceptions"] = control.get("exceptions", [])
    S["event_seq"] = (read_json(EVENT_JSON) or {}).get("seq", 0)   # 다시 실행해도 번호가 이어지도록 (assistant가 놓치지 않게)
    print("Checkout Checker 시작 - 숙소 비품만 있는 상태에서 i 키(또는 '체크인')")

    while True:
        ret, frame = cap.read()
        now = time.time()
        if not ret:
            break

        cv2.imshow(WIN, process(S, yolo, class_ids, frame, now))
        if not handle_key(S, cv2.waitKey(1) & 0xFF):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
