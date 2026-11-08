"""부재 모드 (방범)

외출 중 내 물건이 사라지거나 새 물체가 생긴 순간만 사진으로 저장한다. (영상 녹화·녹음 없음)
- vision.py : new_away_state(), set_away(), record_changes(), draw_status()
- assistant.py : away_report()
이 파일을 쓰지 않으려면 vision.py에서 away 호출 부분만 빼면 된다.
"""
import os
import time
from collections import deque

import cv2

from config import AWAY_DIR, AWAY_BUFFER, CARRIERS, RED
from common import josa, clock_ko, track_name


def new_away_state():
    return {"away": False, "away_log": [], "away_buffer": deque(maxlen=AWAY_BUFFER), "away_saved": set()}


def set_away(S, on):
    S["away"] = on
    if on:
        S["away_log"] = []
        S["away_buffer"].clear()
        print("[외출 모드 시작]")
    else:
        print(f"[외출 모드 종료] 기록 {len(S['away_log'])}건")


def save_frames(S):
    """메모리에 보관 중인 최근 사진(변화 직전 + 직후) 중 아직 저장하지 않은 것을 파일로 저장"""
    os.makedirs(AWAY_DIR, exist_ok=True)
    paths = []
    for snap_time, img in S["away_buffer"]:
        path = os.path.join(AWAY_DIR, time.strftime("away_%Y%m%d_%H%M%S", time.localtime(snap_time)) + ".jpg")
        if path not in S["away_saved"]:
            cv2.imwrite(path, img)
            S["away_saved"].add(path)
        paths.append(path)
    return paths


def record_changes(S, frame, now):
    """사진 처리마다 호출. 외출 중 내 물건(가방 포함)이 사라지거나 새 물체가 생기면 사진 저장 + 기록
    S["changes"]: vision.update_tracks가 쌓아 둔 (종류, track) 목록"""
    changes, S["changes"] = S["changes"], []
    if not S["away"]:
        return

    S["away_buffer"].append((now, frame.copy()))
    for kind, t in changes:
        if t["owner"] != "mine" and t["class"] not in CARRIERS:
            continue                                  # 숙소 비품·무시한 영역은 기록하지 않음
        if kind == "appeared" and t["status"] != "present":
            continue
        images = save_frames(S)
        when = t["taken_at"] if kind == "taken" else t["first_seen"]
        S["away_log"].append({"time": when, "type": kind, "name": track_name(t),
                              "zone": t["zone"]["name"], "images": images})
        print(f"[부재 기록] {kind}: {track_name(t)} ({t['zone']['name']}) 사진 {len(images)}장")


def draw_status(S, frame):
    if not S["away"]:
        return
    w = frame.shape[1]
    cv2.circle(frame, (w - 20, 18), 7, RED, -1)
    cv2.putText(frame, f"AWAY  changes: {len(S['away_log'])}", (w - 190, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, RED, 1, cv2.LINE_AA)


def away_report(state, returned):
    """외출 기록 → '다녀오셨어요. 7시 13분에 침대 왼쪽 배낭이 사라졌어요. 사진 4장을 저장해 뒀어요.'"""
    log = state.get("away_log", [])
    greeting = "다녀오셨어요. " if returned else ""
    if not log:
        return greeting + "외출하신 동안 물건에 변화는 없었어요."

    by_time, images = {}, set()
    for entry in log:
        action = "사라졌어요" if entry["type"] == "taken" else "새로 생겼어요"
        by_time.setdefault(clock_ko(entry["time"]), []).append(
            f"{entry['zone']} {josa(entry['name'], ('이', '가'))} {action}")
        images.update(entry["images"])

    # 같은 시각의 변화는 한 문장으로 묶음
    sentences = [f"{when}에 " + ", ".join(parts) + "." for when, parts in by_time.items()]
    return greeting + " ".join(sentences) + f" 사진 {len(images)}장을 저장해 뒀고, 24시간 뒤 자동으로 지워져요."
