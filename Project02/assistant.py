"""프로세스 2: 음성(또는 키보드) 질문 → 의도 분석 → 규칙 판단 → 음성 답변, vision 이벤트 처리"""
import sys
import time
import select
import subprocess

from config import *
from common import write_json, read_json, name_ko, josa, ago_ko, is_exception, is_my_item, is_left
from intent import parse_intent, name_item_image
from speech import speech_to_text, text_to_speech


def my_items(state):
    return [i for i in state["items"] if is_my_item(i)]


def left_items(state, exceptions):
    """남은 물건 (판단 기준은 common.is_left, vision과 동일)"""
    return [i for i in state["items"] if is_left(i, exceptions)]


def key_name(key, state):
    """물건 키 → 한국어 이름. 'id:N'은 Gemma가 붙인 이름"""
    if key.startswith("id:"):
        for item in (state or {}).get("items", []):
            if f"id:{item['id']}" == key:
                return item["name"]
        return "그 물건"
    return name_ko(key)


def items_of(key, state):
    if key.startswith("id:"):
        return [i for i in state["items"] if f"id:{i['id']}" == key]
    return [i for i in state["items"] if i["class"] == key]


def left_sentence(left):
    """[휴대폰(노트북 왼쪽), 여권(컵 아래쪽)] → '잠깐만요! 노트북 왼쪽에 휴대폰, 컵 아래쪽에 여권이 아직 남아 있어요.'"""
    by_where = {}
    for item in left:
        by_where.setdefault(item["where"], []).append(item["name"])
    parts = [f"{where}에 {', '.join(names)}" for where, names in by_where.items()]
    return f"잠깐만요! {josa(', '.join(parts), ('이', '가'))} 아직 남아 있어요."


def decide(intent, state, control, now):
    """의도 + 현재 상태 → 답변 문장. 남은 물건 판단은 LLM이 아니라 코드가 함 (없는 물건을 지어내지 않도록)"""
    name = intent["intent"]
    items = intent["items"]

    if state is None:
        return "카메라 프로그램이 아직 실행되지 않았어요."

    if name in ("check", "depart"):
        left = left_items(state, control["exceptions"])
        return left_sentence(left) if left else "빠뜨린 물건이 없어요. 조심히 다녀오세요!"

    if name == "locate":
        if not items:
            return "어떤 물건을 찾으시는지 다시 말씀해 주세요."
        answers, missing = [], []
        for key in items:
            found = items_of(key, state)
            if not found:
                missing.append(key)
                continue
            item = max(found, key=lambda i: i["last_seen"])
            if item["status"] == "present":
                answers.append(f"{josa(item['name'], ('은', '는'))} 지금 {item['where']}에 있어요.")
            else:
                answers.append(f"{josa(item['name'], ('은', '는'))} {ago_ko(item['taken_at'], now)} "
                               f"{item['where']}에서 챙기신 뒤로 안 보여요.")
        if not answers:                                     # 찾은 물건이 하나도 없을 때만 '본 적 없음' 안내
            answers = [f"{josa(key_name(key, state), ('은', '는'))} 아직 본 적이 없어요." for key in missing]
        return " ".join(answers)

    if name == "list":
        mine = my_items(state)
        if not mine:
            return "아직 등록된 소지품이 없어요."
        present = ", ".join(i["name"] for i in mine if i["status"] == "present")
        taken = ", ".join(i["name"] for i in mine if i["status"] == "taken")
        if present and taken:
            answer = f"지금 꺼내 둔 물건은 {josa(present, ('이고', '고'))}, 챙긴 물건은 {josa(taken, ('이에요', '예요'))}."
        elif present:
            answer = f"지금 꺼내 둔 물건은 {josa(present, ('이에요', '예요'))}."
        else:
            answer = f"꺼내 둔 물건은 없고, 챙긴 물건은 {josa(taken, ('이에요', '예요'))}."
        return answer

    if name == "exception_add":
        if not items:
            return "어떤 물건을 두고 가실지 다시 말씀해 주세요."
        names = ", ".join(key_name(k, state) for k in items)
        return f"알겠어요. {josa(names, ('은', '는'))} 확인할 때 빼 둘게요."

    if name == "exception_remove":
        if not items:
            return "어떤 물건을 다시 챙기실지 말씀해 주세요."
        names = ", ".join(key_name(k, state) for k in items)
        return f"알겠어요. {names}도 다시 확인할게요."

    if name == "checkin":
        return "체크인했어요. 지금 보이는 물건은 숙소 비품으로 기억할게요."

    return "잘 못 알아들었어요. '나 이제 나가도 돼?'나 '내 폰 어디 있어?'처럼 말씀해 주세요."


def send_control(control, command=None):
    control["seq"] += 1
    control["command"] = command
    write_json(CONTROL_JSON, control)


def apply_intent(intent, control):
    """예외 목록·명령을 control.json에 반영. 명령이 없는 질문도 'snapshot'을 보내 새 사진을 찍게 함"""
    name = intent["intent"]
    if name == "exception_add":
        control["exceptions"] = sorted(set(control["exceptions"]) | set(intent["items"]))
    elif name == "exception_remove":
        control["exceptions"] = [c for c in control["exceptions"] if c not in intent["items"]]

    command = {"checkin": "checkin", "depart": "depart"}.get(name, "snapshot")
    send_control(control, command)
    return control["seq"]


def wait_for_snapshot(seq):
    """vision이 방금 보낸 요청(seq)까지 반영한 사진 결과를 저장할 때까지 대기"""
    deadline = time.time() + SNAPSHOT_WAIT_SEC
    state = read_json(STATE_JSON)
    while time.time() < deadline:
        if state and state.get("control_seq", 0) >= seq:
            return state
        time.sleep(0.1)
        state = read_json(STATE_JSON)
    print("[주의] 새 사진 결과를 받지 못해 마지막 기록으로 답합니다.")
    return state


def handle_question(question, control):
    print(f"[질문] {question}")
    state = read_json(STATE_JSON)
    sync_session(control, (state or {}).get("session"))
    intent = parse_intent(question, state)
    print(f"[의도] {intent}")

    seq = apply_intent(intent, control)
    state = wait_for_snapshot(seq)
    text_to_speech(decide(intent, state, control, time.time()))


def handle_unknown_item(event, control):
    """vision이 모르는 물건을 찾았을 때: ROI 이미지 → Gemma가 이름 생성 → control.json으로 vision에 전달"""
    name = None
    if USE_LLM and USE_VISION_LLM:
        try:
            start = time.perf_counter()
            name = name_item_image(event["image"])
            print(f"[Gemma 이미지 인식] {time.perf_counter() - start:.1f}s")   # Jetson 처리 시간 측정용
        except FileNotFoundError:
            print("[주의] ROI 이미지를 찾을 수 없습니다:", event["image"])
            return

        if name is None:
            # Gemma가 물건이 아니라고 판단 (그림자, 이불 주름 등) → 오탐지로 보고 무시하도록 vision에 전달
            control.setdefault("ignored", []).append(event["track_id"])
            send_control(control, "snapshot")
            print(f"[새 물건] id {event['track_id']} → 물건이 아님 (무시)")
            return

    name = name or f"물건 {event['track_id']}번"
    print(f"[새 물건] id {event['track_id']} → {name}")

    control.setdefault("labels", {})[str(event["track_id"])] = name
    send_control(control, "snapshot")


def handle_depart(event, control):
    """vision이 출발을 감지했을 때 자동 안내 (LLM 의도 분석 생략, 남은 물건은 event에 담긴 목록 사용)"""
    left = [i for i in event["items_left"] if not is_exception(i, control["exceptions"])]
    text_to_speech(left_sentence(left) if left else "빠뜨린 물건이 없어요. 조심히 다녀오세요!")


def read_line_nonblocking(timeout):
    """터미널 입력이 있으면 한 줄 반환, 없으면 None (Linux select)"""
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.readline() if ready else None


def sync_session(control, session):
    """vision을 다시 실행하면 물건 id가 1번부터 새로 매겨지므로, 이전 실행의 이름·무시·id 예외 목록을 비움"""
    if session is None or control.get("session") == session:
        return
    control["session"] = session
    control["labels"], control["ignored"] = {}, []
    control["exceptions"] = [e for e in control["exceptions"] if not e.startswith("id:")]
    print("[vision 재시작 감지] 이전 물건 이름 기록 초기화")


def new_events(last_seq):
    """event.json에서 아직 처리하지 않은 이벤트만 순서대로 반환"""
    events = (read_json(EVENT_JSON) or {}).get("events", [])
    return [e for e in events if e["seq"] > last_seq]


def main():
    control = read_json(CONTROL_JSON) or {"seq": 0, "command": None, "exceptions": [], "labels": {}, "ignored": []}
    old = new_events(0)
    last_event_seq = old[-1]["seq"] if old else 0          # 시작 전에 쌓인 이벤트는 무시

    sync_session(control, (read_json(STATE_JSON) or {}).get("session"))

    print("Checkout Assistant 시작")
    print("Enter: 음성 질문 녹음" if USE_STT else "질문을 입력하고 Enter")
    print("텍스트를 입력하고 Enter: 텍스트로 질문 / q: 종료\n")

    while True:
        try:
            # 1. vision 이벤트 확인 (새 물건 / 출발)
            for event in new_events(last_event_seq):
                last_event_seq = event["seq"]
                sync_session(control, event.get("session"))
                if event["type"] == "unknown_item":
                    handle_unknown_item(event, control)
                elif event["type"] == "depart" and event.get("reason") != "voice":
                    handle_depart(event, control)   # 음성으로 출발을 알린 경우는 handle_question에서 이미 답함

            # 2. 사용자 입력 확인 (0.1초 대기, CPU를 계속 쓰지 않도록)
            line = read_line_nonblocking(0.1)
            if line is None:
                continue
            line = line.strip()
            if line == "q":
                break
            if line == "" and USE_STT:
                print("말씀해 주세요...")
                line = speech_to_text()
                if not line:
                    print("[STT] 인식된 말이 없습니다. 다시 Enter를 눌러 주세요.")
            if line:
                handle_question(line, control)

        except (subprocess.CalledProcessError, OSError, RuntimeError) as error:
            # 마이크·스피커·whisper 실행 오류로 프로그램 전체가 멈추지 않도록
            print("[오류]", error)


if __name__ == "__main__":
    main()
