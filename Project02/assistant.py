import re
import sys
import json
import time
import base64
import select
import subprocess

from config import *
from common import write_json, read_json, name_ko, josa, ago_ko, is_exception, is_my_item, is_left
from away import away_report


# ======================== 초기화 ========================
llm = None
if USE_LLM:
    from llama_cpp import Llama
    chat_handler = None
    if USE_VISION_LLM:                       # 이미지 입력용 mmproj (06-B)
        from llama_cpp.llama_chat_format import Gemma4ChatHandler
        chat_handler = Gemma4ChatHandler(clip_model_path=MMPROJ_PATH)
    llm = Llama(
        model_path=GEMMA_MODEL_PATH,
        chat_handler=chat_handler,
        n_gpu_layers=-1,
        n_ctx=CONTEXT_WINDOW,
        n_batch=32,
        n_ubatch=32,
        verbose=False,
    )

ALL_ITEMS = YOLO_ITEMS + list(MARKERS.keys())
GENERIC_WORDS = {"물건", "소지품", "케이스"}     # "내 물건 뭐 있어?"의 '물건'이 이름 붙인 물건과 매칭되지 않도록
INTENTS = ["check", "locate", "list", "exception_add", "exception_remove", "checkin", "depart",
           "away_on", "away_off", "away_report", "unknown"]


# ======================== 1. STT / TTS (06-C) ========================
def speech_to_text():
    subprocess.run(
        ["pasuspender", "--", "arecord", "-D", MIC_DEVICE, "-f", "S16_LE", "-r", "16000",
         "-c", "1", "-d", str(RECORD_SECONDS), AUDIO_FILE],
        check=True,
    )
    result = subprocess.run(
        [WHISPER_PATH, "-m", WHISPER_MODEL, "-f", AUDIO_FILE, "-l", "ko", "--no-timestamps"],
        text=True, capture_output=True, check=True,
    )
    return re.sub(r"\[.*?\]", "", result.stdout).strip()     # [BLANK_AUDIO] 같은 태그 제거


def text_to_speech(text):
    print(f"\n[응답] {text}\n")
    if not USE_TTS:
        return
    subprocess.run([PIPER_PYTHON, "-m", "piper", "-m", PIPER_MODEL, "-f", OUTPUT_FILE, "--", text], check=True)
    subprocess.run(["aplay", OUTPUT_FILE], check=True)


# ======================== 2. LLM ========================
def generate(messages, max_tokens):
    response = llm.create_chat_completion(messages=messages, max_tokens=max_tokens, temperature=0.1)
    return response["choices"][0]["message"]["content"].strip()


def parse_intent_llm(text):
    """사용자 발화 → {"intent": ..., "items": [...]} (Structured Output)"""
    item_list = ", ".join(f"{key}({name_ko(key)})" for key in ALL_ITEMS)
    system_prompt = f"""
Instruction:
숙소 체크아웃 도우미에게 한 사용자 발화를 분석하여 의도와 물건을 JSON으로 출력하시오.

Context:
의도 목록:
- check: 나가도 되는지, 빠뜨린 물건이 없는지 확인 ("나 이제 나가도 돼?", "빠진 거 없어?")
- locate: 특정 물건의 위치를 물음 ("내 폰 어디 있어?")
- list: 내 물건 목록을 물음 ("내 물건 뭐 있어?")
- exception_add: 두고 가도 되는 물건을 알려줌 ("우산은 두고 가도 돼", "컵은 숙소 거야")
- exception_remove: 예외를 취소함 ("우산도 챙길래")
- checkin: 체크인, 지금 있는 물건은 숙소 비품이라고 알려줌 ("체크인", "지금 있는 건 숙소 거야")
- depart: 체크아웃하고 숙소를 떠난다고 알림 ("나 이제 나갈게", "출발할게")
- away_on: 짐을 두고 잠깐 외출함 ("외출할게", "잠깐 나갔다 올게")
- away_off: 외출에서 돌아옴 ("다녀왔어", "나 왔어")
- away_report: 외출한 동안 생긴 일을 물음 ("나 없는 동안 무슨 일 있었어?")
- unknown: 위에 해당하지 않음
물건 키 목록: {item_list}

Constraint:
items에는 물건 키 목록에 있는 키만 넣으시오. 해당 물건이 없으면 빈 리스트로 두시오.
JSON 외의 다른 문장은 출력하지 마시오.

Output Format:
{{"intent": "의도", "items": ["물건 키"]}}
"""
    content = generate(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": text}],
        MAX_TOKENS_INTENT,
    )
    match = re.search(r"\{.*\}", content, re.S)
    if not match:
        return None
    try:
        result = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    if result.get("intent") not in INTENTS:
        return None
    result["items"] = [i for i in result.get("items", []) if i in ALL_ITEMS]
    return result


def parse_intent_rule(text):
    """Gemma를 쓰지 않거나 JSON 파싱에 실패했을 때 쓰는 키워드 규칙"""
    items = [key for key in ALL_ITEMS if name_ko(key) in text or key in text]
    if "폰" in text and "cell phone" not in items:
        items.append("cell phone")

    if "체크인" in text or "숙소 거" in text:
        intent = "checkin" if not items else "exception_add"
    elif any(k in text for k in ["없는 동안", "무슨 일"]):
        intent = "away_report"
    elif any(k in text for k in ["다녀왔", "나 왔어", "돌아왔"]):
        intent = "away_off"
    elif any(k in text for k in ["외출", "나갔다 올", "다녀올게"]):
        intent = "away_on"
    elif any(k in text for k in ["두고 가도", "놓고 가도", "안 챙겨도", "필요 없"]):
        intent = "exception_add"
    elif any(k in text for k in ["챙길", "가져갈"]) and items:
        intent = "exception_remove"
    elif any(k in text for k in ["어디", "위치"]):
        intent = "locate"
    elif any(k in text for k in ["나갈게", "출발할게", "간다"]):
        intent = "depart"
    elif any(k in text for k in ["나가도", "빠진", "빠뜨", "두고 가는", "체크아웃", "다 챙겼"]):
        intent = "check"
    elif any(k in text for k in ["목록", "뭐 있", "뭐뭐"]):
        intent = "list"
    else:
        intent = "unknown"
    return {"intent": intent, "items": items}


def find_named_items(text, state):
    """Gemma가 이름 붙인 모르는 물건 중 발화에 나온 것 → 'id:N' 키 ("충전기 어디 있어?" → 검은색 충전기 케이블)"""
    found = []
    for item in (state or {}).get("items", []):
        if item["class"] != UNKNOWN_CLASS or item["name"].startswith("물건 "):
            continue                                  # Gemma가 이름을 붙이지 않은 물건("물건 3번")은 제외
        words = [w for w in item["name"].split()
                 if len(w) >= 2 and not w.endswith("색") and w not in GENERIC_WORDS]
        if any(w in text for w in words):
            found.append(f"id:{item['id']}")
    return found


def parse_intent(text, state):
    result = parse_intent_llm(text) if USE_LLM else None
    result = result or parse_intent_rule(text)
    result["items"] += [k for k in find_named_items(text, state) if k not in result["items"]]
    return result


def name_item_image(image_path):
    """모르는 물건 ROI 이미지 → Gemma(mmproj)가 짧은 이름 생성 (06-B 이미지 + 텍스트 입력)"""
    with open(image_path, "rb") as file:
        image_base64 = base64.b64encode(file.read()).decode("utf-8")
    image_data = "data:image/jpeg;base64," + image_base64

    response = llm.create_chat_completion(
        messages=[
            {
                "role": "system",
                "content": """
Instruction:
이미지 속 물건 하나의 이름을 짧게 말하시오.

Constraint:
여행자의 소지품 이름으로 답하시오. (예: 검은색 충전기, 흰색 이어폰 케이스, 파란 지갑)
무엇인지 확실하지 않으면 "알 수 없음"이라고만 답하시오.
다른 설명이나 문장을 추가하지 마시오.

Output Format:
색 + 물건 이름, 15자 이내 명사.
""",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 물건은 무엇인가요?"},
                    {"type": "image_url", "image_url": {"url": image_data}},
                ],
            },
        ],
        max_tokens=MAX_TOKENS_NAME,
        temperature=0.0,
    )
    answer = response["choices"][0]["message"]["content"].strip()
    return clean_name(answer)


def clean_name(answer):
    """모델 답변에서 이름만 남김: 첫 줄, 따옴표·마침표·'입니다' 제거, 길이 제한"""
    name = answer.splitlines()[0] if answer else ""
    name = re.sub(r"[\"'`.*:]|입니다|이에요|예요", "", name).strip()
    if not name or "알 수 없" in name:
        return None
    return name[:15]


# ======================== 3. 규칙 판단 (state.json 기반) ========================
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
    """[휴대폰(침대 오른쪽), 충전기(협탁)] → '잠깐만요! 침대 오른쪽에 휴대폰, 협탁에 충전기가 아직 남아 있어요.'"""
    by_zone = {}
    for item in left:
        by_zone.setdefault(item["zone"], []).append(item["name"])
    parts = [f"{zone}에 {', '.join(names)}" for zone, names in by_zone.items()]
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
                answers.append(f"{josa(item['name'], ('은', '는'))} 지금 {item['zone']}에 있어요.")
            else:
                answers.append(f"{josa(item['name'], ('은', '는'))} {ago_ko(item['taken_at'], now)} "
                               f"{item['zone']}에서 챙기신 뒤로 안 보여요.")
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

    if name == "away_on":
        return "다녀오세요. 그동안 물건에 변화가 생기면 사진으로 남겨 둘게요."

    if name in ("away_off", "away_report"):
        return away_report(state, name == "away_off")

    return "잘 못 알아들었어요. '나 이제 나가도 돼?'나 '내 폰 어디 있어?'처럼 말씀해 주세요."





# ======================== 4. control.json (예외 / 명령 전달) ========================
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

    command = {"checkin": "checkin", "depart": "depart",
               "away_on": "away_on", "away_off": "away_off"}.get(name, "snapshot")
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


# ======================== 5. 처리 흐름 ========================
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

        except (subprocess.CalledProcessError, OSError) as error:
            # 마이크·스피커·whisper 실행 오류로 프로그램 전체가 멈추지 않도록
            print("[오류]", error)


if __name__ == "__main__":
    main()
