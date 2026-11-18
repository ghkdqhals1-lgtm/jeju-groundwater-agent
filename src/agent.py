"""제주 삼다수 취수원 지하수 감시정 Agent: Claude API(도구 호출)로 질문에 답한다.

- ask(question, mode, history=[(질문, 답변), ...])를 부르면 Claude가 필요한 도구를 골라 부르고, 도구 결과로만 답을 만든다.
  history는 최근 4번까지만 앞쪽 메시지로 붙인다(도구 결과는 붙이지 않는다).
- 키는 .env의 ANTHROPIC_API_KEY를 python-dotenv로 이 프로세스 안에서만 읽는다
  (os.environ에 넣지 않는다. 화면·로그·오류 메시지에 출력하지 않는다).
- 모델 이름은 .env의 ANTHROPIC_MODEL에서 읽고, 없으면 claude-sonnet-5-5를 쓴다.
- 도구 호출 루프에서는 Claude 응답 블록(생각 블록 포함)을 수정하지 않고 그대로 다음 요청에 되돌려 보낸다
  (공식 문서: thinking 블록은 도구 사용 중 그대로 돌려줘야 도구 호출의 추론이 이어진다).

실행:
  python src/agent.py --check   # 키·모델 연결 확인 (아주 작은 요청 1번)
  python src/agent.py --test    # 질문 10개 시험
  python src/agent.py "질문" [--mode 도민]
"""
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import dotenv_values

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                                              # 측정값 방향 (VALUE_IS_DEPTH)
import status_check as sc                                  # 최근 7일 변화 계산 (추세 불일치 표시용)
from status_check import get_status as status_tool, WELLS  # 상태 진단
import forecast as fc                                      # 2주 예측 (모델 파일 사용)
from rain_analysis import zone_rain                        # zone별 일평균 강수량

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
LOG_PATH = ROOT / "logs" / "agent_log.jsonl"
CHART_DIR = ROOT / "data" / "processed" / "charts"
HOURLY_PATH = ROOT / "data" / "processed" / "groundwater_hourly.csv"
EVENTS_PATH = ROOT / "data" / "processed" / "sensor_events.csv"

# ──────────────────────────────────────────────────────────────────────────────
# 가격 (claude-sonnet-5-5 기준, 100만 토큰당 달러). 공식 가격 페이지에서 확인 필요
#   https://platform.claude.com/docs/en/about-claude/pricing
PRICE_INPUT_PER_MTOK = 2.0
PRICE_OUTPUT_PER_MTOK = 10.0
DEFAULT_MODEL = "claude-sonnet-5-5"
MAX_TOOL_CALLS = 8          # 한 질문당 도구 호출 최대 횟수 (넘으면 지금까지 결과로 답하게 함)
MAX_API_STEPS = 12          # 한 질문당 API 요청 최대 횟수 (무한 반복 방지)
MAX_TOKENS = 16000          # 요청 하나의 최대 출력 토큰 (생각 + 답변)
EFFORT = "medium"           # 여러 단계 도구 호출에 맞춘 생각 깊이 (Sonnet 5.5 기본은 high)
# 안전 분류기가 요청을 거절하면 서버가 권장 모델로 다시 실행하게 한다 (Claude API 권장 기본값).
#   다른 모델이 답하면 가격이 다를 수 있어, 결과에 실제 응답 모델을 함께 남긴다.
FALLBACK_BETA = "server-side-fallback-2026-07-01"
DATA_START, DATA_END = "2022-01-01", "2023-06-30"
MAX_EVENTS = 20             # 센서 이상 이벤트 최대 반환 건수
SAMPLE_DAYS = 10            # 기간 요약에서 보여줄 일별 중앙값 개수
HISTORY_TURNS = 4           # 대화 이어가기: 앞쪽 메시지로 붙일 최근 (질문, 답변) 개수. 도구 결과는 붙이지 않는다

# 추세와 예측의 불일치 표시 (get_forecast, get_overview)
#   최근 7일 측정값 변화(m/일, 계단 변화를 뺀 하루 중앙값 기준 = get_status와 같은 값)와
#   14일 예측 측정값 변화(중앙값 시나리오)를 하루 평균으로 맞춰 비교한다.
#   (a) 부호가 반대: 둘 다 MISMATCH_SIGN_FLOOR 이상일 때만 본다.
#       0.005 m/일(7일이면 0.035 m)은 측정 단위 0.01 m보다 작은 수준이라, 이보다 작으면 부호가 반올림 잡음일 수 있다.
#   (b) 최근 변화는 큰데 예측은 거의 평평: '크다'는 그 감시정의 18개월 7일 변화 크기 분포의 상위 10%
#       (MISMATCH_BIG_PERCENTILE). 감시정마다 움직이는 폭이 달라(2수원은 하루 수 cm, 1수원 감시정3은 수십 cm)
#       하나의 고정값 대신 자기 분포를 쓴다. '거의 평평'은 예측 하루 변화가 최근 하루 변화의 1/4 미만(MISMATCH_FLAT_RATIO).
#   예측 숫자와 시험 MAE는 바꾸지 않고 표시만 붙인다. 이 기준들은 시험 결과를 보기 전에 정했다.
MISMATCH_BIG_PERCENTILE = 90
MISMATCH_FLAT_RATIO = 0.25
MISMATCH_SIGN_FLOOR = 0.005
MISMATCH_LABEL = "추세와 예측이 어긋남"
# ──────────────────────────────────────────────────────────────────────────────


# ═══ 키·클라이언트 ═══════════════════════════════════════════════════════════════
def read_env():
    """.env를 이 프로세스 안에서만 읽어 (키, 모델 이름)을 돌려준다. os.environ에는 넣지 않는다."""
    env = dotenv_values(ENV_PATH)
    return (env.get("ANTHROPIC_API_KEY") or "").strip(), (env.get("ANTHROPIC_MODEL") or "").strip() or DEFAULT_MODEL


API_KEY, MODEL = read_env()


def mask(text):
    """문자열에 키가 들어 있으면 가린다 (오류 메시지·로그 보호)."""
    text = str(text)
    return text.replace(API_KEY, "***KEY***") if API_KEY else text


def make_client():
    """Claude API 클라이언트를 만든다. SDK 자동 재시도는 끄고(재시도는 call_api에서 한 번만) 키는 직접 넘긴다."""
    return anthropic.Anthropic(api_key=API_KEY, max_retries=0, timeout=120.0)


def error_message(e):
    """API 예외를 한국어 안내 문구로 바꾼다 (키는 가림)."""
    if isinstance(e, anthropic.AuthenticationError):
        return "인증 실패: .env의 ANTHROPIC_API_KEY가 맞는지 확인하세요."
    if isinstance(e, anthropic.PermissionDeniedError):
        return "권한 없음: 이 키로는 해당 모델이나 기능을 쓸 수 없습니다."
    if isinstance(e, anthropic.RateLimitError):
        return "한도 초과: 요청이 너무 많습니다. 잠시 뒤 다시 시도하세요."
    if isinstance(e, (anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return "네트워크 오류: Claude API에 연결하지 못했습니다. 인터넷 연결을 확인하세요."
    if isinstance(e, anthropic.APIStatusError):
        body = mask(getattr(e, "message", "")).lower()
        if e.status_code == 402 or getattr(e, "type", "") == "billing_error" or "credit balance" in body:
            return "잔액 부족: API 크레딧이 부족합니다. Claude Console의 결제(Billing) 화면을 확인하세요."
        if e.status_code == 404:
            return f"모델을 찾을 수 없음: .env의 ANTHROPIC_MODEL({MODEL})을 확인하세요."
        if e.status_code >= 500:
            return f"Claude API 서버 오류(상태 코드 {e.status_code}): 잠시 뒤 다시 시도하세요."
        return f"API 요청 오류(상태 코드 {e.status_code}): {mask(getattr(e, 'message', ''))[:200]}"
    return f"알 수 없는 오류: {type(e).__name__}"


def is_retryable(e):
    """한 번 더 시도해 볼 만한 오류(한도 초과, 네트워크, 서버 오류)인지 본다."""
    if isinstance(e, (anthropic.RateLimitError, anthropic.APITimeoutError, anthropic.APIConnectionError)):
        return True
    return isinstance(e, anthropic.APIStatusError) and e.status_code >= 500


def call_api(client, **params):
    """Claude API를 부른다. 재시도할 만한 오류면 3초 쉬고 한 번만 다시 시도하고, 그래도 실패하면 예외를 그대로 낸다."""
    try:
        return client.beta.messages.create(**params)
    except Exception as e:  # noqa: BLE001 - 아래에서 종류를 나눠 처리
        if not is_retryable(e):
            raise
        time.sleep(3)
        return client.beta.messages.create(**params)


# ═══ 입력 검사 ═══════════════════════════════════════════════════════════════════
class ToolInputError(Exception):
    """도구 입력이 잘못됐을 때 쓰는 예외 (모델에게 오류 문구로 돌려준다)."""


def check_well(well, allow_none=False):
    """감시정 이름이 6개 중 하나인지 확인한다."""
    if well is None and allow_none:
        return None
    if well not in WELLS:
        raise ToolInputError(f"감시정 이름이 잘못됐습니다: {well}. 가능한 값: {', '.join(WELLS)}")
    return well


def check_date(text, name, allow_none=False):
    """YYYY-MM-DD 날짜가 데이터 범위(2022-01-01 ~ 2023-06-30) 안인지 확인하고 Timestamp로 돌려준다."""
    if text is None and allow_none:
        return None
    if not isinstance(text, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        raise ToolInputError(f"{name}은(는) YYYY-MM-DD 형식이어야 합니다: {text}")
    try:
        t = pd.Timestamp(text)
    except ValueError:
        raise ToolInputError(f"{name} 날짜가 올바르지 않습니다: {text}")
    if not (pd.Timestamp(DATA_START) <= t <= pd.Timestamp(DATA_END)):
        raise ToolInputError(f"{name} {text}은(는) 데이터 범위 밖입니다. 데이터는 {DATA_START}부터 {DATA_END}까지입니다.")
    return t


def check_period(start, end):
    """기간 시작·끝을 확인한다 (둘 다 범위 안, 시작 ≤ 끝)."""
    s, e = check_date(start, "시작일"), check_date(end, "종료일")
    if s > e:
        raise ToolInputError(f"시작일({start})이 종료일({end})보다 늦습니다.")
    return s, e


# ═══ 도구 구현 ═══════════════════════════════════════════════════════════════════
_cache = {}


def hourly_data():
    """시간별 수위·수온 표를 한 번만 읽는다."""
    if "hourly" not in _cache:
        _cache["hourly"] = pd.read_csv(HOURLY_PATH, index_col="datetime", parse_dates=True, encoding="utf-8-sig")
    return _cache["hourly"]


def events_data():
    """센서 이상 이벤트 표를 한 번만 읽는다."""
    if "events" not in _cache:
        _cache["events"] = pd.read_csv(EVENTS_PATH, encoding="utf-8-sig", parse_dates=["시작 시각", "종료 시각"])
    return _cache["events"]


def tool_get_status(well, as_of):
    """상태 진단 도구: status_check.get_status를 부르고 결과를 그대로 돌려준다."""
    check_well(well)
    check_date(as_of, "기준일", allow_none=True)
    return status_tool(well, as_of)


def model_problem():
    """예측 모델 파일이 없거나 지금 입력 데이터와 맞지 않으면 안내 문구를, 바로 쓸 수 있으면 None을 돌려준다."""
    # 챗 도중 재학습(약 2분)을 막는다: 파일이 없거나 입력 데이터와 맞지 않으면 학습하지 않고 안내만 돌려준다
    if not fc.MODEL_PATH.exists():
        return "예측 모델 파일이 없습니다. 먼저 python src/forecast.py를 실행하세요."
    saved = json.loads(fc.MODEL_PATH.read_text(encoding="utf-8")).get("지문")
    if saved != fc.input_fingerprint():
        return "예측 모델 파일이 지금 입력 데이터와 맞지 않습니다. 먼저 python src/forecast.py를 실행하세요."
    return None


def big_change_threshold(well):
    """감시정 자신의 18개월 7일 측정값 변화(m/일) 크기 분포에서 상위 10% 경계값을 구한다 (한 번만 계산)."""
    key = ("big_change", well)
    if key not in _cache:
        _, _, corrected, _ = sc.load_data()
        change7 = sc.daily_change(corrected[well], 7).abs().dropna()
        _cache[key] = float(change7.quantile(MISMATCH_BIG_PERCENTILE / 100))
    return _cache[key]


def recent_change7(well, day):
    """기준 날의 최근 7일 측정값 하루 평균 변화(m/일)를 get_status와 같은 방식으로 구한다 (계산 못 하면 None)."""
    _, _, corrected, _ = sc.load_data()
    v = sc.daily_change(corrected[well], 7).get(pd.Timestamp(day))
    return None if v is None or pd.isna(v) else float(v)


def _value_word(change):
    """측정값 변화 방향을 '측정값이 커지는(물이 내려가는)' 같은 말로 바꾼다 (물을 함께 밝힌다)."""
    water_up = config.water_sign() * change > 0
    return f"측정값이 {'커지는' if change > 0 else '작아지는'}(물이 {'차오르는' if water_up else '내려가는'})"


def trend_mismatch(well, day, change14):
    """최근 7일 변화와 14일 예측 변화(중앙값 시나리오)를 비교해 '추세와 예측이 어긋남' 표시와 한 줄 설명을 만든다."""
    recent = recent_change7(well, day)
    out = {"최근 7일 측정값 변화(m/일)": None if recent is None else round(recent, 3),
           "14일 예측 측정값 변화(m, 중앙값 시나리오)": round(change14, 2),
           "큰 변화 기준(이 감시정 7일 변화 상위 10%, m/일)": round(big_change_threshold(well), 3)}
    if recent is None:
        out["표시"] = "판단 불가 (최근 7일 변화를 계산할 수 없음)"
        return out
    pred = change14 / 14                                     # 예측도 하루 평균 변화로 맞춘다
    opposite = (abs(recent) >= MISMATCH_SIGN_FLOOR and abs(pred) >= MISMATCH_SIGN_FLOOR
                and (recent > 0) != (pred > 0))
    flat = abs(recent) >= big_change_threshold(well) and abs(pred) < MISMATCH_FLAT_RATIO * abs(recent)
    if not (opposite or flat):
        out["표시"] = "없음"
        return out
    head = f"최근 7일은 하루 {abs(recent):.3f} m씩 {_value_word(recent)} 추세인데"   # get_status와 같은 자릿수
    if opposite:
        body = (f" 14일 예측은 반대로 {_value_word(pred)} 쪽입니다(14일 동안 {abs(change14):.2f} m). "
                "이 모델은 최근 추세를 그대로 따르지 않아 이 구간의 예측은 참고하기 어렵습니다.")
    else:
        body = " 14일 예측은 거의 변하지 않습니다. 이 모델은 최근 추세를 약하게 반영해서 이 구간의 예측은 참고하기 어렵습니다."
    out["표시"] = MISMATCH_LABEL
    out["설명"] = head + body
    return out


def median_forecast14(well, as_of):
    """중앙값 시나리오 14일 예측을 불러 (기준일, 14일 측정값 변화, 14일 뒤 예측 항목)을 돌려준다. 오류면 dict를 돌려준다."""
    r = fc.forecast(well, as_of, 14, "중앙값")
    if "오류" in r:
        return r
    value0 = next(v for k, v in r.items() if k.startswith("기준 측정값"))
    last = r["일별 예측"][-1]
    pred = next(v for k, v in last.items() if k.startswith("예측"))["중앙값"]
    return r["기준일"], pred - value0, last


def tool_get_forecast(well, as_of, horizon, rain_scenario):
    """2주 예측 도구: 모델 파일이 있을 때만 forecast를 부르고, 1·3·7·14일 뒤만 추리고 추세 불일치 표시를 붙여 돌려준다."""
    check_well(well)
    check_date(as_of, "기준일", allow_none=True)
    horizon = 14 if horizon is None else horizon
    if not (isinstance(horizon, int) and 1 <= horizon <= 14):
        raise ToolInputError("horizon은 1~14 사이 정수여야 합니다.")
    if rain_scenario is not None and rain_scenario not in fc.SCENARIOS:
        raise ToolInputError(f"rain_scenario는 {', '.join(fc.SCENARIOS)} 중 하나이거나 null이어야 합니다.")
    problem = model_problem()
    if problem:
        return {"오류": problem}
    r = fc.forecast(well, as_of, horizon, rain_scenario)
    if "오류" in r:
        return r
    m = median_forecast14(well, as_of)       # 불일치는 요청한 일수·시나리오와 관계없이 14일·중앙값 기준으로 본다
    mismatch = m if isinstance(m, dict) else trend_mismatch(well, m[0], m[1])
    keep_h = [h for h in fc.REPORT_H if h <= horizon] or [horizon]
    days = []
    for h in keep_h:
        d = r["일별 예측"][h - 1]
        days.append({"며칠 뒤": h, **{k: v for k, v in d.items() if not k.startswith("실제")}})
    return {
        "감시정": r["감시정"], "기준일": r["기준일"],
        **{k: v for k, v in r.items() if k.startswith("기준 측정값")},
        "설명": r["설명"], "강수 영향 안내": r["강수 영향 안내"],
        "강수 시나리오(14일 누적 mm)": next(v for k, v in r.items() if k.startswith("강수 시나리오")),
        "예측(1·3·7·14일 뒤)": days,
        "시험 구간(2023-04~06) 오차 비교": r["시험 구간(2023-04~06) 오차 비교"],
        "추세와 예측 비교": mismatch,
        "안내": r["안내"],
    }


def tool_get_overview(as_of):
    """전체 요약 도구: 6개 감시정의 등급·백분위·7일 변화·최근 센서 이상·14일 예측 요약을 감시정당 짧게 돌려준다."""
    check_date(as_of, "기준일", allow_none=True)
    problem = model_problem()
    wells, when = {}, None
    for well in WELLS:
        st = status_tool(well, as_of)
        if "오류" in st:
            return st
        when = st["기준 시각"]
        events = st["최근 7일 센서 이상 이벤트"]          # "수위 추세 이탈 의심 (기간)" 형식
        item = {
            "등급": st["등급"],
            "물 높이 백분위(낮은 쪽부터, %)": st["물 높이 백분위(낮은 쪽부터, %)"],
            "최근 7일 측정값 변화(m/일)": st["최근 7일 측정값 변화(m/일)"],
            "최근 7일 센서 이상": {"건수": len(events), "유형": sorted({e.split(" (")[0] for e in events})},
        }
        if problem:
            item["14일 뒤 예측"] = "예측 모델 파일이 없습니다" if not fc.MODEL_PATH.exists() else problem
        else:
            m = median_forecast14(well, as_of)
            if isinstance(m, dict):
                item["14일 뒤 예측"] = m["오류"]
            else:
                day, change14, last = m
                mm = trend_mismatch(well, day, change14)
                item["14일 뒤 예측"] = {
                    "측정값 변화(m, 중앙값 시나리오)": round(change14, 2),
                    "물 방향": config.water_expect(change14),
                    "예상 범위의 시험 포함률(%)": last["예상 범위의 시험 포함률(%)"],
                    "신뢰도": last["신뢰도"],
                    "추세 불일치": mm["표시"] if mm["표시"] != MISMATCH_LABEL else f"{MISMATCH_LABEL}: {mm['설명']}",
                }
        wells[well] = item
    return {"기준 시각": when, "감시정별": wells,
            "안내": sc.GRADE_NOTE + " " + sc.direction_note() + " 자세한 근거는 get_status, get_forecast, get_sensor_events로 보세요."}


def tool_get_sensor_events(well, start, end, type):  # noqa: A002 - 도구 입력 이름과 맞춤
    """센서 이상 이벤트 조회 도구: 감시정·기간·유형으로 거른 이벤트를 최대 20건 돌려준다."""
    check_well(well, allow_none=True)
    s = check_date(start, "시작일", allow_none=True)
    e = check_date(end, "종료일", allow_none=True)
    ev = events_data()
    if well:
        ev = ev[ev["감시정"] == well]
    if s is not None:
        ev = ev[ev["종료 시각"] >= s]
    if e is not None:
        ev = ev[ev["시작 시각"] <= e + pd.Timedelta(hours=23)]
    if type:
        ev = ev[ev["유형"] == type]
    rows = []
    for _, r in ev.head(MAX_EVENTS).iterrows():
        rows.append({"감시정": r["감시정"], "센서": r["센서"], "유형": r["유형"], "판정": r["판정"],
                     "시작": f"{r['시작 시각']:%Y-%m-%d %H:%M}", "종료": f"{r['종료 시각']:%Y-%m-%d %H:%M}",
                     "근거": r["근거"], "다른 감시정 동시 발생": r["다른 감시정 동시 발생"],
                     "직전 7일 강수(mm)": None if pd.isna(r["직전 7일 강수(mm, zone 평균)"]) else r["직전 7일 강수(mm, zone 평균)"],
                     "직전 7일 강수 상위(%)": None if pd.isna(r["직전 7일 강수 상위(%)"]) else r["직전 7일 강수 상위(%)"],
                     "강수 맥락": r["강수 맥락"]})
    return {"조건에 맞는 이벤트 수": len(ev), "보여준 건수": len(rows), "이벤트": rows,
            "안내": "원인은 단정하지 않고 '의심'으로 표시한 결과입니다. 전체 기간 47건 중 조건에 맞는 것만 보여 줍니다."}


def tool_get_values(well, start, end, kind):
    """측정값 기간 요약 도구: 수위 또는 수온의 시작·끝 값, 최소·최대, 평균, 일별 중앙값 몇 개를 돌려준다."""
    check_well(well)
    s, e = check_period(start, end)
    if kind not in ("수위", "수온"):
        raise ToolInputError("kind는 '수위' 또는 '수온'이어야 합니다.")
    col = ("level_" if kind == "수위" else "temp_") + well
    x = hourly_data()[col].loc[s:e + pd.Timedelta(hours=23)]
    daily = x.resample("D").median()
    step = max(1, len(daily) // SAMPLE_DAYS)
    sample = {f"{d:%Y-%m-%d}": round(float(v), 2) for d, v in daily.iloc[::step].items()}
    out = {
        "감시정": well, "항목": kind, "기간": f"{s:%Y-%m-%d} ~ {e:%Y-%m-%d}", "일수": len(daily),
        "단위": config.value_label() if kind == "수위" else "수온(°C)",
        "시작일 중앙값": round(float(daily.iloc[0]), 2), "종료일 중앙값": round(float(daily.iloc[-1]), 2),
        "최소(시간값)": round(float(x.min()), 2), "최소 시각": f"{x.idxmin():%Y-%m-%d %H:%M}",
        "최대(시간값)": round(float(x.max()), 2), "최대 시각": f"{x.idxmax():%Y-%m-%d %H:%M}",
        "평균": round(float(x.mean()), 2), "일별 중앙값 일부": sample,
    }
    if kind == "수위":
        out["기간 변화 해석"] = "기간 시작보다 끝에 " + config.water_compare(daily.iloc[-1] - daily.iloc[0])
        out["방향 가정"] = ("측정값은 물까지의 깊이로 가정(값이 작을수록 물이 높음). 명세로 확인하지 못함."
                          if config.VALUE_IS_DEPTH else "측정값은 해발 수위로 가정(값이 클수록 물이 높음).")
    return out


def tool_get_rainfall(start, end):
    """강수량 요약 도구: zone별(해안·중산간·산지) 일평균 강수량의 기간 합계와 가장 많이 온 날을 돌려준다."""
    s, e = check_period(start, end)
    r = zone_rain().loc[s:e, ["해안", "중산간", "산지"]]
    out = {"기간": f"{s:%Y-%m-%d} ~ {e:%Y-%m-%d}", "일수": len(r),
           "강수 자료가 빈 날": [f"{d:%Y-%m-%d}" for d in r.index[r.isna().all(axis=1)]], "zone별": {}}
    for z in r:
        col = r[z]
        out["zone별"][z] = {"합계(mm)": round(float(col.sum()), 1),
                           "가장 많이 온 날": None if col.isna().all() else f"{col.idxmax():%Y-%m-%d}",
                           "그날 강수(mm)": None if col.isna().all() else round(float(col.max()), 1)}
    out["안내"] = "zone별 값은 그 zone 관측소들의 일 강수량 평균입니다. 빈 날은 채우지 않고 합계에서 뺐습니다."
    return out


def tool_make_chart(well, start, end, kind):
    """차트 도구: 기간의 측정값 선 그래프를 PNG로 저장하고 경로와 한 줄 설명을 돌려준다 (센서 이상 이벤트 표시)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    check_well(well)
    s, e = check_period(start, end)
    if kind not in ("수위", "수온"):
        raise ToolInputError("kind는 '수위' 또는 '수온'이어야 합니다.")
    col = ("level_" if kind == "수위" else "temp_") + well
    x = hourly_data()[col].loc[s:e + pd.Timedelta(hours=23)]
    ev = events_data()
    ev = ev[(ev["감시정"] == well) & (ev["센서"] == kind) & (ev["종료 시각"] >= s) & (ev["시작 시각"] <= x.index.max())]

    plt.rcParams["font.family"] = "Malgun Gothic"     # 한글 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    fig, ax = plt.subplots(figsize=(11, 4))
    for _, r in ev.iterrows():   # 센서 이상 의심 구간: 주황 음영 (범주형 팔레트 2번)
        ax.axvspan(max(r["시작 시각"], x.index.min()), min(r["종료 시각"], x.index.max()) + pd.Timedelta(hours=1),
                   color="#eb6834", alpha=0.25, lw=0)
    ax.plot(x.index, x.values, color="#2a78d6", lw=1.2)
    if kind == "수위":
        config.orient_axis(ax)
    ylabel = config.axis_note() if kind == "수위" else "수온(°C)"
    ax.set_ylabel(ylabel, fontsize=9, color="#52514e")
    ax.set_title(f"{well} {kind} ({s:%Y-%m-%d} ~ {e:%Y-%m-%d})" + (f" · 센서 이상 의심 {len(ev)}건 음영" if len(ev) else ""),
                 loc="left", fontsize=11)
    ax.grid(axis="y", color="#e1e0d9", lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    CHART_DIR.mkdir(parents=True, exist_ok=True)
    path = CHART_DIR / f"{well}_{kind}_{s:%Y%m%d}_{e:%Y%m%d}.png"
    fig.savefig(path, dpi=120, facecolor="#fcfcfb")
    plt.close(fig)
    rel = path.relative_to(ROOT).as_posix()
    return {"경로": rel, "설명": f"{well} {kind} {s:%Y-%m-%d}~{e:%Y-%m-%d} 선 그래프, 센서 이상 의심 {len(ev)}건 표시"}


# ═══ 도구 정의 (입력 스키마) ═════════════════════════════════════════════════════
def _enum_or_null(values, description):
    """정해진 문자열 값 또는 null을 받는 스키마를 만든다(엄격 모드는 enum과 타입 배열을 함께 못 쓴다)."""
    return {"anyOf": [{"type": "string", "enum": list(values)}, {"type": "null"}], "description": description}


_WELL = {"type": "string", "enum": WELLS, "description": "감시정 이름 (6개 중 하나)"}
_WELL_OR_NULL = _enum_or_null(WELLS, "감시정 이름. 전체면 null")
_DATE = {"type": "string", "description": f"YYYY-MM-DD, {DATA_START} ~ {DATA_END} 안"}
_DATE_OR_NULL = {"type": ["string", "null"], "description": f"YYYY-MM-DD ({DATA_START} ~ {DATA_END}) 또는 null"}
_KIND = {"type": "string", "enum": ["수위", "수온"], "description": "측정 항목"}


def _tool(name, description, props):
    """엄격한 입력 스키마(strict, 모든 항목 필수, 추가 항목 금지)를 가진 도구 정의를 만든다."""
    return {"name": name, "description": description, "strict": True,
            "input_schema": {"type": "object", "properties": props, "required": list(props),
                             "additionalProperties": False}}


TOOLS = [
    _tool("get_overview", "6개 감시정 전체를 한 번에 짧게 요약한다: 감시정마다 등급, 물 높이 백분위, 최근 7일 측정값 변화, "
          "최근 7일 센서 이상(건수·유형), 14일 뒤 예측 요약(중앙값 시나리오 변화, 범위의 시험 포함률, 신뢰도, 추세 불일치 표시). "
          "여러 감시정을 묻는 질문(전체, 가장 위험한 곳, 이상한 곳 등)에 먼저 쓴다. as_of가 null이면 2023-06-30 기준.",
          {"as_of": _DATE_OR_NULL}),
    _tool("get_status", "감시정 하나의 지하수 상태 등급(정상/주의/경고)과 근거를 알려준다. 등급은 18개월 관측 안의 상대 위치다. "
          "as_of를 null로 주면 데이터 마지막 시점(2023-06-30)을 기준으로 한다.",
          {"well": _WELL, "as_of": _DATE_OR_NULL}),
    _tool("get_forecast", "감시정 하나의 앞으로 1~14일 측정값 예측(강수 시나리오별), 예상 범위, 신뢰도, 예상 범위의 시험 포함률, "
          "최근 추세와 예측의 불일치 표시를 알려준다.",
          {"well": _WELL, "as_of": _DATE_OR_NULL,
           "horizon": {"type": ["integer", "null"], "description": "예측 일수 1~14. null이면 14"},
           "rain_scenario": _enum_or_null(["무강수", "중앙값", "상위10%"], "강수 가정 하나만 볼 때. null이면 세 가지 모두")}),
    _tool("get_sensor_events", "센서 이상 의심 이벤트(순간 튐, 계단 변화, 값 고착, 추세 이탈, 변동성 변화)를 조건으로 찾는다. 최대 20건.",
          {"well": _WELL_OR_NULL, "start": _DATE_OR_NULL, "end": _DATE_OR_NULL,
           "type": _enum_or_null(["순간 튐", "계단 변화", "값 고착", "추세 이탈", "변동성 변화"], "이벤트 유형. 전체면 null")}),
    _tool("get_values", "감시정 하나의 수위 또는 수온을 기간으로 요약한다(시작·끝 값, 최소·최대, 평균, 일별 중앙값 일부).",
          {"well": _WELL, "start": _DATE, "end": _DATE, "kind": _KIND}),
    _tool("get_rainfall", "기간의 zone별(해안·중산간·산지) 일평균 강수량 합계와 가장 많이 온 날을 알려준다.",
          {"start": _DATE, "end": _DATE}),
    _tool("make_chart", "감시정 하나의 기간 측정값 선 그래프를 PNG로 저장하고 경로를 돌려준다. 센서 이상 의심 구간을 표시한다.",
          {"well": _WELL, "start": _DATE, "end": _DATE, "kind": _KIND}),
]
TOOL_FUNCS = {"get_overview": tool_get_overview, "get_status": tool_get_status, "get_forecast": tool_get_forecast,
              "get_sensor_events": tool_get_sensor_events, "get_values": tool_get_values,
              "get_rainfall": tool_get_rainfall, "make_chart": tool_make_chart}


def run_tool(name, args):
    """도구 하나를 실행해 (결과 JSON 문자열, 오류 여부)를 돌려준다. 잘못된 입력이나 예외는 오류 문구로 바꾼다."""
    func = TOOL_FUNCS.get(name)
    try:
        if func is None:
            raise ToolInputError(f"알 수 없는 도구입니다: {name}")
        result = func(**args)
        is_error = isinstance(result, dict) and "오류" in result
    except ToolInputError as e:
        result, is_error = {"오류": str(e)}, True
    except TypeError as e:
        result, is_error = {"오류": f"도구 입력 항목이 맞지 않습니다: {mask(e)}"}, True
    except Exception as e:  # noqa: BLE001 - 도구 오류로 Agent 전체가 죽지 않게 한다
        result, is_error = {"오류": f"도구 실행 중 오류: {type(e).__name__}: {mask(e)[:200]}"}, True
    return json.dumps(result, ensure_ascii=False, default=str), is_error


# ═══ 시스템 프롬프트 ═════════════════════════════════════════════════════════════
SYSTEM_PROMPT = f"""당신은 제주 삼다수 취수원 표선유역 지하수 감시정 데이터를 분석해 알려주는 도우미입니다.
주 사용자는 물 관리 담당자이고, 부 사용자는 지하수가 궁금한 도민입니다.

데이터
- 감시정 6개: {', '.join(WELLS)}
- 기간: {DATA_START} ~ {DATA_END} (시간별 측정값, zone별 일 강수량)
- '지금', '요즘', '오늘'은 데이터 마지막 시점인 {DATA_END}을 기준으로 답하고, 그 사실을 답변에 밝히세요. 이 기간 밖의 날짜는 조회할 수 없다고 알리세요.

답변 원칙
- 도구 결과에 있는 내용으로만 답하고, 숫자를 지어내지 마세요. 도구 결과의 숫자를 그대로 쓰세요.
- 도구로 알 수 없는 질문은 모른다고 하세요. 수질·오염·공식 제한 단계 판정·취수량처럼 이 데이터 밖의 질문은 정중히 범위 밖이라고 알리세요.
- 등급은 공식 기준이 아니라 18개월 관측 안의 상대적 상태입니다. 측정값은 물까지의 깊이로 '가정'한 것(명세로 확인하지 못함)입니다. 필요한 때 이 두 가지를 밝히세요.
- 예측을 전할 때는 도구 결과의 신뢰도 표시, 예상 범위와 그 범위의 시험 포함률, 앞으로 2주 강수의 효과가 거의 없다는 안내를 그대로 반영하세요. '참고 불가'나 '큰 변화 구간에서 오차가 큼' 표시가 있으면 반드시 전하세요.
- 센서 이상은 원인을 단정하지 말고 '의심'으로 말하고, 근거(변화 속도, 지속 시간, 다른 감시정 동시 발생, 직전 강수)를 함께 말하세요.
- 질문에 감시정이 없으면, 필요한 감시정을 여러 개 조회해 답하거나 어떤 감시정인지 되물어도 됩니다. 도구는 한 질문에 최대 {MAX_TOOL_CALLS}번까지 부를 수 있습니다.
- 여러 감시정을 묻는 질문(전체, 요즘 괜찮은지, 가장 위험한 곳, 이상한 곳 등)은 get_overview를 먼저 부르세요. 더 자세한 근거가 필요한 감시정만 다른 도구로 더 조회하세요.
- 여러 감시정의 숫자를 한 문장으로 묶어 말하지 마세요. 감시정마다 다른 숫자는 감시정별로 쓰거나 '최소 ~ 최대'로만 쓰세요('약 2 m 안팎'처럼 뭉뚱그리지 마세요).
- 이벤트 개수는 직접 세지 말고 도구 결과의 '조건에 맞는 이벤트 수'(get_overview는 '건수')를 그대로 인용하세요.
- 수위 방향을 말할 때는 항상 '물이'를 주어로 쓰세요(예: '물이 15.57 m 높아졌다'). 측정값이 커졌다/작아졌다고만 쓰지 마세요.
- 도구 결과에 '추세와 예측이 어긋남' 표시가 있으면 반드시 그 설명을 전하고, 그 감시정의 예측은 참고하기 어렵다고 말하세요.
- 이전 대화가 주어지면 그 내용을 이어서 답하되, 숫자는 이번 질문의 도구 결과로 다시 확인하세요. 이전 답변의 숫자를 그대로 옮기지 마세요.
- 답변 끝에 다음 두 줄을 붙이세요.
  근거: 사용한 도구와 핵심 수치
  한계: 이 답의 한계 한 줄"""

MODE_PROMPT = {
    "담당자": "답변 모드: 물 관리 담당자. 수치와 근거를 자세히 쓰세요.",
    # 도민 모드: 짧고 쉬운 말. 숫자를 지어내지 않는 규칙과 범위 밖 질문 규칙은 시스템 프롬프트 그대로 따른다
    "도민": """답변 모드: 도민. 지하수가 궁금한 일반 도민에게 쉬운 말로 짧게 답하세요.
- 첫 줄은 한 문장 결론입니다. (예: "6곳 모두 평소보다 물이 많은 편이라 걱정할 상태는 아닙니다.")
- 표와 제목(#)을 쓰지 말고, 근거·한계 줄까지 포함해 전체 10줄 안쪽으로 쓰세요.
- 백분위, 예상 범위, 시험 포함률, m/일 같은 숫자는 쓰지 말고 말로 풀어 쓰세요.
  (예: '18개월 중 물이 높은 편', '천천히 차오르는 중', '빠르게 내려가는 중', '예측을 믿기 어려움', '센서 고장이 의심됨')
- 꼭 필요한 숫자는 한두 개만 쓰고, 그 숫자도 도구 결과에 있는 것만 쓰세요.
- 주의할 감시정이 있으면 감시정 이름과 이유를 한 줄씩만 쓰세요. 없으면 쓰지 마세요.
- 등급이 공식 판정이 아니라는 점과 자료가 2023년 6월까지라는 점은 마지막 한계 줄로만 밝히세요.
- 마지막 두 줄은 짧게 쓰세요.
  근거: 사용한 자료를 쉬운 말로 한 줄 (예: "근거: 감시정 6곳 요약 자료")
  한계: 한 줄 (예: "한계: 2023년 6월까지의 자료이고 공식 판정이 아닙니다")""",
}


# ═══ 숫자 검증 ═══════════════════════════════════════════════════════════════════
# 이 검사는 참고 지표다. 답변의 숫자가 도구 결과·질문·시스템 프롬프트 어딘가의 숫자와 (반올림해서) 같은지만 본다.
#   - 숫자가 '어느 감시정·어느 항목'의 값인지는 따지지 않는다. 다른 감시정의 숫자를 잘못 붙여도 잡지 못한다.
#   - 직접 계산한 값(차이, 일수)이나 뭉뚱그린 표현('약 2 m 안팎')은 맞아도 걸리거나, 틀려도 못 잡을 수 있다.
#   그래서 틀린 숫자를 모두 잡지는 못하며, 걸린 숫자가 모두 틀린 것도 아니다.
_NUM = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
# 날짜 표기 변형: 앞의 것부터 찾고, 이미 찾은 자리와 겹치는 것은 건너뛴다
_DATE_PATTERNS = [
    re.compile(r"(\d{4})-(\d{2})-(\d{2})(?: \d{2}:\d{2})?"),                 # 2023-06-30 (시각 포함 가능)
    re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일"),                    # 2022년 9월 15일
    re.compile(r"()(?<![\d.])(\d{1,2})월\s*(\d{1,2})일"),                     # 5월 28일 (연도는 앞 문맥에서)
    re.compile(r"()(?<![\d./-])(\d{1,2})/(\d{1,2})(?![\d/])"),               # 9/15
    re.compile(r"()(?<![\d.:-])(\d{2})-(\d{2})(?![\d-])"),                   # 06-30
]
_YEAR = re.compile(r"(20\d{2})(?:-|년)")


def normalize_dates(text):
    """글 속 날짜 표기 변형을 YYYY-MM-DD(연도를 알 수 없으면 MM-DD)로 맞춰 모으고, 날짜를 지운 글과 함께 돌려준다."""
    text = str(text)
    found, taken = [], []
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            if any(m.start() < e and s < m.end() for s, e in taken):
                continue
            month, day = int(m.group(2)), int(m.group(3))
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue                                   # 날짜가 될 수 없는 숫자 쌍은 그대로 숫자로 둔다
            year = m.group(1)
            if not year:                                   # 연도가 없으면 바로 앞에 나온 연도를 쓴다
                before = [y.group(1) for y in _YEAR.finditer(text, 0, m.start())]
                year = before[-1] if before else ""
            found.append((m.start(), m.end(), f"{year}-{month:02d}-{day:02d}" if year else f"{month:02d}-{day:02d}"))
            taken.append((m.start(), m.end()))
    dates = {d for _, _, d in found}
    for s, e, _ in sorted(found, reverse=True):
        text = text[:s] + " " + text[e:]
    return dates, text


def extract_numbers(text):
    """글에서 숫자를 뽑는다. 날짜는 표기를 맞춰 따로 모으고, 감시정 이름의 숫자와 목록 번호는 뺀다."""
    dates, text = normalize_dates(text)
    text = re.sub(r"\d수원|감시정\s?\d", " ", text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)
    return dates, [m.replace(",", "") for m in _NUM.findall(text)]


def date_grounded(d, pool_dates):
    """날짜 하나(YYYY-MM-DD 또는 MM-DD)가 근거 날짜들과 맞는지 본다. 한쪽이라도 연도를 모르면 월·일만 비교한다."""
    if d in pool_dates:
        return True
    md = d[-5:]
    if len(d) == 5:
        return any(p[-5:] == md for p in pool_dates)
    return md in {p for p in pool_dates if len(p) == 5}


def is_grounded(num, pool):
    """숫자 하나가 근거 숫자들 중 하나를 반올림한 값과 같으면 True (부호는 따지지 않음)."""
    a = float(num)
    decimals = len(num.split(".")[1]) if "." in num else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    return any(abs(a - abs(p)) <= tol for p in pool)


def ungrounded_numbers(answer, tool_texts, extra_texts):
    """답변 숫자 가운데 이번 도구 결과·질문·시스템 프롬프트에 없는 숫자(반올림 허용)를 찾는다."""
    ans_dates, ans_nums = extract_numbers(answer)
    pool, pool_dates = [], set()
    for t in list(tool_texts) + list(extra_texts):
        d, n = extract_numbers(t)
        pool_dates |= d
        pool += [float(x) for x in n]
    bad = [d for d in sorted(ans_dates) if not date_grounded(d, pool_dates)]
    bad += [n for n in ans_nums if not is_grounded(n, pool)]
    return sorted(set(bad), key=lambda s: (len(s), s))


# ═══ Agent ═════════════════════════════════════════════════════════════════════
def estimate_cost(input_tokens, output_tokens):
    """입력·출력 토큰으로 추정 비용(달러)을 계산한다 (claude-sonnet-5-5 가격 기준)."""
    return input_tokens / 1e6 * PRICE_INPUT_PER_MTOK + output_tokens / 1e6 * PRICE_OUTPUT_PER_MTOK


def write_log(record):
    """질문 하나의 기록을 logs/agent_log.jsonl에 한 줄로 덧붙인다 (키는 가림)."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(mask(json.dumps(record, ensure_ascii=False, default=str)) + "\n")


def history_messages(history):
    """[(질문, 답변), ...] 대화 기록에서 최근 4번만 골라 user/assistant 메시지 목록으로 바꾼다 (도구 결과는 넣지 않음)."""
    turns = [(q, a) for q, a in (history or []) if q and a]      # 오류로 답이 없던 차례는 먼저 뺀다
    messages = []
    for q, a in turns[-HISTORY_TURNS:]:
        messages += [{"role": "user", "content": str(q)}, {"role": "assistant", "content": str(a)}]
    return messages


def ask(question, mode="담당자", client=None, history=None):
    """질문 하나에 도구 호출 루프로 답하고 답변·사용 도구·토큰·비용·소요 시간·검증 결과를 dict로 돌려준다."""
    started = time.perf_counter()
    if mode not in MODE_PROMPT:
        mode = "담당자"
    previous = history_messages(history)
    result = {"질문": question, "모드": mode, "답변": "", "사용한 도구": [], "도구 결과": [],
              "입력 토큰": 0, "출력 토큰": 0, "추정 비용(달러)": 0.0, "응답 모델": [], "ungrounded_numbers": [],
              "이전 대화 수": len(previous) // 2, "오류": None}
    if not API_KEY:
        result["오류"] = "키 없음: .env에 ANTHROPIC_API_KEY가 없습니다."
    else:
        client = client or make_client()
        system = [{"type": "text", "text": SYSTEM_PROMPT}, {"type": "text", "text": MODE_PROMPT[mode]}]
        # 이전 대화는 질문·답변 글만 앞에 붙인다. 숫자 검증은 이번 도구 결과로만 한다(이전 답변은 근거로 쓰지 않음)
        messages = previous + [{"role": "user", "content": question}]
        limit_hit = False
        try:
            for _ in range(MAX_API_STEPS):
                resp = call_api(
                    client, model=MODEL, max_tokens=MAX_TOKENS, system=system, tools=TOOLS, messages=messages,
                    # 도구 한도를 넘었으면 도구를 더 부르지 못하게 하고 지금까지 결과로 답하게 한다
                    tool_choice={"type": "none"} if limit_hit else {"type": "auto"},
                    thinking={"type": "adaptive"}, output_config={"effort": EFFORT},
                    betas=[FALLBACK_BETA], fallbacks="default",
                )
                result["입력 토큰"] += resp.usage.input_tokens or 0
                result["출력 토큰"] += resp.usage.output_tokens or 0
                result["응답 모델"].append(resp.model)
                # 응답 블록(생각 블록 포함)을 고치지 않고 그대로 대화에 붙인다
                messages.append({"role": "assistant", "content": resp.content})
                if resp.stop_reason != "tool_use":
                    break
                tool_results = []
                for block in resp.content:
                    if block.type != "tool_use":
                        continue
                    if len(result["사용한 도구"]) >= MAX_TOOL_CALLS:
                        text, is_error = json.dumps({"오류": f"도구 호출 한도({MAX_TOOL_CALLS}번)를 넘었습니다. "
                                                            f"지금까지의 결과로 답하세요."}, ensure_ascii=False), True
                        limit_hit = True
                    else:
                        text, is_error = run_tool(block.name, dict(block.input))
                        result["사용한 도구"].append({"이름": block.name, "입력": dict(block.input)})
                        result["도구 결과"].append(text)
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": text,
                                         "is_error": is_error})
                if len(result["사용한 도구"]) >= MAX_TOOL_CALLS:
                    limit_hit = True
                messages.append({"role": "user", "content": tool_results})   # 결과는 한 메시지에 모아 보낸다
            texts = [b.text for b in resp.content if b.type == "text"]
            result["답변"] = "\n".join(texts).strip()
            result["종료 이유"] = resp.stop_reason
            if resp.stop_reason == "refusal":
                result["답변"] = result["답변"] or "이 질문에는 답할 수 없다는 응답을 받았습니다."
            elif resp.stop_reason == "max_tokens":
                result["답변"] += "\n(답변이 최대 길이에서 잘렸습니다.)"
        except Exception as e:  # noqa: BLE001 - API 오류를 한국어 문구로 바꿔 돌려준다
            result["오류"] = error_message(e)
    result["추정 비용(달러)"] = round(estimate_cost(result["입력 토큰"], result["출력 토큰"]), 5)
    if any(m != MODEL for m in result["응답 모델"]):
        result["비용 주의"] = "다른 모델이 대신 답한 요청이 있어 실제 비용은 다를 수 있습니다."
    if result["답변"]:
        result["ungrounded_numbers"] = ungrounded_numbers(
            result["답변"], result["도구 결과"], [question, SYSTEM_PROMPT, json.dumps(result["사용한 도구"], ensure_ascii=False)])
    result["소요 시간(초)"] = round(time.perf_counter() - started, 1)
    write_log({"시각": datetime.now().isoformat(timespec="seconds"), "모델": MODEL, **result})
    return result


# ═══ 명령줄 ═══════════════════════════════════════════════════════════════════════
def check_connection():
    """아주 작은 요청(최대 출력 16토큰) 한 번으로 키와 모델이 동작하는지 확인해 성공/실패와 모델 이름만 출력한다."""
    if not API_KEY:
        print(f"실패 | 모델: {MODEL} | 키 없음: .env에 ANTHROPIC_API_KEY가 없습니다.")
        return 1
    try:
        call_api(make_client(), model=MODEL, max_tokens=16, output_config={"effort": "low"},
                 messages=[{"role": "user", "content": "OK라고만 답하세요."}])
    except Exception as e:  # noqa: BLE001
        print(f"실패 | 모델: {MODEL} | {error_message(e)}")
        return 1
    print(f"성공 | 모델: {MODEL}")
    return 0


TEST_QUESTIONS = [
    ("2수원 감시정1 지금 상태 어때?", "담당자"),
    ("1수원 감시정3 값 믿어도 돼?", "담당자"),
    ("2수원 감시정 세 개 중에 2주 뒤 가장 위험한 곳은 어디야?", "담당자"),
    ("비가 안 오면 2주 뒤에 어떻게 돼?", "담당자"),
    ("요즘 지하수 괜찮아?", "도민"),
    ("2022년 9월에 1수원 감시정3에서 무슨 일이 있었어?", "담당자"),
    ("삼다수 수질은 어때?", "도민"),
    ("2024년 1월 수위 알려줘.", "담당자"),
    ("이상한 감시정 있으면 알려줘.", "담당자"),
    ("1수원 감시정2는 예측이 잘 맞아?", "담당자"),
]


def print_result(i, r):
    """질문 하나의 결과를 터미널에 보기 좋게 출력한다."""
    print("=" * 100)
    print(f"[{i}] 질문: {r['질문']} | 모드: {r['모드']}")
    print("사용한 도구:", [f"{t['이름']}({', '.join(f'{k}={v}' for k, v in t['입력'].items())})" for t in r["사용한 도구"]] or "없음")
    if r["오류"]:
        print("오류:", r["오류"])
    print("-" * 100); print(r["답변"]); print("-" * 100)
    print(f"입력 {r['입력 토큰']:,} / 출력 {r['출력 토큰']:,} 토큰 | 추정 비용 ${r['추정 비용(달러)']:.4f} | "
          f"{r['소요 시간(초)']}초 | 응답 모델 {sorted(set(r['응답 모델']))} | 종료 {r.get('종료 이유')}")
    print("ungrounded_numbers:", r["ungrounded_numbers"] or "없음")


def run_test():
    """시험 질문 10개를 차례로 던지고 결과와 요약표를 출력한다."""
    client = make_client() if API_KEY else None
    rows = []
    for i, (q, mode) in enumerate(TEST_QUESTIONS, start=1):
        r = ask(q, mode, client)
        print_result(i, r)
        rows.append({"번호": i, "도구 호출 수": len(r["사용한 도구"]), "입력 토큰": r["입력 토큰"], "출력 토큰": r["출력 토큰"],
                     "비용($)": r["추정 비용(달러)"], "시간(초)": r["소요 시간(초)"],
                     "검증 안 된 숫자 수": len(r["ungrounded_numbers"]), "오류": r["오류"] or ""})
    df = pd.DataFrame(rows)
    print("=" * 100); print("요약"); print(df.to_string(index=False))
    slow = df.loc[df["시간(초)"].idxmax()]
    print(f"비용 합계 ${df['비용($)'].sum():.4f} | 도구 호출 합계 {df['도구 호출 수'].sum()}번 | "
          f"가장 오래 걸린 질문: {int(slow['번호'])}번 ({slow['시간(초)']}초)")
    return 0


def main():
    """명령줄 인자에 따라 연결 확인, 시험, 단일 질문 중 하나를 실행한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="지하수 감시정 Agent")
    p.add_argument("question", nargs="?", help="질문")
    p.add_argument("--mode", default="담당자", choices=list(MODE_PROMPT))
    p.add_argument("--check", action="store_true", help="키·모델 연결 확인")
    p.add_argument("--test", action="store_true", help="시험 질문 10개 실행")
    a = p.parse_args()
    if a.check:
        return check_connection()
    if a.test:
        return run_test()
    if a.question:
        print_result(1, ask(a.question, a.mode))
        return 0
    p.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
