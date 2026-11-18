"""제주 삼다수 취수원 지하수 감시정 Agent 화면 (Streamlit).

실행: streamlit run src/app.py
- 채팅처럼 질문하면 agent.ask(대화 기록 포함)를 불러 답을 보여 준다.
- 키는 agent 모듈이 .env에서 이 프로세스 안에서만 읽는다. 화면에는 키를 절대 보이지 않는다.
"""
import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import agent                                   # Claude API Agent (도구 호출)
import forecast as fc                          # 예측 모델 만들기 버튼용

ROOT = Path(__file__).resolve().parent.parent

TITLE = "제주 삼다수 취수원 지하수 Agent"
SUBTITLE = "표선유역 감시정 6곳의 지하수 상태, 센서 이상 의심, 2주 예측을 말로 물어보세요."
DATA_NOTICE = ("데이터는 2022-01-01 ~ 2023-06-30까지입니다. '지금'은 2023-06-30 기준입니다. "
               "등급은 공식 기준이 아니라 18개월 관측 안의 상대적 상태이고, "
               "측정값은 물까지의 깊이로 가정했습니다(명세로 확인하지 못함).")
FOOTER = ("이 서비스의 분석은 규칙과 통계 모델이 하고, AI는 도구를 골라 부르고 결과를 설명합니다. "
          "답변은 공식 판정이 아닙니다.")
EXAMPLE_QUESTIONS = [
    "2수원 감시정1 지금 상태 어때?",
    "요즘 지하수 괜찮아?",
    "이상한 감시정 있으면 알려줘.",
    "1수원 감시정3 값 믿어도 돼?",
    "비가 안 오면 2주 뒤에 어떻게 돼?",
    "2022년 9월에 1수원 감시정3에서 무슨 일이 있었어?",
]
MODES = ["담당자", "도민"]


# ═══ 화면과 무관한 계산 (단위 시험 대상) ════════════════════════════════════════
def chart_paths(result):
    """ask 결과의 도구 결과 중 make_chart가 만든 그림 파일 경로(실제로 있는 것만, 중복 없이)를 돌려준다."""
    paths = []
    for tool, text in zip(result.get("사용한 도구", []), result.get("도구 결과", [])):
        if tool.get("이름") != "make_chart":
            continue
        try:
            rel = json.loads(text).get("경로")
        except (json.JSONDecodeError, AttributeError):
            continue
        if rel:
            p = (ROOT / rel).resolve()
            if p.exists() and p not in paths:
                paths.append(p)
    return paths


def session_totals(results):
    """이번 세션의 ask 결과 목록에서 (질문 수, 추정 비용 합계 달러)를 계산한다."""
    return len(results), round(sum(r.get("추정 비용(달러)", 0.0) or 0.0 for r in results), 5)


def history_pairs(messages):
    """화면의 대화 목록에서 ask에 넘길 [(질문, 답변), ...]을 만든다 (답이 없던 차례는 빼고, 자르는 건 ask가 한다)."""
    pairs, question = [], None
    for m in messages:
        if m["role"] == "user":
            question = m["content"]
        elif m["role"] == "assistant" and question is not None:
            if m.get("content"):
                pairs.append((question, m["content"]))
            question = None
    return pairs


def safe_markdown(text):
    """답변의 '~'(취소선)와 '$'(수식)가 마크다운으로 해석되지 않게 이스케이프한다 ('245.29~280.50' 보호)."""
    return str(text).replace("~", "\\~").replace("$", "\\$")


def tool_lines(result):
    """사용한 도구를 '이름(입력)' 한 줄씩의 목록으로 만든다."""
    return [f"{t['이름']}({', '.join(f'{k}={v}' for k, v in t['입력'].items())})" for t in result.get("사용한 도구", [])]


def visibility(mode, dev):
    """모드와 '개발자 정보 보기' 여부로 무엇을 보일지 정한다: {사용한 도구, 비용·시간·토큰, 추정 비용 합계}."""
    return {"사용한 도구": mode == "담당자" or dev,      # 담당자는 항상, 도민은 개발자 정보를 켰을 때만
            "비용·시간·토큰": dev,
            "추정 비용 합계": dev}


# ═══ 화면 조각 ═══════════════════════════════════════════════════════════════════
def show_answer(result, show):
    """답변 말풍선 안에 답변(또는 오류 안내), 그림, 그리고 show에 따라 사용한 도구·비용·시간·토큰 접이식을 그린다."""
    if result.get("오류"):
        st.error(result["오류"])
    if result.get("답변"):
        st.markdown(safe_markdown(result["답변"]))
    for p in chart_paths(result):
        st.image(str(p))
    if show["사용한 도구"]:
        tools = tool_lines(result)
        with st.expander(f"사용한 도구 ({len(tools)}개)"):
            st.markdown("\n".join(f"- `{t}`" for t in tools) if tools else "도구를 쓰지 않았습니다.")
    if not show["비용·시간·토큰"]:
        return
    with st.expander("비용·시간·토큰"):
        st.markdown(f"- 추정 비용: ${result.get('추정 비용(달러)', 0):.4f}\n"
                    f"- 소요 시간: {result.get('소요 시간(초)', 0)}초\n"
                    f"- 입력 {result.get('입력 토큰', 0):,} / 출력 {result.get('출력 토큰', 0):,} 토큰\n"
                    f"- 응답 모델: {', '.join(sorted(set(result.get('응답 모델', [])))) or '-'}")
        if result.get("비용 주의"):
            st.caption(result["비용 주의"])
        if result.get("ungrounded_numbers"):
            st.caption("도구 결과에서 찾지 못한 숫자(참고 지표): " + ", ".join(result["ungrounded_numbers"]))


def model_notice():
    """예측 모델 파일이 없거나 맞지 않으면 상단 안내와 '예측 모델 만들기(약 2분)' 버튼을 보여 준다."""
    problem = agent.model_problem()
    if not problem:
        return
    st.info(problem.replace(" 먼저 python src/forecast.py를 실행하세요.", "") +
            " 예측 질문에 답하려면 아래 버튼으로 모델을 한 번 만들어 주세요.")
    if st.button("예측 모델 만들기(약 2분)"):
        try:
            with st.spinner("예측 모델을 만드는 중… (약 2분)"):
                fc.load_bundle.cache_clear()        # 파일이 없거나 맞지 않으면 학습해서 파일로 저장한다
                fc.load_bundle()
            st.success("예측 모델을 만들었습니다.")
            st.rerun()
        except Exception as e:  # noqa: BLE001 - 화면이 멈추지 않게 안내만 한다
            st.error(f"예측 모델을 만들지 못했습니다: {type(e).__name__}")


def sidebar():
    """사이드바(모드 선택, 예시 질문, 대화 지우기, 세션 지표, 개발자 정보 보기)를 그리고 (모드, 보일 항목)을 돌려준다."""
    with st.sidebar:
        mode = st.radio("사용자 모드", MODES, horizontal=True,
                        help="담당자: 수치와 근거를 자세히 / 도민: 쉬운 말로 짧게")
        st.subheader("예시 질문")
        for i, q in enumerate(EXAMPLE_QUESTIONS):
            if st.button(q, key=f"ex{i}", width="stretch"):
                st.session_state.pending = q
        st.divider()
        if st.button("대화 지우기", width="stretch"):
            st.session_state.messages = []
        stats = st.container()                      # 지표 자리를 먼저 잡고, 체크박스는 맨 아래에 둔다
        dev = st.checkbox("개발자 정보 보기", value=False, help="비용·시간·토큰과 도구 사용 내역을 보여 줍니다.")
        show = visibility(mode, dev)
        n, cost = session_totals(st.session_state.results)
        with stats:
            if show["추정 비용 합계"]:
                c1, c2 = st.columns(2)
                c1.metric("이번 세션 질문", f"{n}개")
                c2.metric("추정 비용 합계", f"${cost:.3f}")
            else:
                st.metric("이번 세션 질문", f"{n}개")
    return mode, show


def main():
    """화면 전체를 그리고, 새 질문이 있으면 ask를 불러 답을 덧붙인다."""
    st.set_page_config(page_title=TITLE, layout="centered")
    st.session_state.setdefault("messages", [])     # 화면에 보이는 대화 [{role, content, result}]
    st.session_state.setdefault("results", [])      # 이번 세션의 ask 결과 (대화를 지워도 비용 집계는 남김)

    st.title(TITLE)
    st.caption(SUBTITLE)
    st.warning(DATA_NOTICE)
    if not agent.API_KEY:
        st.error("키 없음: .env에 ANTHROPIC_API_KEY가 없습니다. .env.example을 .env로 복사하고 키를 채운 뒤 다시 실행하세요.")
    model_notice()
    mode, show = sidebar()          # 보일 항목은 지금 고른 모드·체크박스 기준으로 지난 답변에도 똑같이 적용한다

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            if m["role"] == "user":
                st.markdown(safe_markdown(m["content"]))
            else:
                show_answer(m["result"], show)

    question = st.chat_input("지하수에 대해 물어보세요") or st.session_state.pop("pending", None)
    if question:
        history = history_pairs(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(safe_markdown(question))
        with st.chat_message("assistant"):
            with st.spinner("분석 중…"):
                try:
                    result = agent.ask(question, mode, history=history)
                except Exception as e:  # noqa: BLE001 - 예상 못 한 오류도 화면이 멈추지 않게 안내로 바꾼다
                    result = {"답변": "", "오류": f"처리 중 오류가 났습니다: {type(e).__name__}", "사용한 도구": [],
                              "도구 결과": [], "추정 비용(달러)": 0.0}
            show_answer(result, show)
        st.session_state.messages.append({"role": "assistant", "content": result.get("답변", ""), "result": result})
        st.session_state.results.append(result)
        st.rerun()                                  # 사이드바의 질문 수·비용을 바로 갱신한다

    st.divider()
    st.caption(FOOTER)


if __name__ == "__main__":      # streamlit run 때 실행 (시험에서 함수만 불러 쓸 때는 실행하지 않음)
    main()
