"""감시정 지하수 상태 진단: 기준 시각의 지하수가 이 감시정의 평소에 비해 정상/주의/경고 중 어디인지 알려준다.

- Agent는 get_status(well, as_of)를 호출한다.
- 입력: data/processed/groundwater_hourly.csv (src/preprocess.py 결과)
        data/processed/sensor_events.csv (src/sensor_check.py 결과)
- 측정값의 방향(깊이인지 해발 수위인지)은 src/config.py의 VALUE_IS_DEPTH 하나로 정한다.
  원본 측정값은 바꾸지 않고, 등급은 '물 높이 = 부호 × 측정값' 기준으로 판정한다.
- 등급은 공식 기준수위가 아니라, 2022-01 ~ 2023-06 (18개월) 관측 안에서의 상대적 상태다.
- 기준 시각은 데이터 범위(2022-01-01 00:00 ~ 2023-06-30 23:00) 안이어야 한다 (docs/PLAN.md '현재 기준').

실행(확인용): python src/status_check.py
  → 6개 감시정 × 기준 시각 3개 결과, 감시정별 등급 일수, 경고가 몰린 달과 zone 강수, data/processed/status_levels.png
"""
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                                          # 측정값 방향 (VALUE_IS_DEPTH)
from sensor_check import detect_steps, remove_steps   # 계단 변화를 뺀 측정값을 만들 때 쓴다

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "groundwater_hourly.csv"
EVENTS_PATH = ROOT / "data" / "processed" / "sensor_events.csv"
OUT_FIG = ROOT / "data" / "processed" / "status_levels.png"

WELLS = ["1수원감시정1", "1수원감시정2", "1수원감시정3", "2수원감시정1", "2수원감시정2", "2수원감시정3"]

# ──────────────────────────────────────────────────────────────────────────────
# 등급 임계값 (근거: 2022-01 ~ 2023-06 하루 중앙값 분포. 2026-11-14에 물 높이 기준으로 방향을 바꿈)
#
# [물 높이 백분위] 기준 날의 물 높이(부호 × 하루 중앙값)가 18개월 전체 중 낮은 쪽에서 몇 %인가
#   - 백분위 기준은 정의상 감시정마다 물이 가장 낮은 10%의 날(약 55일)이 늘 '경고'가 된다.
#     그래서 '경고가 너무 많은가'는 비율이 아니라 경고가 몰리는 시기(와 그때의 강수)로 판단한다.
WARN_PERCENTILE = 10      # 물이 낮은 쪽 10% 이하 → 경고
CAUTION_PERCENTILE = 25   # 물이 낮은 쪽 25% 이하 → 주의
#
# [물이 빠르게 내려가는 속도] 최근 7일 물 높이의 하루 평균 변화가 자기 분포에서 낮은 쪽 몇 %인가
#   - 감시정마다 오르내리는 폭이 달라서 m 단위 고정값 대신 자기 분포의 백분위를 쓴다.
#   - 5%로 하면 경고 일수가 전체의 12~15%로 늘어 '매우 빠른 변화'만 고르도록 2.5%로 정했다 (2026-11-13 확인).
#   - 계단 변화(예: 2수원 감시정2 2022-04-20 0.86 m)가 속도에 섞이지 않게,
#     센서 이상 진단의 계단 변화를 뺀 측정값으로 변화량을 계산한다.
DROP_PERCENTILE = 2.5     # 물이 내려가는 속도가 자기 분포 낮은 쪽 2.5% 이하 → 경고
#   단, 물 높이 백분위가 50% 이하일 때만 이 규칙으로 경고한다 (2026-11-14 결정).
#   물이 관측 기간 중 높은 편인데 빠르게 내려가는 것은 큰 비 뒤 차오른 물이 빠지는 중일 수 있어,
#   등급은 백분위 기준 그대로 두고 '빠르게 내려가는 중' 표시만 붙인다 (예: 1수원 감시정3 2022-09-30, 백분위 96%).
DROP_RULE_MAX_PERCENTILE = 50
CHANGE_DAYS = (7, 14)     # 보여줄 최근 변화 기간(일). 등급에는 7일만 쓴다
#
# [같은 시기 과거 비교] 2022년 같은 날짜 앞뒤 3일(7일)의 하루 중앙값 중앙값과 비교한다
SAME_PERIOD_HALF_WINDOW = 3
#
# [센서 이상 겹침] 기준 시각 이전 7일 안에 수위 센서 이상 이벤트가 있으면 신뢰도 낮음 가능성을 표시한다
EVENT_LOOKBACK_DAYS = 7
# ──────────────────────────────────────────────────────────────────────────────

DATA_NOTE = "데이터는 2022-01-01 00:00부터 2023-06-30 23:00까지입니다."
GRADE_NOTE = "이 등급은 공식 기준수위가 아니라 2022-01~2023-06 (18개월) 관측 안에서의 상대적 상태입니다."


def direction_note():
    """측정값 방향 가정을 설명하는 문장을 돌려준다 (config.VALUE_IS_DEPTH에 따라 바뀜)."""
    if config.VALUE_IS_DEPTH:
        return "측정값은 물까지의 깊이로 가정했습니다(값이 클수록 지하수가 낮음). 명세로 확인하지 못한 가정입니다."
    return "측정값은 해발 수위로 가정했습니다(값이 클수록 지하수가 높음). 명세로 확인하지 못한 가정입니다."


@lru_cache(maxsize=1)
def load_data():
    """시간별 측정값, 하루 중앙값, 계단 변화를 뺀 하루 중앙값, 센서 이상 이벤트를 읽어 한 번만 준비한다."""
    df = pd.read_csv(IN_PATH, index_col="datetime", parse_dates=True, encoding="utf-8-sig")
    hourly = df.filter(like="level_").rename(columns=lambda c: c.replace("level_", ""))
    daily = hourly.resample("D").median()
    # 변화 속도 계산용: 계단 변화를 뺀 측정값의 하루 중앙값
    corrected = remove_steps(hourly, detect_steps(hourly)).resample("D").median()
    events = pd.read_csv(EVENTS_PATH, encoding="utf-8-sig", parse_dates=["시작 시각", "종료 시각"])
    return hourly, daily, corrected, events


def daily_change(series, days):
    """하루 중앙값의 최근 days일 하루 평균 변화량(m/일)을 날마다 계산한다."""
    return (series - series.shift(days)) / days


def water(series):
    """측정값을 물 높이 방향 값으로 바꾼다 (클수록 물이 높음). 판정에만 쓰고 화면에는 원본 측정값을 보인다."""
    return config.water_sign() * series


@lru_cache(maxsize=None)
def drop_threshold(well):
    """감시정 자신의 7일 물 높이 하루 평균 변화 분포에서 낮은 쪽 2.5% 값(물 높이 m/일)을 구한다."""
    _, _, corrected, _ = load_data()
    return float(daily_change(water(corrected[well]), 7).quantile(DROP_PERCENTILE / 100))


def water_percentile(history, value_now):
    """측정값 하나가 18개월 하루 중앙값 중 물이 낮은 쪽에서 몇 %인지 구한다."""
    return float((water(history) <= config.water_sign() * value_now).mean() * 100)


def is_fast_drop(water_change7, drop_thr):
    """최근 7일 물 높이 변화가 이 감시정 기록 중 가장 빠르게 내려가는 2.5% 안인지 본다."""
    return pd.notna(water_change7) and water_change7 <= drop_thr


def grade_from(percentile, water_change7, drop_thr):
    """물 높이 백분위와 7일 물 높이 변화로 등급(정상/주의/경고), 판정 이유, '빠르게 내려가는 중' 표시를 정한다."""
    reasons = []
    fast = is_fast_drop(water_change7, drop_thr)
    if percentile <= WARN_PERCENTILE:
        reasons.append(f"지하수가 18개월 중 낮은 쪽 {WARN_PERCENTILE}% 이하")
    # 빠른 하강 규칙은 물이 낮은 편(백분위 50% 이하)일 때만 경고로 쓴다
    if fast and percentile <= DROP_RULE_MAX_PERCENTILE:
        reasons.append(f"최근 7일 물이 내려가는 속도가 이 감시정 기록 중 가장 빠른 {DROP_PERCENTILE}% 안 (물이 낮은 편일 때)")
    note = None
    if fast and percentile > DROP_RULE_MAX_PERCENTILE:
        note = "빠르게 내려가는 중: 물은 높은 편이라 경고로 보지 않음 (큰 비 뒤 차오른 물이 빠지는 중일 수 있음)"
    if reasons:
        return "경고", reasons, note
    if percentile <= CAUTION_PERCENTILE:
        return "주의", [f"지하수가 18개월 중 낮은 쪽 {CAUTION_PERCENTILE}% 이하"], note
    return "정상", ["지하수가 낮은 쪽 25%보다 높고, 물이 낮은 편이면서 빠르게 내려가는 상태가 아님"], note


def parse_as_of(as_of, hourly):
    """기준 시각을 Timestamp로 바꾼다. 없으면 마지막 시각, 데이터 범위 밖이면 None을 돌려준다."""
    if as_of is None:
        return hourly.index.max()
    t = pd.Timestamp(as_of)
    if len(str(as_of)) <= 10:          # 날짜만 주면 그날의 마지막 시각(23:00)으로 본다
        t = t + pd.Timedelta(hours=23)
    if t < hourly.index.min() or t > hourly.index.max():
        return None
    return t


def same_period(daily, well, day):
    """기준 날이 2023년 1~6월이면 2022년 같은 시기와 비교한 수치를, 아니면 비교 불가 사유를 돌려준다."""
    if not (day.year == 2023 and day.month <= 6):
        return {"가능": False, "사유": "같은 시기 과거 자료는 2023년 1~6월 기준일 때만 있음(2022년 같은 달과 비교)"}
    last_year = day - pd.DateOffset(years=1)
    w = pd.Timedelta(days=SAME_PERIOD_HALF_WINDOW)
    ref = daily[well].loc[last_year - w:last_year + w]
    month = daily[well][(daily.index.year == 2022) & (daily.index.month == day.month)]
    diff = float(daily[well].loc[day] - ref.median())
    return {
        "가능": True,
        "2022년 같은 날짜(±3일) 측정값(m)": round(float(ref.median()), 2),
        "2022년 대비 측정값 차이(m)": round(diff, 2),
        "2022년 대비 해석": "2022년 같은 시기보다 " + config.water_compare(diff),
        f"2022년 {day.month}월 측정값 범위(m)": f"{month.min():.2f} ~ {month.max():.2f}",
    }


def overlapping_events(events, well, t):
    """기준 시각 이전 7일과 겹치는 이 감시정의 센서 이상 이벤트를 찾는다."""
    start = t - pd.Timedelta(days=EVENT_LOOKBACK_DAYS)
    hit = events[(events["감시정"] == well) & (events["시작 시각"] <= t) & (events["종료 시각"] >= start)]
    return [f"{r['센서']} {r['판정']} ({r['시작 시각']:%Y-%m-%d %H시} ~ {r['종료 시각']:%Y-%m-%d %H시})"
            for _, r in hit.iterrows()], bool((hit["센서"] == "수위").any())


def describe(well, t, grade, value_now, pct, change7, low_trust, fast_note=None):
    """쉬운 설명 문장 한 줄을 만든다 (물 기준 표현, 등급이 상대적 상태라는 점과 방향 가정을 항상 밝힌다)."""
    side = "낮은 편" if pct <= 50 else "높은 편"
    if pd.notna(change7):
        trend = f"최근 7일 동안 하루 평균 {config.water_move(change7)}"
    else:
        trend = "최근 7일 변화는 데이터 시작 직후라 계산할 수 없습니다"
    trust = " 최근 7일 안에 수위 센서 이상 의심이 있어 측정값을 그대로 믿기 어려울 수 있습니다." if low_trust else ""
    fast = (" 물이 빠르게 내려가는 중이지만 아직 높은 편이라 경고로 보지 않았습니다"
            "(큰 비 뒤 차오른 물이 빠지는 중일 수 있음).") if fast_note else ""
    return (f"{well}의 {t:%Y-%m-%d} 지하수는 18개월 관측 중 {side}(물이 낮은 쪽에서 {pct:.0f}% 위치, "
            f"측정값 {value_now:.2f} m)입니다. {trend.rstrip('.')}. 그래서 '{grade}'입니다.{fast}{trust} "
            f"{GRADE_NOTE} {direction_note()}")


def get_status(well, as_of=None):
    """감시정 하나의 기준 시각 상태(등급, 측정값, 비교 근거, 센서 이상 겹침, 설명)를 dict로 돌려준다."""
    hourly, daily, corrected, events = load_data()
    if well not in WELLS:
        return {"오류": f"알 수 없는 감시정입니다: {well}. 가능한 값: {', '.join(WELLS)}"}
    t = parse_as_of(as_of, hourly)
    if t is None:
        return {"오류": f"기준 시각 {as_of}은(는) 데이터 범위 밖입니다. {DATA_NOTE}"}

    day = t.normalize()
    # 최근 측정값: 기준 날의 0시부터 기준 시각까지의 하루 중앙값
    value_now = float(hourly[well].loc[day:t].median())
    history = daily[well]
    pct = water_percentile(history, value_now)
    changes = {d: daily_change(corrected[well], d).get(day) for d in CHANGE_DAYS}   # 측정값 기준 변화
    drop_thr = drop_threshold(well)                                                  # 물 높이 기준
    water_change7 = None if pd.isna(changes[7]) else config.water_sign() * changes[7]
    grade, reasons, fast_note = grade_from(pct, water_change7, drop_thr)
    event_list, low_trust = overlapping_events(events, well, t)
    label = config.value_label()

    return {
        "감시정": well,
        "기준 시각": f"{t:%Y-%m-%d %H:%M}",
        "등급": grade,
        "측정값 신뢰도": "측정값 신뢰도 낮음 가능성" if low_trust else "특이사항 없음",
        "판정 이유": reasons,
        "빠르게 내려가는 중": fast_note,
        f"최근 {label} (일 중앙값)": round(value_now, 2),
        "물 높이 백분위(낮은 쪽부터, %)": round(pct, 1),
        f"18개월 {label} 범위": f"{history.min():.2f} ~ {history.max():.2f}",
        "최근 7일 측정값 변화(m/일)": None if pd.isna(changes[7]) else round(float(changes[7]), 3),
        "최근 14일 측정값 변화(m/일)": None if pd.isna(changes[14]) else round(float(changes[14]), 3),
        "물이 매우 빠르게 내려가는 기준(측정값 m/일)": round(config.water_sign() * drop_thr, 3),
        "같은 시기 과거 비교": same_period(daily, well, day),
        "최근 7일 센서 이상 이벤트": event_list,
        "설명": describe(well, t, grade, value_now, pct, changes[7], low_trust, fast_note),
        "주의사항": GRADE_NOTE + " " + DATA_NOTE + " " + direction_note(),
    }


def daily_grades():
    """전체 기간의 날마다(그날 23시 기준) 감시정별 등급을 계산한 표를 돌려준다 (그림·일수 집계용)."""
    hourly, daily, corrected, _ = load_data()
    out = pd.DataFrame(index=daily.index, columns=WELLS, dtype=object)
    for well in WELLS:
        history = daily[well]
        change7 = daily_change(water(corrected[well]), 7)
        thr = drop_threshold(well)
        for day in daily.index:
            pct = water_percentile(history, history.loc[day])
            out.loc[day, well] = grade_from(pct, change7.loc[day], thr)[0]   # 등급만 쓴다
    return out


def plot(grades):
    """감시정별 측정값 선 그래프 위에 날마다의 등급을 색 띠로 표시해 저장한다 (위쪽이 물이 높은 쪽)."""
    plt.rcParams["font.family"] = "Malgun Gothic"     # 한글 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    _, daily, _, _ = load_data()
    # 상태 색(상태 전용 팔레트): 주의 = warning, 경고 = critical. 정상은 배경 없음
    colors = {"주의": "#fab219", "경고": "#d03b3b"}
    fig, axes = plt.subplots(len(WELLS), 1, figsize=(14, 2.4 * len(WELLS)), sharex=True)
    for ax, well in zip(axes, WELLS):
        g = grades[well]
        for label, c in colors.items():
            on = (g == label)
            # 같은 등급이 이어지는 날들을 하나의 띠로 묶어 그린다
            block = (on != on.shift()).cumsum()
            for _, days in g[on].groupby(block[on]):
                ax.axvspan(days.index[0], days.index[-1] + pd.Timedelta(days=1), color=c, alpha=0.28, lw=0, zorder=0)
        ax.plot(daily.index, daily[well], color="#52514e", lw=0.9, zorder=2)
        config.orient_axis(ax)
        ax.set_title(well, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_ylabel("측정값(m)", fontsize=9, color="#52514e")
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    handles = [plt.Line2D([], [], color="#52514e", lw=1)] + \
              [plt.Line2D([], [], color=c, lw=7, alpha=0.4) for c in colors.values()]
    labels = [config.value_label() + " 일 중앙값, 배경 없음 = 정상", "주의", "경고"]
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    flip = " · y축 반전: 위로 갈수록 물이 높음" if config.VALUE_IS_DEPTH else ""
    fig.suptitle(f"감시정별 지하수 상태 등급 (18개월 관측 안에서의 상대 위치, 공식 기준수위 아님{flip})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_FIG, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def warning_months_with_rain(grades):
    """감시정별 경고 일수를 달마다 세고, 같은 달의 zone별 일평균 강수 월합계(mm)를 옆에 붙인다."""
    from rain_analysis import zone_rain
    warn = (grades == "경고").resample("MS").sum().astype(int)
    rain = zone_rain()[["해안", "중산간", "산지"]].resample("MS").sum(min_count=1).round(0)
    rain.columns = [f"강수_{c}" for c in rain.columns]
    out = warn.join(rain)
    out.index = out.index.strftime("%Y-%m")
    return out


def main():
    """확인용: 기준 시각 3개 결과표, 감시정별 등급 일수, 경고 달과 강수, 1수원 감시정3 변화기 등급을 출력하고 그림을 저장한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 60)
    print("방향 설정: VALUE_IS_DEPTH =", config.VALUE_IS_DEPTH, "|", direction_note())

    rows = []
    for as_of in [None, "2022-09-30", "2023-03-15"]:
        for well in WELLS:
            r = get_status(well, as_of)
            sp = r["같은 시기 과거 비교"]
            rows.append({
                "기준": r["기준 시각"], "감시정": well, "등급": r["등급"], "신뢰도": r["측정값 신뢰도"],
                "측정값": r[f"최근 {config.value_label()} (일 중앙값)"], "물높이백분위": r["물 높이 백분위(낮은 쪽부터, %)"],
                "7일(m/일)": r["최근 7일 측정값 변화(m/일)"], "14일(m/일)": r["최근 14일 측정값 변화(m/일)"],
                "2022대비": sp.get("2022년 대비 해석", "비교 불가"),
                "빠른하강표시": "있음" if r["빠르게 내려가는 중"] else "",
            })
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n설명 예:", get_status("1수원감시정3")["설명"])
    print("설명 예:", get_status("2수원감시정1", "2022-09-30")["설명"])

    grades = daily_grades()
    counts = grades.apply(lambda s: s.value_counts()).reindex(["정상", "주의", "경고"]).fillna(0).astype(int)
    print("\n감시정별 등급 일수 (전체 546일):"); print(counts.to_string())
    wm = warning_months_with_rain(grades)
    print("\n달별 경고 일수와 zone 강수 월합계(mm):"); print(wm.to_string())

    w = "1수원감시정3"
    for a, b in [("2022-09-01", "2022-10-31"), ("2023-05-01", "2023-06-30")]:
        g = grades[w].loc[a:b]
        print(f"{w} {a}~{b} 등급 일수:", g.value_counts().to_dict())
    plot(grades)
    print("저장:", OUT_FIG)
    return 0


if __name__ == "__main__":
    sys.exit(main())
