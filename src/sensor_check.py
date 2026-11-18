"""지하수 감시정 센서 이상 진단: 순간 튐, 계단 변화, 값 고착, 추세 이탈, 변동성 변화를 찾는다.

- 입력: data/processed/groundwater_hourly.csv (src/preprocess.py 결과)
- 출력: data/processed/sensor_events.csv (이벤트 표)
        data/processed/sensor_events_level.png (수위 차트에 이상 구간 표시)
- 원인은 단정하지 않는다. 모든 판정은 '~ 의심'으로 표시하고 근거 수치를 함께 적는다.
- 이벤트마다 '직전 큰 비' 맥락(시작 전 7일 강수 합계와 그 순위)을 덧붙인다. 판정 자체는 바꾸지 않는다.

실행: python src/sensor_check.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config                        # 측정값 방향 (그래프 축 방향·이름에만 씀)

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "groundwater_hourly.csv"
OUT_EVENTS = ROOT / "data" / "processed" / "sensor_events.csv"
OUT_FIG = ROOT / "data" / "processed" / "sensor_events_level.png"

# ──────────────────────────────────────────────────────────────────────────────
# 임계값 (근거: 2022-01 ~ 2023-06 시간별 데이터의 분포, 2026-11-13 확인)
#
# 감시정마다 평소 움직임이 크게 다르다.
#   - 시간당 수위 변화 |Δ|의 99.99% 값: 2수원 감시정 0.12~0.13 m, 1수원 감시정1 0.11 m,
#     1수원 감시정3 0.77 m, 1수원 감시정2 1.69 m
#   - 1수원 감시정2는 2022-01-11부터 하루에도 여러 번 약 2 m씩 오르내리는 진동이 계속된다
#     (일교차 중앙값 0.84~1.43 m).
# 그래서 하나의 고정값을 쓰지 않고, "직전 7일 동안의 평소 움직임 × 배수"와
# "고정 하한" 중 큰 값을 임계값으로 쓴다. 같은 진동이 계속되면 평소 움직임으로 학습되어
# 더 이상 경보를 내지 않는다.
# ──────────────────────────────────────────────────────────────────────────────

# 직전 기준 기간: 7일(168시간). 하루 주기 변동을 여러 번 포함할 만큼 길고, 계절 변화는 덜 섞이는 길이
BASELINE_HOURS = 168
# 직전 기간의 "평소 움직임"으로 99% 값을 쓴다 (가끔 있는 큰 값 1~2개에는 흔들리지 않게)
BASELINE_QUANTILE = 0.99

# [순간 튐] 값과 앞뒤 13시간 중앙값의 차이(잔차)로 판단한다.
#   중앙값 창이 13시간이므로 6시간 이내로 솟았다 돌아오는 변화만 큰 잔차를 만든다.
SPIKE_WINDOW = 13
#   고정 하한 0.3 m: 조용한 감시정(2수원, 1수원1)의 잔차 99.99% 값이 0.07~0.12 m라 그 2배 이상
SPIKE_MIN_M = 0.3
#   배수 3: 직전 7일 잔차 99% 값의 3배를 넘으면 평소와 다르다고 본다
SPIKE_K = 3.0
#   튐으로 볼 최대 지속 시간(시간). 더 길면 튐이 아니라 계단/추세 쪽으로 본다
SPIKE_MAX_HOURS = 6

# [계단 변화] 앞 6시간 중앙값과 뒤 6시간 중앙값의 차이(점프)로 판단한다.
STEP_WINDOW = 6
#   고정 하한 0.3 m: 2수원 감시정의 6시간 점프 99.9% 값이 0.135~0.14 m라 그 2배 이상
STEP_MIN_M = 0.3
STEP_K = 3.0
#   유지 확인: 점프 7~24시간 뒤에도 점프 크기의 50% 이상이 같은 방향으로 남아 있어야 계단으로 본다
STEP_HOLD_FROM, STEP_HOLD_TO = 7, 24
STEP_HOLD_RATIO = 0.5

# [값 고착] 소수 둘째 자리까지 같은 값이 이어진 시간
#   수위 24시간: 6개 감시정의 연속 같은 값 99.9% 값이 7~15시간, 최대 12~48시간
STUCK_LEVEL_HOURS = 24
#   수온 720시간(30일): 수온은 0.01 °C 단위라 원래 같은 값이 오래 이어진다
#   (연속 같은 값 99% 값 25~248시간). 30일은 모든 감시정의 99% 값보다 3배 이상 길다
STUCK_TEMP_HOURS = 720

# [추세 이탈] 일평균 수위로 판단한다. 이벤트는 2일 이내 간격이면 하나로 묶는다.
TREND_MERGE_GAP_DAYS = 2
#   같은 수원 비교(2수원): 7일 변화량이 3개 감시정 7일 변화량의 중앙값과 얼마나 다른가
#     (계단 변화로 잡힌 순간의 점프는 빼고 비교한다)
#     0.25 m: 계단 보정 후 2수원 감시정 1의 7일 차이 최대값 0.235 m보다 조금 크다
GROUP_TREND_MIN_M = 0.25
GROUP_TREND_DAYS = 7
#   다른 수원 비교(1수원 감시정1·2): 7일 변화량이 2수원 3개 감시정 7일 변화량의 중앙값과 얼마나 다른가
#     계산 방식은 2수원과 같다. 다만 1수원 감시정1·2는 2수원보다 크게 오르내려서
#     (7일 변화량 기울기: 1수원1은 2수원 중앙값의 1.75배) 2수원의 0.25 m를 쓰면
#     546일 중 216일(1수원1), 255일(1수원2)이 걸린다. 그래서 임계값은 감시정 자신의
#     이탈량 분포에서 중앙값 + 5 × (MAD 기반 표준편차)로 정하고, 하한은 2수원과 같은 0.25 m로 둔다
REF_TREND_K = 5.0
#   자기 과거 비교(1수원 감시정3): 직전 14일 기울기로 7일 뒤를 연장한 값과 실제 값의 차이
SELF_SLOPE_DAYS = 14
SELF_AHEAD_DAYS = 7
#     차이가 감시정 자신의 전체 기간 중앙값 + 5 × (MAD 기반 표준편차)를 넘으면 이탈
#     (정규분포라면 약 100만분의 1 확률, 분포가 꼬리가 길어도 상위 1% 안쪽만 남는다)
SELF_TREND_K = 5.0
#     고정 하한 0.5 m: 1수원 감시정1·2의 7일 연장 오차 90% 값이 0.65~0.77 m
SELF_TREND_MIN_M = 0.5

# [변동성 변화] 하루 변동폭(하루 최대 − 최소)이 직전 평소 수준과 크게 달라진 상태가 며칠 이어지는가
#   하루 변동폭 중앙값: 2수원 0.11~0.12 m, 1수원1 0.06 m, 1수원3 0.13 m, 1수원2 0.98 m
#   직전 평소 수준: 직전 14일 하루 변동폭의 중앙값 (2주면 며칠짜리 큰 날에 흔들리지 않는다)
VOL_BASE_DAYS = 14
#   데이터 시작 부분: 직전 기간이 7일 이상 쌓인 날부터 판단한다 (2022-01-01~07은 판단하지 않음)
VOL_MIN_BASE_DAYS = 7
#   배수 3: 평소의 3배 이상(커짐) 또는 1/3 이하(작아짐). 조용한 감시정들의 '하루 변동폭 ÷ 직전 평소' 비율
#     99% 값이 1.7~1.9라 3배는 평소 흔들림을 넘는다
VOL_RATIO = 3.0
#   절대 차이 0.3 m 이상: 2수원 하루 변동폭 99% 값이 0.20~0.22 m라, 변동폭이 작은 감시정에서
#     0.05 → 0.15 m 같은 작은 차이가 비율만으로 걸리는 것을 막는다
VOL_MIN_DIFF_M = 0.3
#   평소 수준이 0에 가까울 때 비율이 터지지 않게 하는 하한 (수위 측정 단위 0.01 m의 5배)
VOL_BASE_FLOOR_M = 0.05
#   3일 이상 이어질 때만 상태 변화로 본다 (하루 이틀은 순간 튐·계단 변화 쪽에서 본다)
VOL_MIN_DAYS = 3

# [동시 발생] 다른 감시정의 같은 유형 이벤트가 ±3시간 안에 있으면 '동시'로 표시
SIMULTANEOUS_HOURS = 3

# [직전 큰 비] 이벤트 시작 전 7일(시작일 전날까지)의 zone 평균 일강수량 합계를 본다
#   기준 분포: 전체 기간(2022-01 ~ 2023-06)의 모든 '7일 합계'. 그 안에서 상위 5% 안이면 강수 반응일 수 있다고 표시한다
#   강수 자료가 빈 날짜(2022-01-20)는 채우지 않고 합계에서 뺀다 (7일 중 6일 이상 있을 때만 합계를 낸다)
RAIN_LOOKBACK_DAYS = 7
RAIN_MIN_DAYS = 6
RAIN_TOP_PERCENT = 5.0

# 추세 이탈 비교 방식 (docs/PLAN.md 5장 '이상 판정 기준')
GROUP_WELLS = ["2수원감시정1", "2수원감시정2", "2수원감시정3"]   # 같은 수원 내 비교
REF_WELLS = ["1수원감시정1", "1수원감시정2"]                    # 2수원 3개 중앙값과 비교
SELF_WELLS = ["1수원감시정3"]                                   # 자기 과거와 비교

TYPE_LABEL = {"spike": "순간 튐", "step": "계단 변화", "stuck": "값 고착", "trend": "추세 이탈",
              "volatility": "변동성 변화"}


def load():
    """전처리된 시간별 표를 읽어 수위·수온 표 두 개로 나눠 돌려준다 (컬럼 이름은 감시정 이름만 남김)."""
    df = pd.read_csv(IN_PATH, index_col="datetime", parse_dates=True, encoding="utf-8-sig")
    level = df.filter(like="level_").rename(columns=lambda c: c.replace("level_", ""))
    temp = df.filter(like="temp_").rename(columns=lambda c: c.replace("temp_", ""))
    return level, temp


def adaptive_threshold(stat, k, floor):
    """직전 7일 |stat|의 99% 값 × k와 고정 하한 중 큰 값을 시각마다 계산한다."""
    base = stat.abs().rolling(BASELINE_HOURS, min_periods=24).quantile(BASELINE_QUANTILE).shift(1)
    return np.maximum(base.fillna(0) * k, floor)


def runs(mask, max_gap=0):
    """True가 이어지는 구간을 (시작 위치, 끝 위치) 목록으로 돌려준다. max_gap 이하 빈틈은 이어 붙인다."""
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > max_gap + 1:
            out.append((start, prev))
            start = i
        prev = i
    out.append((start, prev))
    return out


def make_event(well, sensor, kind, start, end, evidence, hours):
    """이벤트 한 건을 표의 한 행(dict)으로 만든다."""
    return {
        "감시정": well, "센서": sensor, "유형": TYPE_LABEL[kind],
        "시작 시각": start, "종료 시각": end, "지속 시간(h)": hours,
        "판정": f"{TYPE_LABEL[kind]} 의심", "근거": evidence,
    }


def detect_spikes(level):
    """순간 튐: 앞뒤 13시간 중앙값에서 크게 벗어났다가 6시간 안에 돌아온 구간을 찾는다."""
    events = []
    for well in level:
        x = level[well]
        resid = x - x.rolling(SPIKE_WINDOW, center=True, min_periods=1).median()
        thr = adaptive_threshold(resid, SPIKE_K, SPIKE_MIN_M)
        for s, e in runs(resid.abs() > thr, max_gap=1):
            hours = e - s + 1
            if hours > SPIKE_MAX_HOURS:
                continue
            seg = resid.iloc[s:e + 1]
            peak_t = seg.abs().idxmax()
            rate = x.diff().abs().iloc[s:e + 2].max()
            evidence = (f"최대 벗어남 {seg[peak_t]:+.2f} m ({peak_t:%m-%d %H시}), "
                        f"임계값 {thr[peak_t]:.2f} m, 최대 시간당 변화 {rate:.2f} m/h, "
                        f"{hours}시간 뒤 복귀")
            events.append(make_event(well, "수위", "spike", x.index[s], x.index[e], evidence, hours))
    return events


def detect_steps(level):
    """계단 변화: 앞뒤 6시간 중앙값이 크게 달라지고, 그 차이가 하루 가까이 유지되는 시점을 찾는다."""
    events = []
    n = STEP_WINDOW
    for well in level:
        x = level[well]
        before = x.rolling(n).median()                         # t-5 ~ t
        after = x[::-1].rolling(n).median()[::-1].shift(-1)    # t+1 ~ t+6
        jump = after - before
        thr = adaptive_threshold(jump, STEP_K, STEP_MIN_M)
        # 유지 확인용: t+7 ~ t+24 중앙값
        hold = x[::-1].rolling(STEP_HOLD_TO - STEP_HOLD_FROM + 1).median()[::-1].shift(-STEP_HOLD_FROM)
        kept = (hold - before) / jump
        cand = (jump.abs() > thr) & (kept >= STEP_HOLD_RATIO)
        for s, e in runs(cand, max_gap=2):
            seg = jump.iloc[s:e + 1]
            t = seg.abs().idxmax()
            # 6시간 중앙값 차이는 실제 변화 시각보다 앞서 커질 수 있으므로,
            # 앞뒤 6시간 안에서 시간당 변화가 가장 큰 시각을 실제 변화 시각으로 본다
            d = x.diff().abs().loc[t - pd.Timedelta(hours=n):t + pd.Timedelta(hours=n)]
            change_t = d.idxmax()
            evidence = (f"점프 {jump[t]:+.2f} m (앞 6시간 {before[t]:.2f} → 뒤 6시간 {after[t]:.2f}), "
                        f"임계값 {thr[t]:.2f} m, 최대 시간당 변화 {d.max():.2f} m/h, "
                        f"7~24시간 뒤 유지율 {kept[t]:.0%}")
            # 시작·종료: 가장 크게 변한 1시간 (직전 시각 → 변화 시각)
            events.append(make_event(well, "수위", "step", change_t - pd.Timedelta(hours=1), change_t, evidence, 1))
    return events


def detect_stuck(df, sensor, min_hours):
    """값 고착: 같은 값이 min_hours 이상 이어진 구간을 찾는다 (수위·수온 공통)."""
    events = []
    for well in df:
        x = df[well]
        group = (x != x.shift()).cumsum()
        for _, seg in x.groupby(group):
            if len(seg) >= min_hours:
                evidence = f"값 {seg.iloc[0]} 이(가) {len(seg)}시간({len(seg) / 24:.1f}일) 동안 변하지 않음, 기준 {min_hours}시간"
                events.append(make_event(well, sensor, "stuck", seg.index[0], seg.index[-1], evidence, len(seg)))
    return events


def remove_steps(level, step_events):
    """계단 변화로 잡힌 순간의 점프를 0으로 바꿔, 계단이 추세 비교에 섞이지 않게 한 수위를 돌려준다."""
    out = level.copy()
    for ev in step_events:
        well = ev["감시정"]
        d = out[well].diff()
        win = slice(ev["시작 시각"] - pd.Timedelta(hours=STEP_WINDOW), ev["시작 시각"] + pd.Timedelta(hours=STEP_WINDOW))
        # 점프 구간에서 평소(직전 중앙값)보다 큰 변화만 지운다
        normal = d.loc[:ev["시작 시각"]].tail(BASELINE_HOURS).abs().median()
        big = d.loc[win].abs() > max(normal * 5, 0.05)
        d.loc[big[big].index] = 0
        out[well] = out[well].iloc[0] + d.fillna(0).cumsum()
    return out


def trend_events(well, dev, thr, basis):
    """일별 이탈량(dev)이 임계값을 넘는 날을 묶어 추세 이탈 이벤트로 만든다."""
    events = []
    for s, e in runs(dev.abs() > thr, max_gap=TREND_MERGE_GAP_DAYS):
        seg = dev.iloc[s:e + 1]
        t = seg.abs().idxmax()
        days = e - s + 1
        evidence = (f"{basis}: 최대 이탈 {seg[t]:+.2f} m ({t:%Y-%m-%d}), 임계값 {thr:.2f} m, "
                    f"{days}일 지속")
        end = dev.index[e] + pd.Timedelta(hours=23)
        events.append(make_event(well, "수위", "trend", dev.index[s], end, evidence, days * 24))
    return events


def robust_threshold(dev, k, floor):
    """이탈량 분포의 중앙값과 MAD 기반 표준편차로 (중앙값, 임계값)을 구한다. 임계값은 floor보다 작아지지 않는다."""
    center = dev.median()
    sigma = 1.4826 * (dev - center).abs().median()
    return center, max(k * sigma, floor)


def detect_trend_group(daily):
    """같은 수원 비교(2수원): 7일 변화량이 3개 감시정 중앙값에서 크게 벗어난 기간을 찾는다."""
    change = daily[GROUP_WELLS].diff(GROUP_TREND_DAYS)
    dev = change.sub(change.median(axis=1), axis=0)
    events = []
    for well in GROUP_WELLS:
        basis = "PLAN.md 규칙: 같은 수원 비교(7일 변화량 − 2수원 3개 감시정 7일 변화량 중앙값)"
        events += trend_events(well, dev[well].dropna(), GROUP_TREND_MIN_M, basis)
    return events


def detect_trend_ref(daily):
    """다른 수원 비교(1수원 감시정1·2): 7일 변화량이 2수원 3개 감시정 중앙값에서 크게 벗어난 기간을 찾는다."""
    change = daily.diff(GROUP_TREND_DAYS)
    ref = change[GROUP_WELLS].median(axis=1)
    events = []
    for well in REF_WELLS:
        dev = (change[well] - ref).dropna()
        center, thr = robust_threshold(dev, REF_TREND_K, GROUP_TREND_MIN_M)
        basis = ("PLAN.md 규칙: 2수원 3개 감시정 중앙값과 비교(7일 변화량 − 2수원 7일 변화량 중앙값, "
                 f"평소 차이 {center:+.2f} m를 뺀 값)")
        events += trend_events(well, dev - center, thr, basis)
    return events


def detect_trend_self(daily):
    """자기 과거 비교(1수원 감시정3): 직전 14일 기울기로 7일 뒤를 연장한 값과 실제 값이 크게 다른 기간을 찾는다."""
    events = []
    for well in SELF_WELLS:
        x = daily[well]
        slope = x.diff().rolling(SELF_SLOPE_DAYS).mean()          # 직전 14일 평균 일 변화량
        expected = x.shift(SELF_AHEAD_DAYS) + SELF_AHEAD_DAYS * slope.shift(SELF_AHEAD_DAYS)
        dev = (x - expected).dropna()
        center, thr = robust_threshold(dev, SELF_TREND_K, SELF_TREND_MIN_M)
        basis = "PLAN.md 규칙: 자기 과거 비교(14일 기울기로 7일 연장한 값과의 차이)"
        events += trend_events(well, dev - center, thr, basis)
    return events


def detect_volatility(level):
    """변동성 변화: 하루 변동폭이 직전 14일 평소의 3배 이상 또는 1/3 이하로 바뀐 상태가 3일 이상 이어진 기간을 찾는다."""
    day_range = level.resample("D").max() - level.resample("D").min()
    events = []
    for well in level:
        r = day_range[well]
        base = r.rolling(VOL_BASE_DAYS, min_periods=VOL_MIN_BASE_DAYS).median().shift(1)
        ratio = r / base.clip(lower=VOL_BASE_FLOOR_M)
        big_diff = (r - base).abs() >= VOL_MIN_DIFF_M
        for direction, cond in (("커짐", ratio >= VOL_RATIO), ("작아짐", ratio <= 1 / VOL_RATIO)):
            for s, e in runs((cond & big_diff).fillna(False)):
                days = e - s + 1
                if days < VOL_MIN_DAYS:
                    continue
                seg = slice(s, e + 1)
                # 배수는 날마다 그날의 직전 14일 평소와 비교한 값이다. 첫날 평소로 나눈 값과 다르므로
                #   커짐이면 최대 배수, 작아짐이면 최소 배수만 적고 평소가 날마다 바뀐다는 점을 밝힌다
                extreme = (f"최대 {ratio.iloc[seg].max():.1f}배" if direction == "커짐"
                           else f"최소 {ratio.iloc[seg].min():.1f}배")
                evidence = (f"하루 변동폭 {direction}: 이 기간 하루 변동폭 {r.iloc[seg].min():.2f}~{r.iloc[seg].max():.2f} m, "
                            f"매일 직전 14일 평소와 비교한 {extreme}(평소는 날마다 바뀜, 첫날 평소 {base.iloc[s]:.2f} m), "
                            f"{days}일 지속. 이후에는 바뀐 상태가 새 평소로 학습됨")
                end = r.index[e] + pd.Timedelta(hours=23)
                events.append(make_event(well, "수위", "volatility", r.index[s], end, evidence, days * 24))
    return events


def mark_simultaneous(events):
    """같은 유형·같은 센서의 이벤트가 다른 감시정에서 ±3시간 안에 있으면 근거에 동시 발생으로 적는다."""
    tol = pd.Timedelta(hours=SIMULTANEOUS_HOURS)
    for ev in events:
        others = sorted({o["감시정"] for o in events
                         if o is not ev and o["유형"] == ev["유형"] and o["센서"] == ev["센서"]
                         and o["감시정"] != ev["감시정"]
                         and o["시작 시각"] <= ev["종료 시각"] + tol and o["종료 시각"] >= ev["시작 시각"] - tol})
        ev["다른 감시정 동시 발생"] = ", ".join(others) if others else "없음"
    return events


def add_rain_context(ev):
    """이벤트마다 시작 전 7일 zone 평균 강수 합계, 전체 7일 합계 중 상위 몇 %인지, 큰 비 뒤 여부를 덧붙인다."""
    from rain_analysis import zone_rain    # 기상 자료가 필요할 때만 읽는다
    r = zone_rain()["평균"]
    # 날짜 d의 값 = d-7일 ~ d-1일 합계 (빈 날짜는 빼고, 6일 이상 있을 때만)
    before7 = r.rolling(RAIN_LOOKBACK_DAYS, min_periods=RAIN_MIN_DAYS).sum().shift(1)
    dist = r.rolling(RAIN_LOOKBACK_DAYS, min_periods=RAIN_MIN_DAYS).sum().dropna().to_numpy()
    totals, tops, labels = [], [], []
    for start in ev["시작 시각"]:
        v = before7.get(pd.Timestamp(start).normalize())
        if v is None or pd.isna(v):
            totals.append(None); tops.append(None); labels.append("강수 자료 부족")
            continue
        top = float((dist >= v).mean() * 100)          # 이 값 이상인 7일 합계의 비율 = 상위 몇 %
        totals.append(round(float(v), 1)); tops.append(round(top, 1))
        labels.append("큰 비 뒤의 변화: 강수 반응일 수 있음" if top <= RAIN_TOP_PERCENT else "없음")
    out = ev.copy()
    out["직전 7일 강수(mm, zone 평균)"] = totals
    out["직전 7일 강수 상위(%)"] = tops
    out["강수 맥락"] = labels
    return out


def plot(level, events):
    """감시정별 측정값 선 그래프 위에 이상 유형별 색으로 이상 구간을 표시해 저장한다 (위쪽이 물이 높은 쪽)."""
    plt.rcParams["font.family"] = "Malgun Gothic"     # 한글 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    # 유형별 색: 검증된 범주형 팔레트 1~5번 (파랑, 주황, 청록, 노랑, 분홍)
    colors = {"순간 튐": "#2a78d6", "계단 변화": "#eb6834", "값 고착": "#1baf7a", "추세 이탈": "#eda100",
              "변동성 변화": "#e87ba4"}
    spans = ("값 고착", "추세 이탈", "변동성 변화")   # 며칠 이상 이어지는 유형은 구간 음영으로 그린다
    fig, axes = plt.subplots(len(level.columns), 1, figsize=(14, 2.3 * len(level.columns)), sharex=True)
    for ax, well in zip(axes, level.columns):
        for ev in events:
            if ev["감시정"] != well or ev["센서"] != "수위":
                continue
            c = colors[ev["유형"]]
            if ev["유형"] not in spans:
                # 짧은 이벤트는 세로선으로 (구간으로 그리면 보이지 않음)
                ax.axvline(ev["시작 시각"], color=c, lw=1.2, alpha=0.9, zorder=1)
            else:
                ax.axvspan(ev["시작 시각"], ev["종료 시각"], color=c, alpha=0.3, lw=0, zorder=0)
        ax.plot(level.index, level[well], color="#52514e", lw=0.6, zorder=2)
        config.orient_axis(ax)
        ax.set_title(well, loc="left", fontsize=10, color="#0b0b0b")
        ax.set_ylabel("측정값(m)", fontsize=9, color="#52514e")
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
    handles = [plt.Line2D([], [], color=colors[k], lw=6 if k in spans else 2,
                          alpha=0.45 if k in spans else 1) for k in colors]
    fig.legend(handles, [f"{k} 의심" for k in colors], loc="upper right", ncol=5, frameon=False, fontsize=9)
    flip = ", y축 반전: 위로 갈수록 물이 높음" if config.VALUE_IS_DEPTH else ""
    fig.suptitle(f"감시정별 {config.value_label()}과 센서 이상 의심 구간 (2022-01 ~ 2023-06{flip})",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(OUT_FIG, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def run_all(level, temp):
    """네 가지 진단을 모두 실행해 이벤트 표(DataFrame)를 돌려준다."""
    spikes = detect_spikes(level)
    steps = detect_steps(level)
    stuck = detect_stuck(level, "수위", STUCK_LEVEL_HOURS) + detect_stuck(temp, "수온", STUCK_TEMP_HOURS)
    daily = remove_steps(level, steps).resample("D").mean()
    trends = detect_trend_group(daily) + detect_trend_ref(daily) + detect_trend_self(daily)
    volatility = detect_volatility(level)
    events = mark_simultaneous(spikes + steps + stuck + trends + volatility)
    cols = ["감시정", "센서", "유형", "시작 시각", "종료 시각", "지속 시간(h)", "판정", "근거", "다른 감시정 동시 발생"]
    return pd.DataFrame(events, columns=cols).sort_values(["감시정", "시작 시각", "유형"]).reset_index(drop=True)


def main():
    """진단을 실행하고 이벤트 표와 그림을 저장한 뒤, 감시정별·유형별 개수와 월평균 경보 수를 출력한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 200)
    level, temp = load()
    ev = add_rain_context(run_all(level, temp))
    ev.to_csv(OUT_EVENTS, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M", lineterminator="\n")
    plot(level, ev.to_dict("records"))

    months = (level.index.max() - level.index.min()).days / 30.44
    print("저장:", OUT_EVENTS, "|", OUT_FIG)
    print(f"이벤트 {len(ev)}건, 기간 {months:.1f}개월 → 월평균 {len(ev) / months:.2f}건")
    table = pd.crosstab(ev["감시정"] + " " + ev["센서"], ev["유형"], margins=True, margins_name="합계")
    print(table)
    print("\n강수 맥락:", ev["강수 맥락"].value_counts().to_dict())
    flagged = ev[ev["강수 맥락"].str.startswith("큰 비")]
    print(flagged[["감시정", "센서", "유형", "시작 시각", "직전 7일 강수(mm, zone 평균)", "직전 7일 강수 상위(%)"]]
          .to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
