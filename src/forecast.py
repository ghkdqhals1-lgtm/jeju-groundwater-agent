"""2주 수위 예측: 앞으로 1~14일의 수위를 예측한다.

[v0] 비 정보 없이 수위 자신의 과거만 쓴다.
- 모델: 감시정마다 릿지 회귀(numpy로 직접 구현). 입력 하나로 h = 1~14일 뒤 변화량 14개를 한꺼번에 푼다.
  h마다 따로 푼 것과 수식상 같고(가중치가 h마다 따로 나옴), 한 번의 행렬 계산으로 끝나서 이렇게 했다.
- 결과 파일 data/processed/forecast_eval_v0.* 는 v0 평가 때 저장한 것을 그대로 둔다 (main_v0로만 다시 만든다).

[v1] v0 입력에 강수를 더한다.
- 과거 강수: 기준일까지의 최근 누적 강수량 (zone과 기간은 감시정마다 검증 구간으로 고른다)
- 미래 강수: h일 뒤 예측에는 '기준일 다음 날부터 h일까지의 누적 강수량'을 넣는다. 그래서 h마다 모델을 따로 푼다.
- 사용 시에는 미래 강수를 알 수 없으므로 시나리오 3개(무강수 / 관측 기간 중앙값 수준 / 관측 기간 상위 10% 수준)로 예측한다.

공통
- 비교 기준: (1) 직전 값 유지, (2) 최근 7일 하루 평균 변화를 그대로 연장
- 분할: 학습 2022-01-01 ~ 2022-12-31, 검증 2023-01-01 ~ 2023-03-31 (설정 고르기),
        설정을 고른 뒤 2023-03-31까지 전체로 다시 학습, 최종 시험 2023-04-01 ~ 2023-06-30
- Agent는 forecast(well, as_of, horizon, rain_scenario)를 호출한다 (v1).

실행(확인용): python src/forecast.py
  → data/processed/forecast_eval_v1.csv, data/processed/forecast_eval_v1.png
"""
import json
import sys
from functools import lru_cache
from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 화면 없이 그림 파일만 저장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from status_check import parse_as_of, WELLS, DATA_NOTE  # 기준 시각 규칙을 상태 진단과 같게 쓴다
from rain_analysis import zone_rain as read_zone_rain    # zone별 일평균 강수량 (빈 날짜는 빈 값)
import config                                            # 측정값 방향 (설명 문장에만 씀)

ROOT = Path(__file__).resolve().parent.parent
IN_PATH = ROOT / "data" / "processed" / "groundwater_hourly.csv"
OUT_CSV = ROOT / "data" / "processed" / "forecast_eval_v0.csv"
OUT_FIG = ROOT / "data" / "processed" / "forecast_eval_v0.png"

# ──────────────────────────────────────────────────────────────────────────────
# 기간 분할 (기준일 t 기준. 목표값 t+h도 같은 구간 안에 있어야 그 구간의 표본으로 쓴다)
TRAIN_END = pd.Timestamp("2022-12-31")    # 설정 고르기용 학습 끝
VALID_END = pd.Timestamp("2023-03-31")    # 검증 끝 = 최종 학습 끝
TEST_END = pd.Timestamp("2023-06-30")     # 최종 시험 끝
MAX_H = 14                                # 최대 예측 일수
REPORT_H = [1, 3, 7, 14]                  # 보고할 예측 일수
# 1수원 감시정3 큰 하강 구간: 이 날 이후 기준일만 따로 평가 (센서 이상 진단의 변동성 변화가 알린 날)
DROP_SUBSET = ("1수원감시정3", pd.Timestamp("2023-05-08"))

# 입력: 기준일 t까지의 정보만 쓴다
CHANGE_LAGS = [1, 3, 7, 14]               # 최근 1·3·7·14일 변화량
RANGE_DAYS = 7                            # 최근 7일 시간별 수위 변동폭(최대 − 최소)
# 설정 후보 (검증 구간 평균 MAE가 가장 작은 것을 감시정마다 고른다)
#   규제 강도: 표본이 300개 남짓이라 0.1(거의 규제 없음) ~ 1000(거의 평균만 예측)까지 넓게 본다
RIDGE_LAMBDAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
#   기준일 수위 자체를 입력에 넣을지: 수위 높이에 따라 내려가는 속도가 다를 수 있지만,
#   18개월에 계절이 1.5번뿐이라 과적합될 수 있어 넣는 것과 빼는 것을 둘 다 본다
USE_LEVEL_OPTIONS = [True, False]
# ──────────────────────────────────────────────────────────────────────────────

RAIN_NOTE = "미래 강수는 알 수 없습니다. 이 예측은 비 정보 없이 수위의 최근 흐름만으로 계산했습니다."


@lru_cache(maxsize=1)
def load_series():
    """시간별 수위를 읽어 하루 중앙값 수위와 최근 7일 시간별 변동폭(날마다)을 만든다. 계단 변화는 빼지 않는다."""
    df = pd.read_csv(IN_PATH, index_col="datetime", parse_dates=True, encoding="utf-8-sig")
    hourly = df.filter(like="level_").rename(columns=lambda c: c.replace("level_", ""))
    daily = hourly.resample("D").median()
    # 기준일 t의 최근 7일 변동폭: t-6일 0시 ~ t일 23시의 최대 − 최소
    day_max, day_min = hourly.resample("D").max(), hourly.resample("D").min()
    range7 = day_max.rolling(RANGE_DAYS).max() - day_min.rolling(RANGE_DAYS).min()
    return hourly, daily, range7


def make_features(well, use_level):
    """기준일마다 입력 표(X)와 1~14일 뒤 변화량 표(Y)를 만든다. 입력은 기준일까지의 값만 쓴다."""
    _, daily, range7 = load_series()
    x = daily[well]
    feats = {f"변화_{k}일": x - x.shift(k) for k in CHANGE_LAGS}
    feats["변동폭_7일"] = range7[well]
    if use_level:
        feats["수위"] = x
    X = pd.DataFrame(feats)
    Y = pd.DataFrame({h: x.shift(-h) - x for h in range(1, MAX_H + 1)})
    return X, Y


def ridge_fit(X, Y, lam):
    """입력을 표준화한 뒤 릿지 회귀 가중치를 닫힌 식으로 구한다 (절편은 규제하지 않음)."""
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd = np.where(sd == 0, 1.0, sd)
    Xs = (X - mu) / sd
    y_mean = Y.mean(axis=0)
    W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(Xs.shape[1]), Xs.T @ (Y - y_mean))
    return {"mu": mu, "sd": sd, "W": W, "y_mean": y_mean}


def ridge_predict(model, X):
    """학습된 릿지 모델로 1~14일 뒤 변화량을 예측한다."""
    return ((X - model["mu"]) / model["sd"]) @ model["W"] + model["y_mean"]


def rows_between(X, Y, start, end, need_full_target=True):
    """기준일이 start~end 사이이고 입력이 모두 있는 행을 고른다. need_full_target이면 14일 뒤 목표값도 end 안에 있어야 한다."""
    idx = X.index[(X.index >= start) & (X.index <= end) & X.notna().all(axis=1)]
    if need_full_target:
        idx = idx[idx + pd.Timedelta(days=MAX_H) <= end]
    return idx


def mae_by_h(pred, Y, idx, end):
    """예측 일수 h마다, 목표 날짜(t+h)가 end 안에 있는 기준일만으로 평균절대오차를 구한다."""
    out = {}
    for h in range(1, MAX_H + 1):
        ok = idx[idx + pd.Timedelta(days=h) <= end]
        out[h] = (float(np.abs(pred.loc[ok, h] - Y.loc[ok, h]).mean()), len(ok))
    return out


def select_setting(well):
    """학습(~2022-12-31)으로 학습하고 검증(2023-01~03) 평균 MAE가 가장 작은 (규제 강도, 수위 입력 여부)를 고른다."""
    best = None
    for lam, use_level in product(RIDGE_LAMBDAS, USE_LEVEL_OPTIONS):
        X, Y = make_features(well, use_level)
        tr = rows_between(X, Y, X.index.min(), TRAIN_END)
        va = rows_between(X, Y, TRAIN_END + pd.Timedelta(days=1), VALID_END, need_full_target=False)
        model = ridge_fit(X.loc[tr].to_numpy(), Y.loc[tr].to_numpy(), lam)
        pred = pd.DataFrame(ridge_predict(model, X.loc[va].to_numpy()), index=va, columns=Y.columns)
        score = np.mean([m for m, _ in mae_by_h(pred, Y, va, VALID_END).values()])
        if best is None or score < best["검증 MAE"]:
            best = {"규제 강도": lam, "수위 입력": use_level, "검증 MAE": score}
    return best


@lru_cache(maxsize=None)
def final_model(well):
    """고른 설정으로 2023-03-31까지 전체(학습+검증)를 다시 학습한 모델과 설정을 돌려준다."""
    setting = select_setting(well)
    X, Y = make_features(well, setting["수위 입력"])
    idx = rows_between(X, Y, X.index.min(), VALID_END)
    model = ridge_fit(X.loc[idx].to_numpy(), Y.loc[idx].to_numpy(), setting["규제 강도"])
    return model, setting, len(idx)


def baselines(well, idx):
    """비교 기준 두 가지의 1~14일 뒤 변화량 예측을 만든다: 직전 값 유지(0), 최근 7일 기울기 연장."""
    _, daily, _ = load_series()
    x = daily[well]
    slope = ((x - x.shift(7)) / 7).loc[idx]
    persist = pd.DataFrame(0.0, index=idx, columns=range(1, MAX_H + 1))
    trend = pd.DataFrame({h: slope * h for h in range(1, MAX_H + 1)}, index=idx)
    return persist, trend


@lru_cache(maxsize=1)
def test_evaluation():
    """시험 구간(2023-04~06) 모든 기준일에서 모델·직전 값 유지·기울기 연장의 h별 MAE를 한 번만 계산한다."""
    records = []
    for well in WELLS:
        model, setting, _ = final_model(well)
        X, Y = make_features(well, setting["수위 입력"])
        idx = rows_between(X, Y, VALID_END + pd.Timedelta(days=1), TEST_END, need_full_target=False)
        pred = pd.DataFrame(ridge_predict(model, X.loc[idx].to_numpy()), index=idx, columns=Y.columns)
        persist, trend = baselines(well, idx)
        subsets = [("시험 전체", idx)]
        if well == DROP_SUBSET[0]:
            subsets.append((f"{DROP_SUBSET[1]:%Y-%m-%d} 이후", idx[idx >= DROP_SUBSET[1]]))
        for subset, sub_idx in subsets:
            for name, p in [("모델", pred), ("직전 값 유지", persist), ("기울기 연장", trend)]:
                for h, (mae, n) in mae_by_h(p, Y, sub_idx, TEST_END).items():
                    records.append({"감시정": well, "구간": subset, "방법": name, "예측일수": h, "MAE(m)": mae, "기준일 수": n})
    return pd.DataFrame(records)


def forecast_v0(well, as_of=None, horizon=14, rain_scenario=None):
    """v0(비 정보 없음) 예측 도구. v1이 생긴 뒤에는 비교·참고용으로만 남겨 둔다."""
    if rain_scenario is not None:
        return {"오류": "강수 가정별 예측(rain_scenario)은 아직 구현되지 않았습니다. 지금은 비 정보 없이 예측합니다.",
                "안내": DATA_NOTE}
    if well not in WELLS:
        return {"오류": f"알 수 없는 감시정입니다: {well}. 가능한 값: {', '.join(WELLS)}"}
    if not (isinstance(horizon, int) and 1 <= horizon <= MAX_H):
        return {"오류": f"예측 일수는 1~{MAX_H}일 사이 정수여야 합니다."}
    hourly, daily, _ = load_series()
    t = parse_as_of(as_of, hourly)
    if t is None:
        return {"오류": f"기준 시각 {as_of}은(는) 데이터 범위 밖입니다. {DATA_NOTE}"}
    # 기준일: 그날 23시 자료까지 있어야 하루 중앙값을 쓸 수 있으므로, 23시 전이면 전날을 기준일로 쓴다
    base = t.normalize() if t.hour == 23 else t.normalize() - pd.Timedelta(days=1)

    model, setting, n_train = final_model(well)
    X, _ = make_features(well, setting["수위 입력"])
    if base not in X.index or X.loc[base].isna().any():
        return {"오류": f"{base:%Y-%m-%d}은(는) 최근 14일 변화량을 계산할 수 없습니다(데이터 시작 직후). "
                        f"2022-01-15 이후를 기준일로 해 주세요.", "안내": DATA_NOTE}
    delta = ridge_predict(model, X.loc[[base]].to_numpy())[0]
    level0 = float(daily[well].loc[base])
    days = []
    for h in range(1, horizon + 1):
        d = base + pd.Timedelta(days=h)
        actual = daily[well].get(d)
        days.append({"날짜": f"{d:%Y-%m-%d}", "예측 수위(m)": round(level0 + float(delta[h - 1]), 2),
                     "실제 수위(m)": None if actual is None or pd.isna(actual) else round(float(actual), 2)})

    ev = test_evaluation()
    ev = ev[(ev["감시정"] == well) & (ev["구간"] == "시험 전체")]
    compare = {}
    for h in REPORT_H:
        m = ev[(ev["예측일수"] == h)].set_index("방법")["MAE(m)"]
        compare[f"{h}일 뒤"] = {"모델 MAE(m)": round(float(m["모델"]), 3),
                              "직전 값 유지 MAE(m)": round(float(m["직전 값 유지"]), 3),
                              "모델이 더 나음": bool(m["모델"] < m["직전 값 유지"])}
    notes = [RAIN_NOTE, DATA_NOTE]
    if base <= VALID_END:
        notes.append("기준일이 모델 학습 기간(2023-03-31까지) 안이라, 이 예측의 오차는 실제 사용 때보다 작게 보일 수 있습니다.")
    return {
        "감시정": well, "기준일": f"{base:%Y-%m-%d}", "기준 수위(일 중앙값, m)": round(level0, 2),
        "일별 예측": days,
        "시험 구간(2023-04~06) 오차 비교": compare,
        "모델 설정": {"규제 강도": setting["규제 강도"], "수위 입력": setting["수위 입력"], "학습 기준일 수": n_train},
        "안내": " ".join(notes),
    }


def plot_v0(ev):
    """v0: 감시정별로 예측일수에 따른 MAE를 세 방법의 선으로 그려 저장한다."""
    plt.rcParams["font.family"] = "Malgun Gothic"     # 한글 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    # 방법별 색: 검증된 범주형 팔레트 1~3번 (파랑, 주황, 청록)
    colors = {"모델": "#2a78d6", "직전 값 유지": "#eb6834", "기울기 연장": "#1baf7a"}
    panels = [(w, "시험 전체") for w in WELLS] + [(DROP_SUBSET[0], f"{DROP_SUBSET[1]:%Y-%m-%d} 이후")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, (well, subset) in zip(axes.flat, panels):
        sub = ev[(ev["감시정"] == well) & (ev["구간"] == subset)]
        for name, c in colors.items():
            s = sub[sub["방법"] == name].sort_values("예측일수")
            ax.plot(s["예측일수"], s["MAE(m)"], color=c, lw=2, marker="o", ms=3)
        ax.set_title(f"{well}" + ("" if subset == "시험 전체" else f"\n({subset} 기준일만)"),
                     loc="left", fontsize=10, color="#0b0b0b")
        ax.set_xticks([1, 3, 7, 14])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.set_xlabel("예측 일수(일)", fontsize=8, color="#52514e")
        ax.set_ylabel("MAE(m)", fontsize=8, color="#52514e")
    axes.flat[-1].axis("off")
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=4) for c in colors.values()]
    axes.flat[-1].legend(handles, ["릿지 모델(비 정보 없음)", "직전 값 유지", "최근 7일 기울기 연장"],
                         loc="center", frameon=False, fontsize=10)
    fig.suptitle("2주 예측 v0: 시험 구간(2023-04~06) 예측 일수별 평균절대오차 (낮을수록 좋음)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main_v0():
    """v0 확인용: 감시정별 고른 설정, 시험 구간 h별 MAE 표를 출력하고 v0 CSV와 그림을 저장한다 (기본 실행에서는 부르지 않음)."""
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("감시정별 설정 (검증 2023-01~03에서 선택):")
    for well in WELLS:
        _, s, n = final_model(well)
        print(f"  {well}: 규제 강도 {s['규제 강도']}, 수위 입력 {s['수위 입력']}, 검증 평균 MAE {s['검증 MAE']:.3f} m, 최종 학습 기준일 {n}개")

    ev = test_evaluation()
    ev.round(4).to_csv(OUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
    plot_v0(ev)
    table = ev[ev["예측일수"].isin(REPORT_H)].pivot_table(
        index=["감시정", "구간", "예측일수"], columns="방법", values="MAE(m)").round(3)
    table = table[["모델", "직전 값 유지", "기울기 연장"]]
    table["모델<유지"] = table["모델"] < table["직전 값 유지"]
    print("\n시험 구간 MAE(m):"); print(table.to_string())
    n = ev[ev["방법"] == "모델"].groupby(["감시정", "구간"])["기준일 수"].agg(["max", "min"])
    print("\n기준일 수(h=1 / h=14):"); print(n.to_string())
    print("저장:", OUT_CSV, "|", OUT_FIG)

    r = forecast_v0("2수원감시정1")
    print("\n도구 예시:", {k: r[k] for k in ("감시정", "기준일", "기준 수위(일 중앙값, m)")}, r["일별 예측"][:3], "...")
    print("  ", r["시험 구간(2023-04~06) 오차 비교"])
    print("  ", r["안내"])
    print("범위 밖 예:", forecast_v0("2수원감시정1", as_of="2023-07-05"))
    return 0


# ══════════════════════════════════════════════════════════════════════════════
# v1: 강수 포함
# ══════════════════════════════════════════════════════════════════════════════
OUT_CSV_V1 = ROOT / "data" / "processed" / "forecast_eval_v1.csv"
OUT_FIG_V1 = ROOT / "data" / "processed" / "forecast_eval_v1.png"
OUT_INTERVAL_V1 = ROOT / "data" / "processed" / "forecast_eval_v1_interval.csv"   # 예상 범위의 시험 포함률

# 과거 강수 입력 후보 (감시정마다 검증 구간 평균 MAE가 가장 작은 조합을 고른다)
#   zone: 해안·중산간·산지 일평균 강수량과 3개 zone의 평균
#   기간: 기준일까지 최근 3·7·14·30일 누적 (비와 수위의 시차 상관 확인에서 쓴 후보와 같다)
RAIN_ZONES = ["해안", "중산간", "산지", "평균"]
RAIN_WINDOWS = [3, 7, 14, 30]
# 미래 강수 시나리오: 학습 구간(2022-01 ~ 2023-03)의 zone별 14일 누적 강수량 분포에서 구한다.
#   h일 누적은 '14일 값 ÷ 14 × h' (같은 양이 매일 내린다는 단순 가정). 18개월 자료라 '평년'이 아니다.
SCENARIOS = {"무강수": None, "중앙값": 0.5, "상위10%": 0.9}   # 이름 → 분위수 (무강수는 0 mm)
SCENARIO_BASE_DAYS = 14
SCENARIO_FIT_END = VALID_END
# 1수원 감시정3은 시험 구간 큰 하강에서 오차가 매우 커서 항상 따로 알린다
BIG_DROP_WELL = "1수원감시정3"

RAIN_NOTE_V1 = ("미래 강수는 알 수 없습니다. 그래서 무강수 / 관측 기간 중앙값 수준 / 관측 기간 상위 10% 수준의 "
                "세 가지 가정으로 계산했습니다. 18개월 자료라 평년 값이 아니며, 같은 양이 매일 내린다고 단순하게 가정했습니다.")


@lru_cache(maxsize=1)
def rain():
    """zone별 일평균 강수량 표를 한 번만 읽는다 (받지 못한 날짜는 빈 값 그대로)."""
    return read_zone_rain()


def future_rain(r, h):
    """기준일 다음 날부터 h일까지(t+1 ~ t+h)의 누적 강수량을 날마다 계산한다. 빈 날짜가 끼면 빈 값."""
    return r.rolling(h).sum().shift(-h)


def make_features_v1(well, use_level, zone, window):
    """v0 입력에 '기준일까지 최근 window일 누적 강수'를 더한 X, 목표 Y, h별 미래 누적 강수 F를 만든다."""
    X, Y = make_features(well, use_level)
    r = rain()[zone]
    X = X.copy()
    X[f"과거_{window}일_강수"] = r.rolling(window).sum().reindex(X.index)
    F = pd.DataFrame({h: future_rain(r, h).reindex(X.index) for h in range(1, MAX_H + 1)})
    return X, Y, F


def rows_h(X, Y, F, h, start, end):
    """h일 뒤 예측에 쓸 기준일을 고른다: start~end 사이, 입력·미래 강수·목표가 모두 있고 t+h가 end 안."""
    ok = (X.index >= start) & (X.index <= end) & X.notna().all(axis=1) & F[h].notna() & Y[h].notna()
    idx = X.index[ok]
    return idx[idx + pd.Timedelta(days=h) <= end]


def fit_v1(X, Y, F, lam, end):
    """h = 1~14마다 [v0 입력 + 과거 강수 + h일 미래 강수]로 릿지 모델을 따로 학습한다 (기준일 t+h ≤ end)."""
    models = {}
    for h in range(1, MAX_H + 1):
        idx = rows_h(X, Y, F, h, X.index.min(), end)
        A = np.column_stack([X.loc[idx].to_numpy(), F.loc[idx, h].to_numpy()])
        models[h] = ridge_fit(A, Y.loc[idx, [h]].to_numpy(), lam)
    return models


def predict_v1(models, X, idx, h, future):
    """h일 뒤 변화량을 예측한다. future는 기준일마다 넣을 h일 미래 누적 강수(실제 값 또는 시나리오 값)."""
    A = np.column_stack([X.loc[idx].to_numpy(), np.broadcast_to(np.asarray(future, dtype=float), (len(idx),))])
    return pd.Series(ridge_predict(models[h], A)[:, 0], index=idx)


def select_setting_v1(well):
    """학습(~2022-12-31)으로 학습하고 검증(2023-01~03, 실제 강수 사용) 평균 MAE가 가장 작은 설정을 고른다."""
    best = None
    for zone, window, lam, use_level in product(RAIN_ZONES, RAIN_WINDOWS, RIDGE_LAMBDAS, USE_LEVEL_OPTIONS):
        X, Y, F = make_features_v1(well, use_level, zone, window)
        models = fit_v1(X, Y, F, lam, TRAIN_END)
        maes = []
        for h in range(1, MAX_H + 1):
            va = rows_h(X, Y, F, h, TRAIN_END + pd.Timedelta(days=1), VALID_END)
            maes.append(float(np.abs(predict_v1(models, X, va, h, F.loc[va, h]) - Y.loc[va, h]).mean()))
        score = float(np.mean(maes))
        if best is None or score < best["검증 MAE"]:
            best = {"zone": zone, "과거 강수 기간": window, "규제 강도": lam, "수위 입력": use_level, "검증 MAE": score}
    return best


@lru_cache(maxsize=None)
def final_model_v1(well):
    """고른 설정으로 2023-03-31까지 전체를 h마다 다시 학습한 모델, 설정, 입력 표를 돌려준다."""
    s = select_setting_v1(well)
    X, Y, F = make_features_v1(well, s["수위 입력"], s["zone"], s["과거 강수 기간"])
    models = fit_v1(X, Y, F, s["규제 강도"], VALID_END)
    return models, s, (X, Y, F)


@lru_cache(maxsize=None)
def scenario_daily_mm(zone):
    """시나리오별 하루 강수량(mm/일)을 구한다: 학습 구간 14일 누적 강수 분포의 분위수 ÷ 14 (무강수는 0)."""
    s14 = rain()[zone].rolling(SCENARIO_BASE_DAYS).sum().loc[:SCENARIO_FIT_END].dropna()
    return {name: 0.0 if q is None else float(s14.quantile(q)) / SCENARIO_BASE_DAYS for name, q in SCENARIOS.items()}


@lru_cache(maxsize=1)
def test_evaluation_v1():
    """시험 구간(2023-04~06)에서 v1(실제 강수 / 시나리오 / 가장 가까운 시나리오), v0, 직전 값 유지, 기울기 연장을 평가한다."""
    records = []
    for well in WELLS:
        models, s, (X, Y, F) = final_model_v1(well)
        daily_mm = scenario_daily_mm(s["zone"])
        m0, s0, _ = final_model(well)
        X0, _ = make_features(well, s0["수위 입력"])
        _, daily, _ = load_series()
        slope = (daily[well] - daily[well].shift(7)) / 7
        start = VALID_END + pd.Timedelta(days=1)
        subsets = [("시험 전체", start)]
        if well == DROP_SUBSET[0]:
            subsets.append((f"{DROP_SUBSET[1]:%Y-%m-%d} 이후", DROP_SUBSET[1]))
        for subset, sub_start in subsets:
            for h in range(1, MAX_H + 1):
                # 모든 방법을 같은 기준일로 비교한다 (v1 입력·실제 미래 강수·목표가 모두 있는 날)
                idx = rows_h(X, Y, F, h, sub_start, TEST_END)
                idx = idx[X0.loc[idx].notna().all(axis=1)]
                y = Y.loc[idx, h]
                preds = {
                    "v1 (실제 강수)": predict_v1(models, X, idx, h, F.loc[idx, h]),
                    "v0 모델": pd.Series(ridge_predict(m0, X0.loc[idx].to_numpy())[:, h - 1], index=idx),
                    "직전 값 유지": pd.Series(0.0, index=idx),
                    "기울기 연장": slope.loc[idx] * h,
                }
                scen = {name: predict_v1(models, X, idx, h, mm * h) for name, mm in daily_mm.items()}
                for name, p in scen.items():
                    preds[f"v1 시나리오 {name}"] = p
                # 가장 가까운 시나리오: 실제 14일 누적 강수(t+1~t+14)와 시나리오 14일 누적이 가장 가까운 것
                idx14 = idx[F.loc[idx, 14].notna() & (idx + pd.Timedelta(days=14) <= TEST_END)]
                names = list(daily_mm)
                gap = np.abs(F.loc[idx14, 14].to_numpy()[:, None] - np.array([daily_mm[n] * 14 for n in names])[None, :])
                pick = [names[i] for i in gap.argmin(axis=1)]
                closest = pd.Series([scen[n].loc[t] for n, t in zip(pick, idx14)], index=idx14)
                for name, p in preds.items():
                    records.append({"감시정": well, "구간": subset, "방법": name, "예측일수": h, "지표": "MAE(m)",
                                    "값": float(np.abs(p - y).mean()), "기준일 수": len(idx)})
                records.append({"감시정": well, "구간": subset, "방법": "v1 (가장 가까운 시나리오)", "예측일수": h,
                                "지표": "MAE(m)", "값": float(np.abs(closest - y.loc[idx14]).mean()), "기준일 수": len(idx14)})
                low = pd.concat(scen, axis=1).min(axis=1)
                high = pd.concat(scen, axis=1).max(axis=1)
                inside = ((y >= low - 1e-9) & (y <= high + 1e-9)).mean() * 100
                records.append({"감시정": well, "구간": subset, "방법": "v1 시나리오 범위", "예측일수": h,
                                "지표": "포함률(%)", "값": float(inside), "기준일 수": len(idx)})
                if h == MAX_H:
                    for n in names:
                        records.append({"감시정": well, "구간": subset, "방법": f"가장 가까운 시나리오={n}", "예측일수": h,
                                        "지표": "선택 횟수", "값": float(pick.count(n)), "기준일 수": len(idx14)})
    return pd.DataFrame(records)


@lru_cache(maxsize=None)
def reliability(well):
    """시험 구간 평가로 예측일수별 신뢰도 표시를 정한다: v1(실제 강수 또는 가장 가까운 시나리오)이 직전 값 유지보다 못하면 '참고 불가'."""
    ev = test_evaluation_v1()
    ev = ev[(ev["감시정"] == well) & (ev["구간"] == "시험 전체") & (ev["지표"] == "MAE(m)")]
    out = {}
    for h in range(1, MAX_H + 1):
        m = ev[ev["예측일수"] == h].set_index("방법")["값"]
        bad = (m["v1 (실제 강수)"] >= m["직전 값 유지"]) or (m["v1 (가장 가까운 시나리오)"] >= m["직전 값 유지"])
        out[h] = "참고 불가 (시험 구간에서 직전 값 유지보다 못함)" if bad else "참고 가능"
    return out


# ── v1 예상 범위와 모델 파일 ─────────────────────────────────────────────────────
# 예상 범위: 세 강수 시나리오의 최소~최대는 시험에서 실제 값을 0~22%밖에 담지 못해 쓰지 않는다.
#   대신 '2022년으로만 학습한 모델'의 검증 구간(2023-01~03) 예측 오차(실제 − 예측) 분포에서
#   예측일수별 10%~90% 구간을 구해, 중앙값 시나리오 예측에 더해 보인다. 시험 결과를 보고 조정하지 않는다.
INTERVAL_Q = (0.10, 0.90)
INTERVAL_SCENARIO = "중앙값"
# 모델 파일: 고른 설정·학습된 가중치·예상 범위·시험 평가 요약을 저장해 두고 도구 호출 때 읽는다.
#   입력 파일(수위·기상)의 내용이나 아래 형식 표시가 바뀌면 다시 학습한다.
MODEL_PATH = ROOT / "data" / "processed" / "forecast_v1_model.json"
MODEL_FORMAT = "v1-2026-11-14c"   # c: 예상 범위의 시험 포함률을 함께 저장
# 강수 영향 안내: 무강수와 상위 10% 시나리오의 14일 뒤 차이가 0.2 m 미만이면 '거의 영향 없음'으로 안내한다
#   (0.2 m는 2수원 감시정 14일 예측 MAE 0.30~0.33 m보다 작은 값. 1수원 감시정3만 0.92 m로 이 기준을 넘는다)
SMALL_RAIN_EFFECT_M = 0.2
WEATHER_PATH = ROOT / "data" / "external" / "weather_daily_2022_2023.csv"


def input_fingerprint():
    """입력 파일(수위·기상) 내용과 모델 형식·설정 후보로 지문(sha256)을 만든다. 이 값이 바뀌면 다시 학습한다."""
    import hashlib
    h = hashlib.sha256()
    for p in (IN_PATH, WEATHER_PATH):
        h.update(p.read_bytes())
    h.update(repr((MODEL_FORMAT, RAIN_ZONES, RAIN_WINDOWS, RIDGE_LAMBDAS, USE_LEVEL_OPTIONS, SCENARIOS,
                   INTERVAL_Q, INTERVAL_SCENARIO, str(TRAIN_END), str(VALID_END), str(TEST_END))).encode())
    return h.hexdigest()


def validation_interval(well, s):
    """고른 설정으로 2022년만 학습한 모델의 검증 구간 오차(실제 − 중앙값 시나리오 예측) 10%·90% 값을 예측일수별로 구한다."""
    X, Y, F = make_features_v1(well, s["수위 입력"], s["zone"], s["과거 강수 기간"])
    models = fit_v1(X, Y, F, s["규제 강도"], TRAIN_END)
    mm = scenario_daily_mm(s["zone"])[INTERVAL_SCENARIO]
    out = {}
    for h in range(1, MAX_H + 1):
        va = rows_h(X, Y, F, h, TRAIN_END + pd.Timedelta(days=1), VALID_END)
        err = Y.loc[va, h] - predict_v1(models, X, va, h, mm * h)
        out[h] = [float(err.quantile(INTERVAL_Q[0])), float(err.quantile(INTERVAL_Q[1]))]
    return out


def rain_effect(models, s, daily_mm):
    """학습된 가중치에서 미래 강수·과거 강수가 측정값에 주는 영향(10 mm당 m)과 시나리오 사이 14일 뒤 차이를 구한다."""
    def per10(model, col):
        return float(model["W"][col, 0] / model["sd"][col] * 10)
    past_col = -2                                   # 입력 순서: [v0 입력..., 과거 누적 강수, 미래 누적 강수]
    m14 = models[MAX_H]
    spread = per10(m14, -1) / 10 * (daily_mm["상위10%"] - daily_mm["무강수"]) * MAX_H
    return {"미래 강수 10mm당 14일 뒤 측정값 변화(m)": round(per10(m14, -1), 4),
            f"과거 {s['과거 강수 기간']}일 강수 10mm당 14일 뒤 측정값 변화(m)": round(per10(m14, past_col), 4),
            "무강수와 상위10% 시나리오의 14일 뒤 차이(m)": round(spread, 3)}


def coverage_for(well, s, models, daily_mm, interval):
    """한 감시정의 예상 범위가 시험 구간(2023-04~06) 실제 값을 담은 비율(%)·기준일 수·범위 폭을 예측일수별로 구한다."""
    X, Y, F = make_features_v1(well, s["수위 입력"], s["zone"], s["과거 강수 기간"])
    mm = daily_mm[INTERVAL_SCENARIO]
    out = {}
    for h in range(1, MAX_H + 1):
        idx = rows_h(X, Y, F, h, VALID_END + pd.Timedelta(days=1), TEST_END)
        pred = predict_v1(models, X, idx, h, mm * h)
        lo, hi = interval[h]
        inside = ((Y.loc[idx, h] >= pred + lo) & (Y.loc[idx, h] <= pred + hi)).mean() * 100
        out[h] = {"포함률(%)": float(inside), "기준일 수": len(idx), "범위 폭(m)": hi - lo}
    return out


def interval_test_coverage(bundle):
    """모델 파일에 저장된 시험 포함률을 감시정·예측일수별 표로 만든다 (보고용)."""
    rows = []
    for well in WELLS:
        for h, c in bundle["wells"][well]["범위 시험 포함률"].items():
            rows.append({"감시정": well, "예측일수": h, **c})
    return pd.DataFrame(rows)


def coverage_sentence(pct):
    """예상 범위의 시험 포함률을 쉬운 문장으로 바꾼다 (목표 80%와 비교)."""
    target = round((INTERVAL_Q[1] - INTERVAL_Q[0]) * 100)
    if pct < target:
        return (f"이 범위는 {target}%를 담도록 만들었지만, 시험 구간에서는 실제 값의 {pct:.0f}%만 담았습니다"
                f"(범위가 좁게 잡혔습니다).")
    return f"이 범위는 {target}%를 담도록 만들었고, 시험 구간에서는 실제 값의 {pct:.0f}%를 담았습니다."


def models_to_json(models):
    """h별 릿지 모델(numpy 배열)을 JSON에 넣을 수 있는 리스트로 바꾼다."""
    return {str(h): {k: np.asarray(v).tolist() for k, v in m.items()} for h, m in models.items()}


def models_from_json(obj):
    """JSON에서 읽은 h별 릿지 모델을 다시 numpy 배열로 바꾼다."""
    return {int(h): {k: np.asarray(v, dtype=float) for k, v in m.items()} for h, m in obj.items()}


def build_bundle():
    """감시정마다 설정 고르기·학습·예상 범위·시험 평가 요약을 계산해 모델 파일 내용(dict)을 만든다 (오래 걸림)."""
    ev = test_evaluation_v1()
    wells = {}
    for well in WELLS:
        models, s, _ = final_model_v1(well)
        mm = scenario_daily_mm(s["zone"])
        sub = ev[(ev["감시정"] == well) & (ev["구간"] == "시험 전체")]
        compare = {}
        for h in REPORT_H:
            m = sub[sub["예측일수"] == h].set_index("방법")["값"]
            compare[f"{h}일 뒤"] = {
                "v1 실제 강수 사용 MAE(m)": round(float(m["v1 (실제 강수)"]), 3),
                "v1 가장 가까운 시나리오 MAE(m)": round(float(m["v1 (가장 가까운 시나리오)"]), 3),
                "직전 값 유지 MAE(m)": round(float(m["직전 값 유지"]), 3),
            }
        interval = validation_interval(well, s)
        cover = coverage_for(well, s, models, mm, interval)
        wells[well] = {"설정": s, "모델": models_to_json(models), "시나리오 mm/일": mm,
                       "예상 범위 오차": {str(h): v for h, v in interval.items()},
                       "범위 시험 포함률": {str(h): v for h, v in cover.items()},
                       "신뢰도": {str(h): v for h, v in reliability(well).items()},
                       "시험 오차 비교": compare, "강수 영향": rain_effect(models, s, mm)}
    return {"지문": input_fingerprint(), "형식": MODEL_FORMAT, "wells": wells}


@lru_cache(maxsize=1)
def load_bundle():
    """모델 파일을 읽는다. 파일이 없거나 입력 지문이 다르면 다시 학습해 파일로 저장한다."""
    fp = input_fingerprint()
    if MODEL_PATH.exists():
        bundle = json.loads(MODEL_PATH.read_text(encoding="utf-8"))
        if bundle.get("지문") == fp:
            return decode_bundle(bundle)
    bundle = build_bundle()
    MODEL_PATH.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return decode_bundle(json.loads(MODEL_PATH.read_text(encoding="utf-8")))


def decode_bundle(bundle):
    """파일에서 읽은 내용의 모델·예상 범위·신뢰도를 쓰기 좋은 형태(numpy, 정수 키)로 바꾼다."""
    for b in bundle["wells"].values():
        b["모델"] = models_from_json(b["모델"])
        b["예상 범위 오차"] = {int(h): v for h, v in b["예상 범위 오차"].items()}
        b["범위 시험 포함률"] = {int(h): v for h, v in b["범위 시험 포함률"].items()}
        b["신뢰도"] = {int(h): v for h, v in b["신뢰도"].items()}
    return bundle


def forecast(well, as_of=None, horizon=14, rain_scenario=None):
    """감시정 하나의 기준일 이후 1~horizon일 측정값을 강수 시나리오별로 예측해 dict로 돌려준다 (Agent 호출용, v1)."""
    if well not in WELLS:
        return {"오류": f"알 수 없는 감시정입니다: {well}. 가능한 값: {', '.join(WELLS)}"}
    if not (isinstance(horizon, int) and 1 <= horizon <= MAX_H):
        return {"오류": f"예측 일수는 1~{MAX_H}일 사이 정수여야 합니다."}
    if rain_scenario is not None and rain_scenario not in SCENARIOS:
        return {"오류": f"rain_scenario는 {', '.join(SCENARIOS)} 중 하나이거나 비워 두어야 합니다 (비우면 세 가지 모두 계산)."}
    hourly, daily, _ = load_series()
    t = parse_as_of(as_of, hourly)
    if t is None:
        return {"오류": f"기준 시각 {as_of}은(는) 데이터 범위 밖입니다. {DATA_NOTE}"}
    # 기준일: 그날 23시 자료까지 있어야 하루 중앙값을 쓸 수 있으므로, 23시 전이면 전날을 기준일로 쓴다
    base = t.normalize() if t.hour == 23 else t.normalize() - pd.Timedelta(days=1)

    b = load_bundle()["wells"][well]
    s, models, daily_mm = b["설정"], b["모델"], b["시나리오 mm/일"]
    X, _, _ = make_features_v1(well, s["수위 입력"], s["zone"], s["과거 강수 기간"])
    if base not in X.index or X.loc[base].isna().any():
        return {"오류": f"{base:%Y-%m-%d}은(는) 입력을 계산할 수 없습니다. 데이터 시작 직후(2022-01-15 이전)이거나, "
                        f"최근 {s['과거 강수 기간']}일 안에 받지 못한 기상 날짜가 있습니다.", "안내": DATA_NOTE}
    names = [rain_scenario] if rain_scenario else list(SCENARIOS)
    value0 = float(daily[well].loc[base])
    label = config.value_label()
    days = []
    for h in range(1, horizon + 1):
        d = base + pd.Timedelta(days=h)
        delta = {n: float(predict_v1(models, X, pd.DatetimeIndex([base]), h, daily_mm[n] * h).iloc[0]) for n in SCENARIOS}
        lo, hi = b["예상 범위 오차"][h]
        mid = value0 + delta[INTERVAL_SCENARIO]
        actual = daily[well].get(d)
        days.append({
            "날짜": f"{d:%Y-%m-%d}",
            f"예측 {label}": {n: round(value0 + delta[n], 2) for n in names},
            "예상 범위(m, 과거 예측 오차 10~90%)": f"{mid + lo:.2f} ~ {mid + hi:.2f}",
            "예상 범위의 시험 포함률(%)": round(b["범위 시험 포함률"][h]["포함률(%)"], 1),
            f"실제 {label}": None if actual is None or pd.isna(actual) else round(float(actual), 2),
            "신뢰도": b["신뢰도"][h] + (" / 큰 변화 구간에서 오차가 큼" if well == BIG_DROP_WELL else ""),
        })

    last = days[-1]
    mid_last = value0 + float(predict_v1(models, X, pd.DatetimeIndex([base]), horizon,
                                         daily_mm[INTERVAL_SCENARIO] * horizon).iloc[0])
    effect = b["강수 영향"]
    per10 = abs(effect["미래 강수 10mm당 14일 뒤 측정값 변화(m)"])
    spread = abs(effect["무강수와 상위10% 시나리오의 14일 뒤 차이(m)"])
    numbers = f"(미래 강수 10 mm당 14일 뒤 {per10:.3f} m, 무강수와 상위 10% 시나리오의 차이 {spread:.2f} m)"
    if spread < SMALL_RAIN_EFFECT_M:
        rain_note = (f"앞으로 2주의 강수량은 이 감시정 값에 거의 영향을 주지 않는 것으로 나타났습니다{numbers}. "
                     f"최근 {s['과거 강수 기간']}일간 이미 내린 비의 영향이 큽니다.")
    else:
        rain_note = (f"이 감시정은 앞으로 2주의 강수량에 따라 14일 뒤 값이 달라질 수 있습니다{numbers}. "
                     f"다만 시험 구간 14일 예측 오차가 이 차이보다 훨씬 커서, 시나리오 사이의 차이를 믿기 어렵습니다. "
                     f"최근 {s['과거 강수 기간']}일간 이미 내린 비의 영향도 큽니다.")
    explain = (f"{well}의 {base:%Y-%m-%d} 측정값은 {value0:.2f} m이고, {horizon}일 뒤({last['날짜']})에는 "
               f"{mid_last:.2f} m(예상 범위 {last['예상 범위(m, 과거 예측 오차 10~90%)']})로, "
               f"지금보다 {config.water_expect(mid_last - value0)} (관측 기간 중앙값 수준의 비를 가정). "
               + coverage_sentence(b["범위 시험 포함률"][horizon]["포함률(%)"]))
    notes = [RAIN_NOTE_V1, DATA_NOTE]
    if base <= VALID_END:
        notes.append("기준일이 모델 학습 기간(2023-03-31까지) 안이라, 이 예측의 오차는 실제 사용 때보다 작게 보일 수 있습니다.")
    notes.append("측정값은 " + ("물까지의 깊이로 가정했습니다(값이 클수록 지하수가 낮음). " if config.VALUE_IS_DEPTH
                               else "해발 수위로 가정했습니다(값이 클수록 지하수가 높음). ") + "명세로 확인하지 못한 가정입니다.")
    return {
        "감시정": well, "기준일": f"{base:%Y-%m-%d}", f"기준 {label} (일 중앙값)": round(value0, 2),
        "설명": explain,
        "강수 영향 안내": rain_note,
        "강수 시나리오(14일 누적 mm, zone=" + s["zone"] + ")": {n: round(daily_mm[n] * 14, 1) for n in names},
        "일별 예측": days,
        "시험 구간(2023-04~06) 오차 비교": b["시험 오차 비교"],
        "강수 영향 수치": effect,
        "모델 설정": {k: s[k] for k in ("zone", "과거 강수 기간", "규제 강도", "수위 입력")},
        "안내": " ".join(notes),
    }


def plot_v1(ev):
    """감시정별로 예측일수에 따른 MAE를 v1(실제 강수 / 가장 가까운 시나리오), v0, 비교 기준 선으로 그려 저장한다."""
    plt.rcParams["font.family"] = "Malgun Gothic"     # 한글 깨짐 방지
    plt.rcParams["axes.unicode_minus"] = False
    # 방법별 색: v0 그림과 같은 방법은 같은 색(v0 파랑, 유지 주황, 기울기 청록), v1은 팔레트 7번(보라)·5번(분홍)
    colors = {"v1 (실제 강수)": "#4a3aa7", "v1 (가장 가까운 시나리오)": "#e87ba4",
              "v0 모델": "#2a78d6", "직전 값 유지": "#eb6834", "기울기 연장": "#1baf7a"}
    labels = ["v1: 실제 강수 사용(비 영향 확인용)", "v1: 실제와 가장 가까운 시나리오",
              "v0: 비 정보 없음", "직전 값 유지", "최근 7일 기울기 연장"]
    mae = ev[ev["지표"] == "MAE(m)"]
    panels = [(w, "시험 전체") for w in WELLS] + [(DROP_SUBSET[0], f"{DROP_SUBSET[1]:%Y-%m-%d} 이후")]
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, (well, subset) in zip(axes.flat, panels):
        sub = mae[(mae["감시정"] == well) & (mae["구간"] == subset)]
        for name, c in colors.items():
            s = sub[sub["방법"] == name].sort_values("예측일수")
            ax.plot(s["예측일수"], s["값"], color=c, lw=2, marker="o", ms=3)
        ax.set_title(f"{well}" + ("" if subset == "시험 전체" else f"\n({subset} 기준일만)"),
                     loc="left", fontsize=10, color="#0b0b0b")
        ax.set_xticks([1, 3, 7, 14])
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color="#e1e0d9", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(colors="#52514e", labelsize=8)
        ax.set_xlabel("예측 일수(일)", fontsize=8, color="#52514e")
        ax.set_ylabel("MAE(m)", fontsize=8, color="#52514e")
    axes.flat[-1].axis("off")
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=4) for c in colors.values()]
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False, fontsize=9)
    fig.suptitle("2주 예측 v1(강수 포함): 시험 구간(2023-04~06) 예측 일수별 평균절대오차 (낮을수록 좋음)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT_FIG_V1, dpi=130, facecolor="#fcfcfb")
    plt.close(fig)


def main():
    """v1 확인용: 감시정별 고른 설정, 시험 구간 MAE·포함률 표를 출력하고 v1 CSV와 그림을 저장한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print("v1 감시정별 설정 (검증 2023-01~03, 실제 강수 사용으로 선택):")
    for well in WELLS:
        _, s, _ = final_model_v1(well)
        mm = scenario_daily_mm(s["zone"])
        print(f"  {well}: zone {s['zone']}, 과거 {s['과거 강수 기간']}일 누적, 규제 강도 {s['규제 강도']}, "
              f"수위 입력 {s['수위 입력']}, 검증 평균 MAE {s['검증 MAE']:.3f} m (v0 {final_model(well)[1]['검증 MAE']:.3f} m) | "
              f"시나리오 14일 누적 mm: " + ", ".join(f"{k} {v * 14:.1f}" for k, v in mm.items()))

    ev = test_evaluation_v1()
    ev.round(4).to_csv(OUT_CSV_V1, index=False, encoding="utf-8-sig", lineterminator="\n")
    plot_v1(ev)

    mae = ev[(ev["지표"] == "MAE(m)") & ev["예측일수"].isin(REPORT_H)]
    cols = ["v1 (실제 강수)", "v1 (가장 가까운 시나리오)", "v1 시나리오 무강수", "v1 시나리오 중앙값",
            "v1 시나리오 상위10%", "v0 모델", "직전 값 유지", "기울기 연장"]
    table = mae.pivot_table(index=["감시정", "구간", "예측일수"], columns="방법", values="값")[cols].round(3)
    print("\n시험 구간 MAE(m):"); print(table.to_string())
    cov = ev[(ev["지표"] == "포함률(%)") & ev["예측일수"].isin(REPORT_H)]
    print("\n실제 값이 세 시나리오 예측 범위(최소~최대) 안에 든 비율(%):")
    print(cov.pivot_table(index=["감시정", "구간"], columns="예측일수", values="값").round(1).to_string())
    pick = ev[ev["지표"] == "선택 횟수"]
    print("\n가장 가까운 시나리오로 뽑힌 횟수(14일):")
    print(pick.pivot_table(index=["감시정", "구간"], columns="방법", values="값").astype(int).to_string())
    n = ev[(ev["방법"] == "v1 (실제 강수)") & (ev["구간"] == "시험 전체")].groupby("감시정")["기준일 수"].agg(["max", "min"])
    print("\n시험 기준일 수(h=1 / h=14, 기상 빈 날짜로 빠진 날 제외):"); print(n.to_string())
    print("저장:", OUT_CSV_V1, "|", OUT_FIG_V1)

    bundle = load_bundle()
    cov_iv = interval_test_coverage(bundle)
    cov_iv.round(4).to_csv(OUT_INTERVAL_V1, index=False, encoding="utf-8-sig", lineterminator="\n")
    print("\n예상 범위(검증 오차 10~90%)가 시험 구간 실제 값을 담은 비율(%) — 보고만, 범위는 조정하지 않음:")
    print(cov_iv[cov_iv["예측일수"].isin(REPORT_H)].pivot_table(index="감시정", columns="예측일수", values="포함률(%)")
          .round(1).to_string())
    print("범위 폭(m):")
    print(cov_iv[cov_iv["예측일수"].isin(REPORT_H)].pivot_table(index="감시정", columns="예측일수", values="범위 폭(m)")
          .round(3).to_string())
    print("\n강수 영향(14일 뒤):")
    for well in WELLS:
        print(" ", well, bundle["wells"][well]["강수 영향"])
    print("저장:", OUT_INTERVAL_V1, "|", MODEL_PATH)

    r = forecast("2수원감시정1")
    print("\n도구 예시:", {k: r[k] for k in list(r)[:3]})
    print("  설명:", r["설명"]); print("  강수:", r["강수 영향 안내"])
    print("  ", r["일별 예측"][0]); print("  ", r["일별 예측"][-1])
    r3 = forecast("1수원감시정3", rain_scenario="무강수", horizon=3)
    print("1수원3 무강수 3일:", [(d["날짜"], d[f"예측 {config.value_label()}"], d["신뢰도"]) for d in r3["일별 예측"]])
    print("잘못된 시나리오 예:", forecast("2수원감시정1", rain_scenario="평년"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
