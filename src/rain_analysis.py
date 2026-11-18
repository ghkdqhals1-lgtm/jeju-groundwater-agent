"""비와 수위의 관계 확인: zone별 강수량과 감시정별 '7일 뒤 수위 변화량'의 시차 상관을 계산한다.

- 학습 구간(2022-01-01 ~ 2022-12-31)만 쓴다. 7일 뒤 수위도 이 구간 안에 있어야 한다.
- 받지 못한 기상 날짜는 채우지 않는다. 상관은 값이 있는 날끼리만 계산한다.
- 입력: data/processed/groundwater_hourly.csv, data/external/weather_daily_2022_2023.csv
- 출력: data/processed/rain_lag_corr.csv (시차 상관 표), 화면 요약

실행: python src/rain_analysis.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LEVEL_PATH = ROOT / "data" / "processed" / "groundwater_hourly.csv"
WEATHER_PATH = ROOT / "data" / "external" / "weather_daily_2022_2023.csv"
OUT_CSV = ROOT / "data" / "processed" / "rain_lag_corr.csv"

START, END = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")   # 학습 구간만
TARGET_DAYS = 7                       # 목표: 기준일 t 대비 7일 뒤 수위 변화량
MAX_LAG = 30                          # 시차 0~30일
CUM_WINDOWS = [3, 7, 14, 30]          # 누적 강수 후보 기간(일)
ZONES = ["해안", "중산간", "산지", "평균"]


def zone_rain():
    """관측소별 일 강수량을 zone별 일평균 강수량으로 바꾸고, zone 3개의 평균도 더한다 (빈 날짜는 빈 값 그대로)."""
    w = pd.read_csv(WEATHER_PATH, encoding="utf-8-sig", parse_dates=["baseDate"])
    z = w.pivot_table(index="baseDate", columns="zone", values="dailyRainfall", aggfunc="mean")
    z = z.reindex(pd.date_range("2022-01-01", "2023-06-30", freq="D"))   # 받지 못한 날짜는 NaN으로 남긴다
    z["평균"] = z[["해안", "중산간", "산지"]].mean(axis=1, skipna=False)
    return z[ZONES]


def level_target():
    """감시정별 하루 중앙값 수위로 '기준일 대비 7일 뒤 변화량'을 만든다 (학습 구간 안에서 끝나는 것만)."""
    df = pd.read_csv(LEVEL_PATH, index_col="datetime", parse_dates=True, encoding="utf-8-sig")
    daily = df.filter(like="level_").rename(columns=lambda c: c.replace("level_", "")).resample("D").median()
    y = daily.shift(-TARGET_DAYS) - daily
    return y.loc[START:END - pd.Timedelta(days=TARGET_DAYS)]


def lag_corr(rain, y):
    """시차 L(0~30일)마다 'L일 전의 일 강수량'과 '7일 뒤 수위 변화량'의 상관계수를 구한다."""
    rows = []
    for well in y:
        for zone in rain:
            for lag in range(MAX_LAG + 1):
                r = rain[zone].shift(lag).reindex(y.index)
                ok = r.notna() & y[well].notna()
                rows.append({"감시정": well, "zone": zone, "시차(일)": lag,
                             "상관": float(np.corrcoef(r[ok], y[well][ok])[0, 1]), "표본": int(ok.sum())})
    return pd.DataFrame(rows)


def cum_corr(rain, y):
    """누적 강수 후보와 7일 뒤 수위 변화량의 상관을 구한다: 기준일까지의 과거 누적, 기준일 다음 날부터 7일 누적(참고)."""
    rows = []
    for well in y:
        for zone in rain:
            for w in CUM_WINDOWS:
                past = rain[zone].rolling(w).sum().reindex(y.index)       # t-w+1 ~ t (기준일까지 알 수 있음)
                ok = past.notna() & y[well].notna()
                rows.append({"감시정": well, "zone": zone, "입력": f"과거 {w}일 누적",
                             "상관": float(np.corrcoef(past[ok], y[well][ok])[0, 1]), "표본": int(ok.sum())})
            fut = rain[zone].rolling(TARGET_DAYS).sum().shift(-TARGET_DAYS).reindex(y.index)   # t+1 ~ t+7
            ok = fut.notna() & y[well].notna()
            rows.append({"감시정": well, "zone": zone, "입력": "미래 7일 누적(t+1~t+7)",
                         "상관": float(np.corrcoef(fut[ok], y[well][ok])[0, 1]), "표본": int(ok.sum())})
    return pd.DataFrame(rows)


def main():
    """시차 상관과 누적 강수 상관을 계산해 저장하고, 감시정·zone별 최대 상관 시차를 출력한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    rain, y = zone_rain(), level_target()
    print(f"분석 기간(기준일): {y.index.min():%Y-%m-%d} ~ {y.index.max():%Y-%m-%d}, "
          f"강수 빈 날짜(학습 구간): {int(rain.loc[START:END, '해안'].isna().sum())}일")

    lc = lag_corr(rain, y)
    lc.round(4).to_csv(OUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
    peak = lc.loc[lc.groupby(["감시정", "zone"])["상관"].idxmax()]
    table = peak.pivot(index="감시정", columns="zone", values=["상관", "시차(일)"])
    print("\n[시차 상관] 감시정 × zone별 최대 상관과 그때의 시차(일):")
    print(table.round(2).to_string())
    print("\n[시차 상관] '평균' zone, 시차 0·3·7·14·21·30일:")
    sel = lc[(lc["zone"] == "평균") & lc["시차(일)"].isin([0, 3, 7, 14, 21, 30])]
    print(sel.pivot(index="감시정", columns="시차(일)", values="상관").round(2).to_string())

    cc = cum_corr(rain, y)
    print("\n[누적 강수] 7일 뒤 수위 변화량과의 상관 (zone별):")
    print(cc.pivot_table(index=["감시정", "입력"], columns="zone", values="상관").round(2)
          .reindex(columns=ZONES).to_string())
    print("저장:", OUT_CSV)
    return 0


if __name__ == "__main__":
    sys.exit(main())
