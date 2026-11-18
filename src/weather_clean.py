"""756번 일별 기상 캐시(JSON)를 하나의 표로 합치고, 757번 관측소 목록에 고도 지역(zone)을 붙인다.

- 입력: data/external/cache/YYYYMMDD.json (src/collect_weather.py 결과)
        data/external/weather_stations.csv (src/test_station_api.py 결과)
- 출력: data/external/weather_daily_2022_2023.csv (관측소 × 날짜, 숫자는 숫자형, baseDate는 날짜형)
        받지 못한 날짜는 채우지 않으므로 그 날짜의 행이 없다.
        data/external/weather_stations.csv (zone 컬럼 추가해서 다시 저장)
- 확인 내용(관측소 수 변화, 강수량 특이값, 주요 관측소 유무, zone별 강수)을 화면에 출력한다.

실행: python src/weather_clean.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data" / "external"
CACHE_DIR = EXT / "cache"
OUT_DAILY = EXT / "weather_daily_2022_2023.csv"
STATIONS = EXT / "weather_stations.csv"
START_DATE, END_DATE = "2022-01-01", "2023-06-30"

# 숫자로 바꿀 항목 (원본은 소수점 4자리 문자열)
NUMERIC_FIELDS = ["averageTemperature", "lowestTemperature", "highestTemperature", "dailyRainfall",
                  "maximumWindSpeed", "averageWindSpeed", "maximumWindSpeedDirection"]
# 시각 항목 (원본은 '459'처럼 앞의 0이 빠진 HHMM 문자열) → 'HH:MM' 문자열로 바꾼다
TIME_FIELDS = ["lowestTemperatureTime", "highestTemperatureTime", "maximumWindSpeedTime"]
# 고도 지역 경계(m): 해안 200 미만, 중산간 200~700, 산지 700 이상
ZONE_BINS = [-float("inf"), 200, 700, float("inf")]
ZONE_LABELS = ["해안", "중산간", "산지"]
# 이름이 같은 관측소는 이 코드로 연결한다 ('강정'은 685가 2018년 종료, 980이 운영 중)
NAME_TO_CODE_OVERRIDE = {"강정": "980"}
# 기상청 주요 관측소 (756번 1일치 응답에 없었던 곳)
MAJOR_STATIONS = ["제주", "서귀포", "성산", "고산"]


def load_cache():
    """캐시 폴더의 날짜별 JSON을 읽어 원본 문자열 그대로의 표로 합치고, 캐시에 없는 날짜 목록을 함께 돌려준다."""
    dates = [d.strftime("%Y%m%d") for d in pd.date_range(START_DATE, END_DATE, freq="D")]
    # 받지 못한 날짜는 채우지 않고 건너뛴다 (합친 표에 그 날짜의 행이 없다)
    missing = [d for d in dates if not (CACHE_DIR / f"{d}.json").exists()]
    rows = []
    for d in dates:
        if d in missing:
            continue
        day = json.loads((CACHE_DIR / f"{d}.json").read_text(encoding="utf-8"))
        for r in day["data"]:
            rows.append({**r, "searchDate": d})
    return pd.DataFrame(rows, dtype=str), missing


def hhmm(s):
    """'459' 같은 HHMM 문자열을 '04:59'로 바꾼다. 비어 있거나 형식이 다르면 빈 값으로 둔다."""
    s = str(s).strip()
    if not s.isdigit() or len(s) > 4:
        return None
    s = s.zfill(4)
    return f"{s[:2]}:{s[2:]}"


def clean(raw):
    """숫자 항목은 숫자형으로, baseDate는 날짜형으로, 시각 항목은 'HH:MM'으로 바꾼다."""
    df = raw.copy()
    for c in NUMERIC_FIELDS:
        df[c] = pd.to_numeric(df[c].str.strip().replace("", None), errors="coerce")
    for c in TIME_FIELDS:
        df[c] = df[c].map(hhmm)
    df["baseDate"] = pd.to_datetime(df["baseDate"], format="%Y%m%d")
    return df


def add_zone(stations):
    """관측소 표에 고도 기준 지역(zone) 컬럼을 붙인다."""
    st = stations.copy()
    st["zone"] = pd.cut(st["altitude_m"], bins=ZONE_BINS, labels=ZONE_LABELS, right=False).astype(str)
    return st


def name_to_station(stations):
    """관측소 이름 → 관측소 한 행을 찾는 표를 만든다 (같은 이름은 지정 코드로, 없으면 운영 중인 곳으로)."""
    st = stations.copy()
    st["observatoryCode"] = st["observatoryCode"].astype(str)
    picked = []
    for name, g in st.groupby("observatoryName"):
        if name in NAME_TO_CODE_OVERRIDE:
            g = g[g["observatoryCode"] == NAME_TO_CODE_OVERRIDE[name]]
        elif len(g) > 1:
            g = g[g["endDate"].isna()]
        picked.append(g.iloc[0])
    return pd.DataFrame(picked).set_index("observatoryName")


def report_raw_rainfall(raw):
    """강수량 원본 문자열에서 빈 값, 음수, -9 같은 특이 코드를 센다."""
    s = raw["dailyRainfall"].str.strip()
    num = pd.to_numeric(s.replace("", None), errors="coerce")
    print("[2] dailyRainfall 원본 점검")
    print("  전체:", len(s), "| 빈 값:", int((s == "").sum()) + int(s.isna().sum()),
          "| 숫자로 못 바꾼 값:", int((num.isna() & (s != "") & s.notna()).sum()),
          "| 음수:", int((num < 0).sum()), "| -9 계열(-9, -99, -999):", int(num.isin([-9, -99, -999]).sum()))
    weird = s[num.isna() | (num < 0)].value_counts().head(10)
    if len(weird):
        print("  특이 값 예:", weird.to_dict())


def main():
    """캐시를 합쳐 저장하고, 관측소 zone을 붙인 뒤 요청된 네 가지를 확인해 출력한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    raw, missing = load_cache()
    print(f"빈 날짜(캐시에 없음): {len(missing)}일", missing if missing else "")
    df = clean(raw)

    stations = pd.read_csv(STATIONS, encoding="utf-8-sig", dtype={"observatoryCode": str, "startDate": str, "endDate": str})
    stations = add_zone(stations)
    stations.to_csv(STATIONS, index=False, encoding="utf-8-sig", lineterminator="\n")

    lookup = name_to_station(stations)
    unmatched = sorted(set(df["observatoryName"]) - set(lookup.index))
    df["observatoryCode"] = df["observatoryName"].map(lookup["observatoryCode"])
    df["zone"] = df["observatoryName"].map(lookup["zone"])
    df["altitude_m"] = df["observatoryName"].map(lookup["altitude_m"])

    cols = ["baseDate", "observatoryName", "observatoryCode", "zone", "altitude_m"] + \
           [c for c in raw.columns if c not in ("baseDate", "observatoryName", "searchDate")]
    df = df[cols].sort_values(["baseDate", "observatoryName"]).reset_index(drop=True)
    df.to_csv(OUT_DAILY, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d", lineterminator="\n")
    print("저장:", OUT_DAILY, df.shape, "|", STATIONS, "(zone 추가)")
    print("baseDate와 요청 날짜가 다른 행:", int((raw["baseDate"] != raw["searchDate"]).sum()))
    print("관측소 목록과 이름이 안 맞는 관측소:", unmatched or "없음")

    # [1] 날짜별 관측소 수
    per_day = df.groupby("baseDate")["observatoryName"].nunique()
    print("[1] 날짜별 관측소 수: 최소", per_day.min(), "최대", per_day.max(),
          "| 분포:", per_day.value_counts().sort_index().to_dict())
    changes = per_day[per_day.diff().fillna(0) != 0]   # 빈 날짜는 건너뛰고 앞뒤 받은 날짜끼리 비교
    print("  관측소 수가 바뀐 날:", {str(k.date()): int(v) for k, v in changes.items()} or "없음")
    # 받은 날짜 수보다 적게 나온 관측소 (빈 날짜는 모든 관측소에서 빠지므로 비교 대상이 아니다)
    partial = df.groupby("observatoryName")["baseDate"].agg(["min", "max", "count"])
    partial = partial[partial["count"] < len(per_day)]
    print(f"  받은 {len(per_day)}일 중 일부 날짜에만 나온 관측소:", partial.astype(str).to_dict("index") or "없음")

    # [2] 강수량 특이값과 최댓값
    report_raw_rainfall(raw)
    r = df["dailyRainfall"]
    top = df.loc[r.idxmax()]
    print(f"  결측: {int(r.isna().sum())} | 최댓값 {r.max():.1f} mm ({top['observatoryName']}, {top['baseDate']:%Y-%m-%d})")
    print("  상위 5개:", df.nlargest(5, "dailyRainfall")[["baseDate", "observatoryName", "dailyRainfall"]]
          .assign(baseDate=lambda t: t["baseDate"].dt.strftime("%Y-%m-%d")).values.tolist())

    # [3] 주요 관측소 4곳이 756번에 있는지
    present = {n: int((df["observatoryName"] == n).sum()) for n in MAJOR_STATIONS}
    print("[3] 주요 관측소 행 수 (전 기간):", present)

    # [4] zone별 관측소 수와 zone별 일평균 강수량의 월별 합계
    zone_n = df.groupby("zone")["observatoryName"].nunique()
    print("[4] zone별 관측소 수(756번에 나온 관측소):", zone_n.to_dict())
    daily_zone = df.pivot_table(index="baseDate", columns="zone", values="dailyRainfall", aggfunc="mean")
    # 빈 날짜는 0으로 채우지 않으므로, 빈 날짜가 있는 달은 합계가 실제보다 작을 수 있다
    monthly = daily_zone.resample("MS").sum().round(1)
    monthly["받은 일수"] = daily_zone.resample("MS").size()
    monthly.index = monthly.index.strftime("%Y-%m")
    print("  zone별 일평균 강수량의 월별 합계(mm):")
    print(monthly[[z for z in ZONE_LABELS if z in monthly] + ["받은 일수"]].to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
