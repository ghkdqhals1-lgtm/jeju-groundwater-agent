"""제주데이터허브 757번(기상관측소 정보) API를 한 번 호출해서 관측소 목록을 data/external/weather_stations.csv로 저장한다."""
import os
import sys
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

BASE_URL = "https://open.jejudatahub.net/api/proxy/t1bab75a555a15715btaa7a5abaa1aab"
EXPECTED_FIELDS = [
    "observatoryCode", "startDate", "endDate", "observatoryName", "managementOffice",
    "longitude", "latitude", "legalDongriCode", "seaLevelAltitude",
]
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "external" / "weather_stations.csv"
# 데이터 기준 기간 (지하수 감시정 데이터)
PERIOD_START, PERIOD_END = pd.Timestamp("2022-01-01"), pd.Timestamp("2023-06-30")

load_dotenv()
KEY = os.getenv("JEJU_DATAHUB_KEY")


def mask(text):
    return str(text).replace(KEY, "***KEY***") if KEY else str(text)


def split_lat_lon(a, b):
    """두 좌표 값 중 제주 범위(위도 33대, 경도 126대)로 위도·경도를 판별한다."""
    a, b = float(a), float(b)
    if 32 <= a <= 35 and 125 <= b <= 128:
        return a, b
    if 32 <= b <= 35 and 125 <= a <= 128:
        return b, a
    return None, None


def to_date(s):
    return pd.to_datetime(s, errors="coerce") if s and str(s).strip() else pd.NaT


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    if not KEY:
        print("JEJU_DATAHUB_KEY가 .env에 없습니다.")
        return 1

    try:
        resp = requests.get(f"{BASE_URL}/{KEY}", params={"limit": 100, "number": 1}, timeout=30)
        resp.raise_for_status()
        body = resp.json()
    except requests.HTTPError as e:
        print("HTTP 오류:", mask(e))
        print("응답 본문:", mask(e.response.text[:500]))
        return 1
    except Exception as e:
        print("호출 실패:", type(e).__name__, mask(e))
        return 1

    rows = body["data"]
    print("totCnt:", body.get("totCnt"), "| hasMore:", body.get("hasMore"), "| 받은 행 수:", len(rows))
    fields = sorted({k for r in rows for k in r})
    print("가이드와 일치:", fields == sorted(EXPECTED_FIELDS),
          "| 가이드에 없는 항목:", sorted(set(fields) - set(EXPECTED_FIELDS)),
          "| 응답에 없는 항목:", sorted(set(EXPECTED_FIELDS) - set(fields)))
    print("첫 행 원본:", rows[0])

    df = pd.DataFrame(rows)
    lat_field = df["latitude"].astype(float)
    print(f"latitude 필드 값 범위: {lat_field.min()} ~ {lat_field.max()}")
    lon_field = df["longitude"].astype(float)
    print(f"longitude 필드 값 범위: {lon_field.min()} ~ {lon_field.max()}")

    coords = [split_lat_lon(a, b) for a, b in zip(df["latitude"], df["longitude"])]
    df["lat"] = [c[0] for c in coords]
    df["lon"] = [c[1] for c in coords]
    df["coord_swapped"] = [c[0] is not None and c[0] != float(a) for c, a in zip(coords, df["latitude"])]
    print("필드 이름과 실제 값이 뒤바뀐 행:", int(df["coord_swapped"].sum()), "/", len(df),
          "| 판별 실패:", int(df["lat"].isna().sum()))

    df["altitude_m"] = pd.to_numeric(df["seaLevelAltitude"], errors="coerce")
    df["start"] = df["startDate"].map(to_date)
    df["end"] = df["endDate"].map(to_date)
    df["active_in_period"] = (df["start"] <= PERIOD_START) & (df["end"].isna() | (df["end"] >= PERIOD_END))
    df = df.sort_values("altitude_m").reset_index(drop=True)

    show = df[["observatoryName", "observatoryCode", "lat", "lon", "altitude_m", "startDate", "endDate", "active_in_period"]]
    print(show.to_string())
    print("기간 내 계속 운영:", int(df["active_in_period"].sum()), "/", len(df))

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["start", "end"]).to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print("저장:", OUT_PATH)
    return 0


if __name__ == "__main__":
    sys.exit(main())
