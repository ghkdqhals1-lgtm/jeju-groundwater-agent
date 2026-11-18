"""제주데이터허브 756번(일별 기상 정보) API 호출 테스트. 하루치만 요청해서 응답 구조를 확인한다."""
import os
import sys

import requests
from dotenv import load_dotenv

BASE_URL = "https://open.jejudatahub.net/api/proxy/1aD5taat1attaa51Db1511b51ab9Da19"
EXPECTED_FIELDS = [
    "observatoryName", "baseDate", "averageTemperature", "lowestTemperature",
    "lowestTemperatureTime", "highestTemperature", "highestTemperatureTime", "dailyRainfall",
    "maximumWindSpeed", "maximumWindSpeedTime", "averageWindSpeed", "maximumWindSpeedDirection",
]

load_dotenv()
KEY = os.getenv("JEJU_DATAHUB_KEY")


def mask(text):
    """출력할 문자열에서 API 키를 가린다."""
    return str(text).replace(KEY, "***KEY***") if KEY else str(text)


def fetch_page(search_date, number=1, limit=100):
    resp = requests.get(
        f"{BASE_URL}/{KEY}",
        params={"searchDate": search_date, "limit": limit, "number": number},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    if not KEY:
        print("JEJU_DATAHUB_KEY가 .env에 없습니다.")
        return 1

    search_date = sys.argv[1] if len(sys.argv) > 1 else "20220101"
    try:
        body = fetch_page(search_date)
    except requests.HTTPError as e:
        print("HTTP 오류:", mask(e))
        print("응답 본문:", mask(e.response.text[:500]))
        return 1
    except Exception as e:
        print("호출 실패:", type(e).__name__, mask(e))
        return 1

    if not isinstance(body, dict) or "data" not in body:
        print("예상과 다른 응답:", mask(str(body)[:500]))
        return 1

    rows = body["data"]
    print("searchDate:", search_date)
    print("최상위 키:", list(body.keys()))
    print("totCnt:", body.get("totCnt"), "| hasMore:", body.get("hasMore"), "| 받은 행 수:", len(rows))

    names = [r.get("observatoryName") for r in rows]
    print(f"관측소 {len(set(names))}곳:", names)

    fields = sorted({k for r in rows for k in r})
    print("가이드와 일치:", fields == sorted(EXPECTED_FIELDS))
    print("가이드에 없는 항목:", sorted(set(fields) - set(EXPECTED_FIELDS)))
    print("응답에 없는 항목:", sorted(set(EXPECTED_FIELDS) - set(fields)))
    print("baseDate 값:", sorted({r.get("baseDate") for r in rows}))
    print("값 타입:", sorted({type(v).__name__ for r in rows for v in r.values()}))
    if rows:
        print("첫 행 예시:", rows[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
