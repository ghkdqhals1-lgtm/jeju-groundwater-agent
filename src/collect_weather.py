"""제주데이터허브 756번(일별 기상 정보)을 하루씩 받아 data/external/cache/에 날짜별 JSON으로 저장한다.

- 이미 받은 날짜는 건너뛰므로, 중간에 끊겨도 다시 실행하면 이어서 받는다.
- 1차: 546일을 차례로 돈다. 502, 503, 504, 시간 초과는 같은 날짜를 최대 3번(5초, 15초, 30초 뒤) 다시 시도한다.
  그래도 실패하거나 그 밖의 오류가 나면 그 날짜는 건너뛰고 실패 목록에 넣은 뒤 다음 날짜로 간다.
- 2차: 1차에서 실패한 날짜만 다시 시도한다 (날짜마다 60초 쉬고 시도, 최대 3번).
- 403, 429는 권한·호출 한도 문제이므로 재시도하지 않고 전체를 즉시 멈춘다.
- 끝까지 실패한 날짜는 data/external/weather_failed_dates.txt에 남긴다.
- API 키는 .env의 JEJU_DATAHUB_KEY에서 읽고, 화면과 오류 메시지에는 가려서 출력한다.

실행: python src/collect_weather.py
"""
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests
from dotenv import load_dotenv

# 756번 일별 기상 정보 API 주소 (키는 뒤에 붙인다)
BASE_URL = "https://open.jejudatahub.net/api/proxy/1aD5taat1attaa51Db1511b51ab9Da19"
# 수집 기간: 지하수 감시정 데이터와 같은 기간
START_DATE, END_DATE = "2022-01-01", "2023-06-30"
PAGE_LIMIT = 100      # 페이지당 최대 개수 (가이드상 최대 100)
SLEEP_SEC = 1.0       # 호출 사이 간격(초)
EXT_DIR = Path(__file__).resolve().parent.parent / "data" / "external"
CACHE_DIR = EXT_DIR / "cache"
FAILED_PATH = EXT_DIR / "weather_failed_dates.txt"

load_dotenv()
KEY = os.getenv("JEJU_DATAHUB_KEY")


# 서버 일시 장애로 보고 같은 날짜를 다시 시도할 HTTP 상태 코드 (연결 시간 초과도 다시 시도)
RETRY_STATUS = {502, 503, 504}
# 즉시 전체를 멈출 HTTP 상태 코드 (403 권한 없음, 429 호출 한도 초과)
STOP_STATUS = {403, 429}
# 1차 재시도 대기(초): 1번째 5초, 2번째 15초, 3번째 30초 → 최대 3번 재시도
RETRY_WAITS = [5, 15, 30]
# 2차(실패 날짜 재시도) 대기(초): 날짜마다 60초 쉬고 시도, 최대 3번
SECOND_PASS_WAITS = [60, 60, 60]


class StopCollecting(Exception):
    """전체 수집을 즉시 멈춰야 하는 오류 (403, 429. 재시도하지 않음)."""


class RetryableError(Exception):
    """서버 일시 장애로 보여 다시 시도할 수 있는 오류 (502, 503, 504, 시간 초과)."""


class DayFailed(Exception):
    """이 날짜는 받지 못했지만 다음 날짜로 넘어가도 되는 오류."""


def mask(text):
    """문자열에 들어 있는 API 키를 ***KEY***로 가린다."""
    return str(text).replace(KEY, "***KEY***") if KEY else str(text)


def fetch_page(search_date, number):
    """하루치의 한 페이지를 요청해서 JSON(dict)으로 돌려준다. 오류 종류에 따라 RetryableError, StopCollecting, DayFailed를 낸다."""
    try:
        resp = requests.get(
            f"{BASE_URL}/{KEY}",
            params={"searchDate": search_date, "limit": PAGE_LIMIT, "number": number},
            timeout=30,
        )
    except requests.Timeout as e:
        raise RetryableError(f"시간 초과: {type(e).__name__} {mask(e)}")
    except requests.RequestException as e:
        raise DayFailed(f"네트워크 오류: {type(e).__name__} {mask(e)}")
    if resp.status_code in STOP_STATUS:
        raise StopCollecting(f"HTTP {resp.status_code}: {mask(resp.text[:300])}")
    if resp.status_code in RETRY_STATUS:
        raise RetryableError(f"HTTP {resp.status_code}: {mask(resp.text[:300])}")
    if resp.status_code != 200:
        raise DayFailed(f"HTTP {resp.status_code}: {mask(resp.text[:300])}")
    try:
        body = resp.json()
    except ValueError:
        raise DayFailed(f"JSON이 아닌 응답: {mask(resp.text[:300])}")
    if not isinstance(body, dict) or "data" not in body:
        raise DayFailed(f"예상과 다른 응답 구조: {mask(str(body)[:300])}")
    return body


def fetch_day(search_date):
    """하루치를 hasMore가 false가 될 때까지 페이지를 넘겨 모두 받아 돌려준다."""
    rows, number, tot_cnt = [], 1, None
    while True:
        body = fetch_page(search_date, number)
        rows.extend(body["data"])
        tot_cnt = body.get("totCnt")
        if not body.get("hasMore"):
            break
        number += 1
        time.sleep(SLEEP_SEC)
    return {"searchDate": search_date, "totCnt": tot_cnt, "pages": number, "data": rows}


def fetch_day_with_retry(search_date):
    """1차용: 하루치를 받되, 서버 일시 장애면 5초·15초·30초 기다리며 최대 3번 다시 시도한다. 모두 실패하면 DayFailed를 낸다."""
    for attempt in range(len(RETRY_WAITS) + 1):
        try:
            return fetch_day(search_date)
        except RetryableError as e:
            if attempt == len(RETRY_WAITS):
                raise DayFailed(f"{len(RETRY_WAITS)}번 재시도 후에도 실패 → {e}")
            wait = RETRY_WAITS[attempt]
            print(f"  {search_date} 일시 오류({e}) → {wait}초 뒤 재시도 {attempt + 1}/{len(RETRY_WAITS)}")
            time.sleep(wait)


def fetch_day_second_pass(search_date):
    """2차용: 날짜마다 60초 쉬고 시도하기를 최대 3번 한다. 모두 실패하면 DayFailed를 낸다."""
    last_error = None
    for attempt, wait in enumerate(SECOND_PASS_WAITS, start=1):
        time.sleep(wait)
        try:
            return fetch_day(search_date)
        except (RetryableError, DayFailed) as e:
            last_error = e
            print(f"  [2차] {search_date} 시도 {attempt}/{len(SECOND_PASS_WAITS)} 실패({e})")
    raise DayFailed(f"2차 {len(SECOND_PASS_WAITS)}번 시도 모두 실패 → {last_error}")


def save_day(result):
    """하루치 결과를 임시 파일에 쓴 뒤 이름을 바꿔 저장한다 (중간에 끊겨도 깨진 파일이 남지 않게)."""
    path = CACHE_DIR / f"{result['searchDate']}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def run_pass(dates, fetch, label):
    """날짜 목록을 차례로 받아 저장하고, 받지 못한 날짜와 그 이유를 {날짜: 오류} 사전으로 돌려준다."""
    failed = {}
    for i, d in enumerate(dates, start=1):
        try:
            result = fetch(d)
        except DayFailed as e:
            failed[d] = str(e)
            print(f"  [{label}] {d} 건너뜀 → {e}")
        else:
            save_day(result)
        if i % 50 == 0 or i == len(dates):
            print(f"  [{label}] {i}/{len(dates)} 처리 (실패 누적 {len(failed)}일)")
        time.sleep(SLEEP_SEC)
    return failed


def saved_count():
    """캐시에 저장된 날짜 수를 센다."""
    return len(list(CACHE_DIR.glob("*.json")))


def main():
    """1차로 캐시에 없는 날짜를 모두 돌고, 2차로 실패한 날짜만 다시 시도한 뒤 결과를 출력한다."""
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)  # 백그라운드에서도 진행 상황이 바로 보이게
    if not KEY:
        print("JEJU_DATAHUB_KEY가 .env에 없습니다.")
        return 1
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    dates = [d.strftime("%Y%m%d") for d in pd.date_range(START_DATE, END_DATE, freq="D")]
    todo = [d for d in dates if not (CACHE_DIR / f"{d}.json").exists()]
    print(f"전체 {len(dates)}일 중 이미 받은 날짜 {len(dates) - len(todo)}일, 받을 날짜 {len(todo)}일")

    try:
        failed = run_pass(todo, fetch_day_with_retry, "1차")
        print(f"1차 끝: 저장 {saved_count()}일, 실패 {len(failed)}일")
        if failed:
            failed = run_pass(sorted(failed), fetch_day_second_pass, "2차")
            print(f"2차 끝: 저장 {saved_count()}일, 여전히 실패 {len(failed)}일")
    except StopCollecting as e:
        print(f"[중단] 즉시 멈춤 오류 → {e}")
        print(f"받아 둔 날짜: {saved_count()}일")
        return 2

    FAILED_PATH.write_text("".join(f"{d}\t{msg}\n" for d, msg in sorted(failed.items())), encoding="utf-8")
    print(f"수집 끝: 저장 {saved_count()}/{len(dates)}일, 실패 {len(failed)}일 → {FAILED_PATH.name}")
    if failed:
        print("실패 날짜:", ", ".join(sorted(failed)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
