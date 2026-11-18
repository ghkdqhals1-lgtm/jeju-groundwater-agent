"""지하수 감시정 원본 CSV 4개(수위·수온 × 2022·2023)를 하나의 시간별 표로 합친다.

- 원본(data/raw)은 읽기만 하고 수정하지 않는다.
- 결과: data/processed/groundwater_hourly.csv
  컬럼: datetime, level_1수원감시정1 ... level_2수원감시정3, temp_1수원감시정1 ... temp_2수원감시정3
- 같은 원본이면 몇 번을 실행해도 같은 파일이 나온다.

실행: python src/preprocess.py
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "processed" / "groundwater_hourly.csv"
ENCODING = "cp949"          # 원본 CSV 인코딩
YEARS = ["2022", "2023"]
# 원본 파일 이름에 들어 있는 항목 이름 → 결과 컬럼 접두어
KINDS = {"지하수위": "level", "지하수온": "temp"}


def raw_path(kind, year):
    """항목(지하수위/지하수온)과 연도로 원본 파일 경로를 찾는다."""
    return RAW_DIR / f"제주특별자치도개발공사_지하수 감시정 {kind} 정보_{year}년.csv"


def read_one(kind, year):
    """원본 CSV 한 개를 읽어, 시각을 인덱스로 하고 컬럼 이름을 '접두어_감시정이름'으로 바꾼다."""
    df = pd.read_csv(raw_path(kind, year), encoding=ENCODING)
    # 시각: 수위 파일은 '0:00', 수온 파일은 '00:00' 형식이라 형식을 지정해 같은 값으로 만든다
    df.index = pd.to_datetime(df.pop("일시"), format="%Y-%m-%d %H:%M")
    df.index.name = "datetime"
    # 컬럼 이름: '1수원 감시정1 (m)' → 'level_1수원감시정1' (단위와 공백 제거)
    prefix = KINDS[kind]
    df.columns = [f"{prefix}_" + c.replace("(m)", "").replace(" ", "") for c in df.columns]
    # 값: 수온 파일은 값 뒤에 공백이 있어 숫자로 확실히 바꿔 둔다
    return df.apply(pd.to_numeric, errors="coerce")


def build():
    """연도별로 수위와 수온을 시각 기준으로 합친 뒤, 연도를 위아래로 이어 붙여 돌려준다."""
    parts = []
    for year in YEARS:
        level, temp = read_one("지하수위", year), read_one("지하수온", year)
        # 두 파일의 시각이 완전히 같아야 한다 (다르면 합치기 전에 멈춘다)
        if not level.index.equals(temp.index):
            raise ValueError(f"{year}년 수위·수온 파일의 시각이 서로 다릅니다.")
        parts.append(level.join(temp))
    df = pd.concat(parts).sort_index()
    if df.index.duplicated().any():
        raise ValueError("이어 붙인 결과에 중복된 시각이 있습니다.")
    return df


def main():
    """전처리 결과를 저장하고 행 수, 기간, 빠진 시각, 결측 개수를 출력한다."""
    sys.stdout.reconfigure(encoding="utf-8")
    df = build()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M", lineterminator="\n")

    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    print("저장:", OUT_PATH)
    print("행 수:", len(df), "| 열 수:", df.shape[1])
    print("기간:", df.index.min(), "~", df.index.max())
    print("빠진 시각:", len(full.difference(df.index)))
    print("결측 개수:", int(df.isna().sum().sum()), dict(df.isna().sum()))
    print("컬럼:", list(df.columns))
    return 0


if __name__ == "__main__":
    sys.exit(main())
