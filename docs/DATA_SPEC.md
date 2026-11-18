# 데이터 활용 명세

출처: 제주데이터허브(jejudatahub.net). 제조 공기업(제주개발공사)의 취수원 감시 센서 데이터이며, 설비 모니터링 방식(이상 유형 진단, 예측)을 수원 감시 센서 데이터에 적용했습니다.

## 1. 사용 데이터 4종

| 번호 | 이름 | 형식 | 기간 | 규모 | 컬럼 |
|---|---|---|---|---|---|
| 142 | 제주특별자치도개발공사 지하수 감시정 지하수위 정보 | CSV 2개(2022년, 2023년), CP949 | 2022-01-01 00:00 ~ 2023-06-30 23:00 | 2022년 8,760행 + 2023년 4,344행 = 13,104행 | `일시`, `1수원 감시정1 (m)` ~ `2수원 감시정3 (m)` (6개) |
| 79 | 제주특별자치도개발공사 지하수 감시정 지하수온 정보 | CSV 2개(2022년, 2023년), CP949 | 위와 같음 | 8,760 + 4,344 = 13,104행 | `일시`, `1수원 감시정1` ~ `2수원 감시정3` (6개) |
| 756 | 일별 기상 정보 | API, 하루씩 수집 | 2022-01-01 ~ 2023-06-30 (546일) | 관측소×날짜 19,023행 × 15컬럼 (`data/external/weather_daily_2022_2023.csv`) | `baseDate`, `observatoryName`, `observatoryCode`, `zone`(추가), `altitude_m`(추가), `dailyRainfall` 외 기온·풍속 등 |
| 757 | 기상관측소 정보 | API | — | 42행 × 19컬럼 (`data/external/weather_stations.csv`) | `observatoryCode`, `observatoryName`, `longitude`, `latitude`, `seaLevelAltitude`, `zone`(추가) 등 |

- 수위·수온은 1시간 간격이며 빠진 시각·중복·결측이 없습니다. 2023년 파일은 1~6월입니다. 감시정 좌표·고유 ID는 파일에 없습니다.
- 수위 값은 약 245~331 m, 수온은 약 13.4~14.1 °C입니다.
- 기상은 하루 31~35개 관측소가 응답했고(35곳인 날 469일), 2022-01-20은 응답이 0건이라 빈 날짜로 두었습니다. `dailyRainfall`의 빈 값은 127개(관측소·날짜 단위)이고 음수나 특이 코드는 없으며, 최댓값은 703.0 mm(윗세오름, 2022-09-05)입니다.
- 기상청 주요 관측소(제주·서귀포·성산·고산)는 756번에 없어 허브 관측소만 사용했습니다.
- 수집 실패 날짜 목록(`weather_failed_dates.txt`)은 비어 있습니다.

### 측정값 방향
수위 파일 값은 강수 뒤 값이 줄어드는 반응과 표선유역 연구를 근거로 '물까지의 깊이'로 해석했습니다.
- 6개 감시정 모두 과거 30일 누적 강수가 많을수록 이후 7일간 값이 내려갑니다(상관 −0.36~−0.87, 2022년).
- 1수원 감시정3의 값이 크게 줄어든 두 번이 큰 비 3일 뒤에 시작했습니다.
- 개발공사 연구(신문주 외, 2020, 한국수자원학회논문집 53(7))가 표선유역 중산간 관측정의 지하수위가 선행 강수 뒤 13~40일 지체로 상승한다고 보고했습니다.

해석이 달라지면 `src/config.py`의 `VALUE_IS_DEPTH` 한 줄(True/False)로 판정·설명 방향이 전환되도록 설계했습니다. 원본 값은 어느 쪽이든 바꾸지 않습니다.

### 기준 시점
데이터가 2023-06-30까지라 '지금'은 2023-06-30 기준입니다. 새 파일을 `data/raw`에 넣고 전처리부터 다시 실행하면 기준일이 따라갑니다. 실시간 연결은 향후 확장 방향입니다.

## 2. 이용조건
- 허브 142번 페이지에는 등록일·수정일·수집기간·간략설명·태그만 표시되고, 756·757번 API 페이지, API 가이드(HWP), 제주빅데이터센터 데이터명세서에도 별도 이용조건·라이선스 표시가 없습니다. 79번은 142번과 같은 개발공사 데이터로 같은 형식입니다.
- 허브에 별도 이용조건 표시가 없어 출처를 표기하고 대회 목적으로만 사용합니다.
- 제출 ZIP에는 심사 재현을 위해 원본·가공 데이터를 포함하며, 대회 심사 목적으로만 포함하고 외부 공개·재배포를 금지합니다.

## 3. 단위
- 수위 측정값: m (컬럼 이름 ` (m)`)
- 수온: 표기 없음, 값 범위(약 13.4~14.1)로 °C로 판단
- 강수량(`dailyRainfall`): 허브 756번 API 가이드에는 단위 표기가 없고, 제주빅데이터센터 데이터 명세서(날씨 정보 시트)에 일강수량 mm로 적혀 있음
- 고도(`seaLevelAltitude`, 가공 컬럼 `altitude_m`): 허브 757번 API 가이드에는 단위 표기가 없고, 같은 데이터 명세서에 해발고도 m로 적혀 있음. zone은 해안(200 m 미만) 19곳, 중산간(200~700 m) 8곳, 산지(700 m 이상) 8곳

## 4. 전처리·분석 흐름

| 순서 | 스크립트 | 입력 | 출력 |
|---|---|---|---|
| 1 | `src/preprocess.py` | `data/raw/*.csv` 4개 | `data/processed/groundwater_hourly.csv` (13,104행 × 12열: 수위 6 + 수온 6) |
| 2 | `src/collect_weather.py` | 허브 756번 API | `data/external/cache/YYYYMMDD.json` (546개) |
| 3 | `src/weather_clean.py` | 캐시, 757번 관측소 목록 | `weather_daily_2022_2023.csv`, `weather_stations.csv`(zone 추가) |
| 4 | `src/rain_analysis.py` | 시간별 표, 일별 기상 | `rain_lag_corr.csv` |
| 5 | `src/sensor_check.py` | 시간별 표, 기상 | `sensor_events.csv`(47건), `sensor_events_level.png` |
| 6 | `src/status_check.py` | 시간별 표, 이벤트 | 화면 출력, `status_levels.png` |
| 7 | `src/forecast.py` | 시간별 표, 기상 | `forecast_eval_v1.csv`, `forecast_eval_v1_interval.csv`, `forecast_eval_v1.png`, `forecast_v1_model.json` |
| 8 | `src/agent.py`, `src/app.py` | 위 결과 전부 | 답변, `logs/agent_log.jsonl`, `data/processed/charts/*.png` |

`forecast_eval_v0.csv·png`는 초기 모델(v0) 평가 결과입니다. 2번은 허브 API, 8번은 Claude API를 호출합니다. 같은 원본이면 1·5·6·7번은 몇 번을 실행해도 결과 파일이 같습니다.

## 5. 외부 데이터·참고 문헌
- 제주도 지하수 관리수위 설정에 관한 연구, 2005, https://koreascience.kr/article/JAKO200518718194061.page
- 신문주 외, 2020, 한국수자원학회논문집 53(7), https://koreascience.kr/article/JAKO202021961382496.pdf
- 가뭄으로 제주시 수리계 28곳 농업용수 제한급수(2026), https://m.seoul.co.kr/news/society/2026/08/06/20260806500068
- 기상은 허브 756·757번만 사용했고 기상청 외부 자료는 사용하지 않았습니다.

## 6. 오픈소스 라이브러리
requirements.txt에 고정한 7개 패키지이며, 라이선스는 설치된 패키지의 pip 메타데이터 기준입니다.

| 이름 | 버전 | 라이선스 |
|---|---|---|
| anthropic | 2.8.0 | Apache-2.0 |
| matplotlib | 3.11.2 | matplotlib 라이선스(PSF 계열) |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| pandas | 3.1.1 | BSD-3-Clause |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| requests | 2.34.2 | Apache-2.0 |
| streamlit | 1.66.1 | Apache-2.0 |

예측 모델(릿지 회귀)은 numpy로 직접 구현했습니다.
