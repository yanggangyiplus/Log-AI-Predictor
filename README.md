# Log Pattern Analyzer & Anomaly Predictor

서버 로그 기반 이상 패턴 분석 및 장애 예측 시스템

> **Nginx 서버 로그를 실시간으로 수집 → 파싱 → 특징 추출 → 이상 탐지 모델 → 장애 발생 전 패턴 자동 감지 → 실시간 대시보드로 모니터링하는 플랫폼**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 목차

- [TL;DR](#tldr)
- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [핵심 기능](#핵심-기능)
- [기술 스택](#기술-스택)
- [사용 시나리오](#사용-시나리오)
- [역량](#역량)
- [설치 및 실행](#설치-및-실행)
- [프로젝트 구조](#프로젝트-구조)
- [설정](#설정)
- [상세 문서](#상세-문서)
- [현재 버전 상태](#현재-버전-상태)

---

## TL;DR

**핵심 요약**:
- **실시간 로그 수집**: Nginx/Apache 로그를 실시간 또는 배치 모드로 수집
- **이상 탐지**: AutoEncoder 및 Isolation Forest 기반 이상 패턴 자동 탐지
- **장애 예측**: KNN 기반 유사 패턴 검색으로 장애 발생 전 경고
- **실시간 대시보드**: Streamlit 기반 모니터링 및 시각화
- **RESTful API**: Flask 기반 데이터 서비스 제공

---

## 프로젝트 개요

### 목적

서버/서비스 장애는 대부분 사전 신호가 있습니다 (응답시간 증가, 5xx 증가, 특정 IP 급증 등). 이 프로젝트는 바로 그 신호를 이상 탐지 모델로 조기 감지하는 시스템입니다.

### 이 프로젝트가 해결하는 문제

현업에서는 아래 이유로 장애 조기 탐지가 매우 어렵습니다:

- 로그가 수십만 ~ 수백만 줄 이상 쌓임
- 사람이 패턴 변화를 실시간으로 감지하기 어려움
- 응답시간·에러율·요청 패턴은 미세하게 흔들리다가 갑자기 폭발
- 장애 직전에는 정상 범위 내에서 "조금씩 이상"해지는 증상 존재

이 프로젝트는 "장애 발생 직전의 정상-같은-비정상 패턴"을 머신러닝으로 감지한다는 점에서 일반 모니터링 도구(Grafana, Datadog)와 차별됩니다.

### 현재 데모에서 사용하는 로그

이 프로젝트는 실제 서버 로그와 동일한 구조를 유지하지만, 데모 버전에서는 다음 경로의 **샘플 Nginx 로그 파일**을 사용합니다:

```
data/raw_logs/nginx_access.log
```

**이 샘플 로그에는 다음 정보가 포함되어 있습니다:**
- Timestamp (요청 시간)
- HTTP Method / URL Path
- Status Code (200/404/500)
- Response Time (ms)
- Client IP
- User-Agent

**데모 작동 방식:**
- Streamlit 대시보드 실행 시 `collector_manager`가 `configs/config_collect.yaml`에 설정된 경로를 읽습니다
- `tail_collector.py`가 해당 파일을 실시간 tail 모드로 스트림 형태로 읽어 실시간 로그처럼 처리합니다

**샘플 로그를 사용하는 이유:**
- 로컬 환경에서 바로 실행해볼 수 있도록 하기 위함
- AutoEncoder/Isolation Forest 모델을 학습시키기 위한 표준 패턴 제공
- 실제 운영 서버와 연결할 때는 `log_path`만 변경하면 즉시 실시간 모니터링 가능

**실제 서버 로그로 변경하려면:**

`configs/config_collect.yaml`에서 아래와 같이 변경하면 됩니다:

```yaml
collector:
  log_path: "/var/log/nginx/access.log"  # 실제 서버 로그 경로
  mode: "realtime"
  log_type: "nginx"
```

바로 실시간 모니터링이 동작하게 됩니다.

### 주요 특징

- **실시간 로그 수집**: `tail -f` 기반 스트리밍 및 배치 모드 지원
- **다중 이상 탐지 모델**: AutoEncoder, Isolation Forest 모델 선택 가능
- **특징 엔지니어링**: 로그 데이터를 시간 윈도우별로 집계하여 통계적 특징 추출
- **장애 예측**: 과거 장애 직전 패턴과 비교하여 유사도 기반 경고
- **실시간 대시보드**: Streamlit 기반 모니터링 및 시각화
- **RESTful API**: Flask 기반 데이터 서비스

---

## 시스템 아키텍처

### 전체 시스템 구조

```
┌──────────┐     ┌──────────────┐     ┌────────────────────┐
│  NGINX    │ --> │ Log Collector │ --> │  Preprocessing      │
└──────────┘     └──────────────┘     └────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────┐
                       │ Feature Engineering │
                       └────────────────────┘
                                   │
                                   ▼
                  ┌──────────────────────────────────┐
                  │ AutoEncoder / Isolation Forest   │
                  └──────────────────────────────────┘
                                   │
                                   ▼
                       ┌────────────────────┐
                       │ Failure Predictor   │
                       └────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Streamlit Monitoring UI  │
                    └──────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │ Alerts & Notifications   │
                    └──────────────────────────┘
```

### 데이터 파이프라인

```
Raw Logs
    │
    ▼
┌──────────────────────────┐
│   Log Collection         │
│  - Real-time / Batch     │
│  - Nginx Parser          │
└───────────┬──────────────┘
            │
            ▼
┌─────────────────────────┐
│   Feature Engineering   │
│  - Time Window Agg      │
│  - Statistical Features │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Anomaly Detection     │
│  - AutoEncoder          │
│  - Isolation Forest     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Failure Prediction    │
│  - KNN Pattern Match    │
│  - Similarity Score     │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────┐
│   Dashboard & Alerts    │
│  - Streamlit UI         │
│  - Real-time Monitoring │
└─────────────────────────┘
```

---

## 핵심 기능

### 1. 로그 수집 (Real-time / Batch)

- **Nginx 로그 파싱**: Combined Log Format 지원
- **Apache 로그 파싱**: Combined 및 Common Log Format 지원
- **다중 사이트/포트 지원**: 로그 파일 경로만 설정하면 어떤 포트/사이트의 로그도 분석 가능
- 실시간 `tail -f` 기반 스트리밍
- Batch 모드 (주기적 수집)

### 2. 로그 파싱

추출 필드:
- IP, Timestamp, Method, URL Path
- Status Code, Response Time, User-Agent

### 3. Feature Engineering

시간 윈도우별 집계:
- 5xx 비율, 404 증가율
- 초당 요청 수 (RPS)
- 분당 오류 증가율
- 특정 API 호출 급증
- Response Time 분포

### 4. 이상 탐지

- **AutoEncoder**: 정상 로그만 학습, Reconstruction Error 기준 이상 탐지
- **Isolation Forest**: 트리 기반 이상 탐지
- **PyTorch AutoEncoder**: PyTorch 기반 구현 (TensorFlow 대안)

#### 왜 AutoEncoder + Isolation Forest인가?

**AutoEncoder**
- 정상 패턴 학습에 매우 강함
- 장애 발생 전 패턴의 "미세한 변화"를 에러로 잘 포착

**Isolation Forest**
- 규칙 기반 없이도 비정상 포인트를 빠르게 분리
- 실시간 예측에 매우 적합 (속도 빠름)

두 모델의 조합은 "빠른 탐지 + 정밀 탐지" 전략을 완성합니다.

### 5. 장애 예측 (Failure Prediction)

- 최근 10분 패턴 → KNN 기반 유사 패턴 검색
- 과거 장애 직전 패턴과 비교
- 유사도 80% 이상 시 경고 알림

### 6. 실시간 대시보드 (Streamlit)

- 실시간 로그 스트림
- 요청 수 (RPS) 그래프
- 에러 비율 그래프
- 응답 시간 추이
- 이상 탐지 점수 시각화
- 알림 패널

### 7. 알림 시스템

조건:
- 5xx 비율 > 5%
- AutoEncoder error > threshold
- 유사도 패턴 경고 발생 시

---

## 기술 스택

- **Python 3.8+**
- **PyTorch**: 딥러닝 모델 (AutoEncoder)
- **scikit-learn**: 머신러닝 (Isolation Forest, KNN)
- **Streamlit**: 실시간 대시보드
- **Plotly**: 인터랙티브 시각화
- **Pandas/NumPy**: 데이터 처리
- **Flask**: REST API

---

## 역량

### 1) 엔드투엔드 데이터 파이프라인 설계 능력

- **Raw 수집 → 전처리 → 특징 추출 → 이상 탐지 → 장애 예측 → UI까지 풀 사이클 개발**
- Collector / Preprocessing / Anomaly / Prediction / API 계층 완전 분리 구성
- 재사용성과 유지보수성 높은 구조

### 2) 로그 파싱 및 데이터 엔지니어링 역량

- **Nginx 로그 파서 직접 구현**
- 실시간 스트리밍 (`tail -f`) 및 배치 모드 지원
- 오류 처리, 재시도 로직, 버퍼 관리

### 3) 특징 엔지니어링 및 시계열 분석 능력

- **시간 윈도우별 집계 및 통계적 특징 추출**
- RPS, 에러율, 응답시간 분포 등 다차원 특징 생성
- 시계열 패턴 분석 및 변화율 계산

### 4) 이상 탐지 모델링 능력

- **AutoEncoder 기반 이상 탐지 직접 구현**
- Isolation Forest 모델 비교 및 선택 가능
- Reconstruction Error 기반 임계값 설정
- PyTorch 및 scikit-learn 활용

### 5) 장애 예측 및 패턴 매칭 능력

- **KNN 기반 유사 패턴 검색 구현**
- 과거 장애 직전 패턴과 현재 패턴 비교
- 유사도 기반 경고 시스템

### 6) 데이터 시각화 & UX 능력

- **Streamlit 기반 실시간 대시보드**
- Plotly 인터랙티브 차트 (RPS, 에러율, 응답시간, 이상 점수)
- 색상 코딩된 로그 테이블 및 알림 시스템
- 데이터 다운로드 기능 (CSV/JSON)

### 7) 백엔드·서비스 설계 역량

- **Flask 기반 RESTful API 설계**
- API 엔드포인트 설계 및 구현
- Dashboard ↔ API 연동 완성

### 8) 문서화·구조화 능력

- 아키텍처 문서
- 설치·실행 가이드
- 모델 비교 문서
- API 문서

---

## 사용 시나리오

실제 운영 환경에서의 사용 흐름:

1. **운영 중인 Nginx 서버에 log collector 실행**
   - 실시간 모드로 로그 수집 시작
   - 또는 기존 로그 파일을 배치 모드로 처리

2. **대시보드 접속 (포트 8501)**
   - Streamlit 대시보드 실행
   - 브라우저에서 `http://localhost:8501` 접속

3. **RPS, 에러율, 응답시간 실시간 확인**
   - 실시간 통계 대시보드에서 주요 메트릭 모니터링
   - 시각화 그래프로 트렌드 분석

4. **AutoEncoder 모델이 미세한 패턴 변화를 감지**
   - 정상 패턴에서 벗어나는 미세한 변화를 Reconstruction Error로 포착
   - Isolation Forest가 빠르게 이상 포인트 분리

5. **비정상 신호 감지 시 알림 패널에 표시**
   - 5xx 에러율 임계값 초과
   - 응답 시간 급증
   - 이상 탐지 점수 상승

6. **장애 발생 전 유사 패턴 발견 → 조기 경보 발생**
   - KNN 기반 패턴 매칭으로 과거 장애 직전 패턴과 비교
   - 유사도 80% 이상 시 경고 알림

이러한 워크플로우를 통해 장애 발생 전 조기 대응이 가능합니다.

---

## 설치 및 실행

### 1. 환경 설정

```bash
# 저장소 클론
git clone <repository-url>
cd Log-AI-Predictor

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 설정 파일 수정

`configs/config_collect.yaml`에서 로그 파일 경로 및 타입 설정:

```yaml
collector:
  # 로그 파일 경로 (다른 포트/사이트의 로그도 경로만 변경하면 분석 가능)
  log_path: "data/raw_logs/nginx_access.log"  # 실제 로그 파일 경로로 변경
  mode: "realtime"
  log_type: "nginx"  # "nginx" 또는 "apache"
  
  # Apache 사용 시
  # apache_format: "combined"  # "combined" 또는 "common"
```

**다른 포트/사이트 로그 분석:**
다른 포트/도메인의 로그도, 해당 access.log 경로만 지정하면 동일하게 분석 가능합니다. 자세한 예시는 [설정](#설정) 섹션 참고.

### 3. 모델 학습

```bash
# Isolation Forest 모델 학습 (빠르고 안정적)
python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log --output models/isolation_forest

# PyTorch AutoEncoder 모델 학습 (더 정확)
python scripts/train_pytorch_autoencoder.py --data data/raw_logs/nginx_access.log --output models/pytorch_autoencoder
```

### 4. 대시보드 실행

```bash
# Streamlit 대시보드 실행
streamlit run app/dashboard.py

# 또는 스크립트 사용
bash scripts/run_dashboard.sh
```

브라우저에서 `http://localhost:8501` 접속

### 5. 실시간 이상 탐지 실행

```bash
python scripts/run_detector.py --model models/isolation_forest
```

---

## 프로젝트 구조

```
Log-AI-Predictor/
├── app/
│   ├── dashboard.py          # Streamlit 대시보드
│   └── api.py                # REST API 엔드포인트
├── configs/
│   ├── config_collect.yaml   # 로그 수집 설정
│   ├── config_model.yaml     # 모델 설정
│   └── config_alert.yaml     # 알림 설정
├── data/
│   ├── raw_logs/             # 원본 로그 파일
│   ├── processed/            # 전처리된 데이터
│   └── database/             # 데이터베이스 파일
├── src/
│   ├── collector/            # 로그 수집 모듈
│   │   ├── tail_collector.py
│   │   ├── nginx_parser.py
│   │   ├── apache_parser.py
│   │   └── collector_manager.py
│   ├── preprocessing/        # 전처리 모듈
│   │   └── feature_engineering.py
│   ├── anomaly/              # 이상 탐지 모듈
│   │   ├── autoencoder.py
│   │   ├── autoencoder_pytorch.py
│   │   ├── isolation_forest.py
│   │   └── detector_manager.py
│   ├── prediction/           # 장애 예측 모듈
│   │   └── failure_predictor.py
│   └── utils/                # 유틸리티
├── scripts/
│   ├── run_collector.py      # 로그 수집기 실행
│   ├── train_isolation_forest.py
│   ├── train_pytorch_autoencoder.py
│   ├── run_detector.py       # 이상 탐지기 실행
│   └── run_dashboard.sh      # 대시보드 실행
├── docs/                     # 문서
├── README.md
└── requirements.txt
```

---

## 설정

### 로그 수집 설정 (`configs/config_collect.yaml`)

```yaml
collector:
  # 로그 파일 경로 (다른 포트/사이트의 로그도 경로만 변경하면 분석 가능)
  log_path: "data/raw_logs/nginx_access.log"
  mode: "realtime"  # realtime 또는 batch
  batch_interval: 3600  # 배치 모드일 때 수집 주기 (초)
  log_type: "nginx"  # "nginx" 또는 "apache"
  apache_format: "combined"  # Apache 사용 시: "combined" 또는 "common"
  buffer_size: 1000
```

**다른 포트/사이트 로그 분석:**

포트는 로그 파일과 직접적인 관련이 없습니다. 각 포트/사이트별로 별도의 로그 파일이 있다면, 해당 파일 경로만 설정하면 자동으로 분석됩니다.

**지원하는 로그 형식:**
- Nginx Combined Log Format (기본)
- Apache Combined Log Format
- Apache Common Log Format

**사용 예시:**
- 포트 80: `/var/log/nginx/default_access.log`
- 포트 8080: `/var/log/nginx/app_access.log`
- 포트 3000: `/var/log/nginx/api_access.log`
- Apache: `/var/log/apache2/access.log`

실시간 모드에서는 `tail -f`로 실시간 스트리밍하여 어떤 포트/사이트의 로그도 실시간으로 분석할 수 있습니다.

### 모델 설정 (`configs/config_model.yaml`)

```yaml
model:
  anomaly_detector: "isolation_forest"  # isolation_forest, autoencoder
  
  autoencoder:
    input_dim: 50
    encoding_dim: 16
    hidden_layers: [32, 24]
    epochs: 50
    batch_size: 32
    
  anomaly_threshold: 0.75
```

### 알림 설정 (`configs/config_alert.yaml`)

```yaml
alert:
  enabled: true
  conditions:
    error_rate_threshold: 5.0  # 5xx 에러 비율 임계값 (%)
    reconstruction_error_threshold: 0.75
    response_time_threshold: 1000  # 응답 시간 임계값 (ms)
```

---

## 상세 문서

- [아키텍처 문서](docs/ARCHITECTURE.md) (예정)
- [모델 비교](docs/MODEL_COMPARISON.md)
- [사용 가이드](docs/USAGE.md)
- [TensorFlow 이슈](docs/TENSORFLOW_ISSUE.md)

---

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

---

## 참고사항

- 실제 프로덕션 환경에서는 데이터베이스 연동 권장
- 대용량 로그 처리를 위한 분산 처리 고려
- 모델 성능 모니터링 및 재학습 파이프라인 구축 권장

---

## 현재 버전 상태

- Nginx 샘플 로그 기반 이상 탐지 + 대시보드 완성
- Isolation Forest / PyTorch AutoEncoder 모델 학습 및 적용
- 실시간 수집(가상 tail 모드) + Streamlit 모니터링 동작
- DB 연동 및 Slack/Email 알림은 향후 버전(v1)에서 계획
