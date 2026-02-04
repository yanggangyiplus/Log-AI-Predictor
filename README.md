# Log-AI-Predictor
Log Pattern Analyzer & Anomaly Predictor

서버 로그 기반 이상 패턴 분석 및 장애 예측 시스템

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 프로젝트 미리보기

*Streamlit 대시보드 화면 - 실시간 로그 모니터링 및 이상 탐지*

## 핵심 성과 요약

| 항목 | 성과 |
|:---:|:---:|
| **로그 수집** | 실시간 tail 모드 및 배치 모드 지원 |
| **이상 탐지** | AutoEncoder 및 Isolation Forest 기반 |
| **장애 예측** | KNN 기반 유사 패턴 검색 |
| **대시보드** | Streamlit 기반 실시간 모니터링 UI |
| **구현 범위** | 수집부터 예측까지 End-to-End 구현 |

## 문제 정의 & 해결 목적

서버/서비스 장애는 대부분 사전 신호가 있습니다. 응답시간 증가, 5xx 증가, 특정 IP 급증 등이 그 예입니다. 하지만 이러한 신호를 실시간으로 감지하고 대응하는 것은 쉽지 않습니다.

로그가 수십만 ~ 수백만 줄 이상 쌓이고, 사람이 패턴 변화를 실시간으로 감지하기 어렵습니다. 응답시간·에러율·요청 패턴은 미세하게 흔들리다가 갑자기 폭발하는 특성이 있습니다.

이 프로젝트는 "장애 발생 직전의 정상-같은-비정상 패턴"을 머신러닝으로 감지하는 시스템입니다. 일반 모니터링 도구(Grafana, Datadog)와 달리 ML 기반 패턴 분석을 통해 미세한 변화를 조기 감지합니다.

## 프로젝트 개요

### 목적
Nginx/Apache 서버 로그를 실시간으로 수집하여 이상 패턴을 분석하고, 장애 발생 전 경고를 제공하는 시스템을 구축합니다.

### 주요 특징
- 실시간 로그 수집: `tail -f` 기반 스트리밍 및 배치 모드 지원
- 다중 이상 탐지 모델: AutoEncoder, Isolation Forest 모델 선택 가능
- 특징 엔지니어링: 로그 데이터를 시간 윈도우별로 집계하여 통계적 특징 추출
- 장애 예측: 과거 장애 직전 패턴과 비교하여 유사도 기반 경고
- 실시간 대시보드: Streamlit 기반 모니터링 및 시각화
- RESTful API: Flask 기반 데이터 서비스

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

## 모델/기술 스택

| 영역 | 기술 | 선택 이유 |
|------|------|----------|
| **Python** | Python 3.8+ | 데이터 처리 및 ML 모델 구현에 표준 언어 |
| **Deep Learning** | PyTorch | AutoEncoder 모델 구현에 유연한 프레임워크 |
| **ML** | scikit-learn | Isolation Forest, KNN 등 ML 알고리즘 제공 |
| **대시보드** | Streamlit | 빠른 프로토타이핑 및 실시간 모니터링 UI 구축 |
| **시각화** | Plotly | 인터랙티브 차트 및 시각화 제공 |
| **데이터 처리** | Pandas, NumPy | 로그 데이터 처리 및 통계 분석 |
| **API** | Flask | RESTful API 서버 구축 |

## 실험 결과

### 모델 비교

| 모델 | 특징 | 장점 | 단점 |
|------|------|------|------|
| **AutoEncoder** | 정상 로그만 학습, Reconstruction Error 기준 | 미세한 패턴 변화 감지에 강함 | 학습 시간이 오래 걸림 |
| **Isolation Forest** | 트리 기반 이상 탐지 | 빠른 처리 속도, 실시간 예측에 적합 | 정밀도가 AutoEncoder보다 낮음 |

### 핵심 성능 지표

- **로그 처리 속도**: 초당 수백 건의 로그 처리 가능
- **탐지 지연**: 이상 발생 후 수초 내 탐지
- **장애 예측 정확도**: 과거 패턴과 유사도 80% 이상 시 경고

## 핵심 기술 설명

### AutoEncoder 선택 이유

AutoEncoder는 정상 패턴 학습에 매우 강합니다. 장애 발생 전 패턴의 "미세한 변화"를 에러로 잘 포착합니다. 정상 로그만 학습하여 Reconstruction Error를 기준으로 이상을 탐지합니다.

### Isolation Forest 선택 이유

Isolation Forest는 규칙 기반 없이도 비정상 포인트를 빠르게 분리합니다. 실시간 예측에 매우 적합하며 속도가 빠릅니다. 두 모델의 조합은 "빠른 탐지 + 정밀 탐지" 전략을 완성합니다.

### KNN 기반 장애 예측

최근 10분 패턴을 KNN 기반으로 유사 패턴 검색합니다. 과거 장애 직전 패턴과 비교하여 유사도 80% 이상 시 경고 알림을 제공합니다.

## 실행 방법

### Quick Start

```bash
# 저장소 클론
git clone <repository-url>
cd Log-AI-Predictor

# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 대시보드 실행
streamlit run app/web/main.py
```

브라우저에서 `http://localhost:8501` 접속

### 상세 설치 가이드

1. **환경 설정**
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

2. **설정 파일 수정**
`configs/config_collect.yaml`에서 로그 파일 경로 및 타입 설정:
```yaml
collector:
  log_path: "data/raw_logs/nginx_access.log"  # 실제 로그 파일 경로로 변경
  mode: "realtime"
  log_type: "nginx"  # "nginx" 또는 "apache"
```

3. **모델 학습**
```bash
# Isolation Forest 모델 학습 (빠르고 안정적)
python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log --output models/isolation_forest

# PyTorch AutoEncoder 모델 학습 (더 정확)
python scripts/train_pytorch_autoencoder.py --data data/raw_logs/nginx_access.log --output models/pytorch_autoencoder
```

4. **대시보드 실행**
```bash
streamlit run app/web/main.py
```

5. **실시간 이상 탐지 실행**
```bash
python scripts/run_detector.py --model models/isolation_forest
```

## 사용 시나리오 (Use Cases)

### 1. 운영 중인 Nginx 서버 모니터링
**시나리오**: 운영 중인 Nginx 서버의 로그를 실시간으로 모니터링하고 싶을 때  
**해결책**: 실시간 모드로 로그 수집 시작, 대시보드에서 RPS, 에러율, 응답시간 확인  
**효과**: 장애 발생 전 징후를 조기 감지하여 사전 대응 가능

### 2. 장애 직전 패턴 감지
**시나리오**: 과거 장애 발생 직전 패턴과 유사한 상황을 감지하고 싶을 때  
**해결책**: KNN 기반 유사 패턴 검색으로 과거 장애 직전 패턴과 비교  
**효과**: 유사도 80% 이상 시 경고 알림으로 장애 예방

### 3. 배치 로그 분석
**시나리오**: 기존 로그 파일을 배치 모드로 분석하고 싶을 때  
**해결책**: 배치 모드로 로그 수집, AutoEncoder 또는 Isolation Forest 모델로 이상 탐지  
**효과**: 과거 로그에서 이상 패턴을 사후 분석

### 4. 다중 사이트/포트 모니터링
**시나리오**: 여러 포트/사이트의 로그를 동시에 모니터링하고 싶을 때  
**해결책**: 각 포트/사이트별 로그 파일 경로만 설정하면 자동으로 분석  
**효과**: 여러 서비스를 통합 모니터링

## 한계 & 개선 방향

### 현재 한계

- **데이터베이스 연동**: 현재는 메모리 기반 처리, 대용량 로그 처리 시 제한
- **분산 처리**: 단일 서버 환경에서만 동작, 대규모 로그 처리 제한
- **모델 재학습**: 수동 모델 재학습 필요, 자동 재학습 파이프라인 부재
- **알림 시스템**: Slack, Email 등 외부 알림 시스템 연동 미지원

### 개선 방향

- **데이터베이스 연동**: PostgreSQL, MongoDB 등 데이터베이스 연동으로 대용량 로그 처리
- **분산 처리**: Apache Kafka, Apache Spark 등을 활용한 분산 처리 아키텍처 설계
- **모델 재학습 파이프라인**: 주기적 모델 재학습 및 성능 모니터링 자동화
- **알림 시스템**: Slack, Email, PagerDuty 등 외부 알림 시스템 연동
- **실시간 스트리밍**: Apache Kafka를 활용한 실시간 로그 스트리밍
- **성능 최적화**: 모델 양자화 및 최적화로 추론 속도 향상

## 개인 기여도

이 프로젝트는 **개인 프로젝트**로, 모든 작업을 직접 수행했습니다.

### 엔드투엔드 데이터 파이프라인 설계
- Raw 수집 → 전처리 → 특징 추출 → 이상 탐지 → 장애 예측 → UI까지 풀 사이클 개발
- Collector / Preprocessing / Anomaly / Prediction / API 계층 완전 분리 구성
- 재사용성과 유지보수성 높은 구조 설계

### 로그 파싱 및 데이터 엔지니어링
- Nginx 로그 파서 직접 구현
- 실시간 스트리밍 (`tail -f`) 및 배치 모드 지원
- 오류 처리, 재시도 로직, 버퍼 관리 구현

### 특징 엔지니어링 및 시계열 분석
- 시간 윈도우별 집계 및 통계적 특징 추출
- RPS, 에러율, 응답시간 분포 등 다차원 특징 생성
- 시계열 패턴 분석 및 변화율 계산

### 이상 탐지 모델링
- AutoEncoder 기반 이상 탐지 직접 구현
- Isolation Forest 모델 비교 및 선택 가능
- Reconstruction Error 기반 임계값 설정
- PyTorch 및 scikit-learn 활용

### 장애 예측 및 패턴 매칭
- KNN 기반 유사 패턴 검색 구현
- 과거 장애 직전 패턴과 현재 패턴 비교
- 유사도 기반 경고 시스템 구현

### 데이터 시각화 & UX
- Streamlit 기반 실시간 대시보드 구현
- Plotly 인터랙티브 차트 (RPS, 에러율, 응답시간, 이상 점수)
- 색상 코딩된 로그 테이블 및 알림 시스템
- 데이터 다운로드 기능 (CSV/JSON)

### 백엔드·서비스 설계
- Flask 기반 RESTful API 설계
- API 엔드포인트 설계 및 구현
- Dashboard ↔ API 연동 완성

### 문서화
- 아키텍처 문서 작성
- 설치·실행 가이드 작성
- 모델 비교 문서 작성
- API 문서 작성

## 프로젝트 구조

```
Log-AI-Predictor/
├── app/                      # 애플리케이션 레이어
│   └── web/
│       └── main.py         # Streamlit 대시보드 (메인 애플리케이션)
│   ├── api.py               # REST API 엔드포인트
│   ├── layout/              # UI 컴포넌트
│   │   ├── sidebar.py       # 사이드바 레이아웃
│   │   ├── metrics.py       # 메트릭 카드
│   │   ├── charts.py        # 차트 컴포넌트
│   │   ├── alerts.py        # 알림 컴포넌트
│   │   ├── logs_table.py    # 로그 테이블
│   │   └── downloads.py     # 다운로드 기능
│   ├── services/            # 비즈니스 로직 서비스
│   │   ├── log_service.py   # 로그 수집 서비스
│   │   ├── model_service.py # 모델 관리 서비스
│   │   ├── feature_service.py # 특징 추출 서비스
│   │   ├── anomaly_service.py # 이상 탐지 서비스
│   │   └── alert_service.py # 알림 서비스
│   └── utils/               # 유틸리티
│       ├── constants.py     # 상수 정의
│       ├── cache.py         # 캐시 관리
│       ├── formatter.py     # 데이터 포맷팅
│       └── session_state.py # 세션 상태 관리
├── configs/                 # 설정 파일
│   ├── config_collect.yaml  # 로그 수집 설정
│   ├── config_model.yaml   # 모델 설정
│   └── config_alert.yaml   # 알림 설정
├── data/                    # 데이터 디렉토리
│   ├── raw_logs/           # 원본 로그 파일
│   ├── processed/          # 전처리된 데이터
│   └── database/           # 데이터베이스 파일
├── src/                     # 소스 코드
│   ├── collector/          # 로그 수집 모듈
│   │   ├── tail_collector.py      # 스레드 기반 수집기
│   │   ├── polling_collector.py   # Polling 방식 수집기
│   │   ├── nginx_parser.py        # Nginx 로그 파서
│   │   ├── apache_parser.py       # Apache 로그 파서
│   │   └── collector_manager.py   # 수집 관리자
│   ├── preprocessing/      # 전처리 모듈
│   │   └── feature_engineering.py # 특징 추출
│   ├── anomaly/            # 이상 탐지 모듈
│   │   ├── autoencoder.py         # TensorFlow AutoEncoder
│   │   ├── autoencoder_pytorch.py # PyTorch AutoEncoder
│   │   ├── isolation_forest.py    # Isolation Forest
│   │   └── detector_manager.py   # 탐지기 관리자
│   ├── prediction/         # 장애 예측 모듈
│   │   └── failure_predictor.py  # 장애 예측기
│   └── utils/              # 공통 유틸리티
├── scripts/                # 실행 스크립트
│   ├── run_collector.py    # 로그 수집기 실행
│   ├── train_isolation_forest.py      # Isolation Forest 학습
│   ├── train_pytorch_autoencoder.py  # PyTorch AutoEncoder 학습
│   └── run_detector.py     # 이상 탐지기 실행
├── docs/                   # 문서
│   ├── MODEL_COMPARISON.md # 모델 비교 문서
│   ├── TENSORFLOW_ISSUE.md # TensorFlow 이슈 문서
│   └── USAGE.md            # 사용 가이드
├── models/                 # 학습된 모델 저장소
├── README.md               # 프로젝트 문서
├── requirements.txt        # Python 패키지 의존성
└── LICENSE                 # 라이선스
```

## 라이선스 & 작성자

이 프로젝트는 MIT 라이선스를 따릅니다.

**작성자**: yanggangyi

- GitHub: [@yanggangyiplus](https://github.com/yanggangyiplus)

## 상세 문서

- [모델 비교](docs/MODEL_COMPARISON.md)
- [사용 가이드](docs/USAGE.md)
- [TensorFlow 이슈](docs/TENSORFLOW_ISSUE.md)
