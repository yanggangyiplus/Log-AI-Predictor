# 사용 가이드

## 빠른 시작 예제

### 1. 샘플 데이터로 테스트

```bash
# 샘플 로그 파일 사용
cp data/raw_logs/sample_nginx.log data/raw_logs/nginx_access.log

# 모델 학습
python scripts/train_isolation_forest.py --data data/raw_logs/sample_nginx.log --output models/isolation_forest

# 대시보드 실행
streamlit run app/dashboard.py
```

### 2. 실제 Nginx 로그 사용

```bash
# Nginx 로그 파일 경로 설정
# configs/config_collect.yaml 파일에서 log_path 수정

# 실시간 수집 시작
python scripts/run_collector.py --mode realtime

# 다른 터미널에서 이상 탐지 실행
python scripts/run_detector.py --model models/isolation_forest
```

## 주요 워크플로우

### 워크플로우 1: 모델 학습부터 배포까지

1. **데이터 준비**
   ```bash
   # 정상 로그 데이터 수집 (최소 1000개 이상 권장)
   ```

2. **모델 학습**
   ```bash
   python scripts/train_isolation_forest.py \
       --data data/raw_logs/nginx_access.log \
       --output models/isolation_forest
   ```

3. **모델 검증**
   ```bash
   # 테스트 데이터로 검증
   python scripts/run_detector.py --model models/isolation_forest
   ```

4. **대시보드 배포**
   ```bash
   streamlit run app/dashboard.py --server.port 8501
   ```

### 워크플로우 2: 실시간 모니터링

1. **로그 수집 시작**
   ```bash
   python scripts/run_collector.py --mode realtime
   ```

2. **대시보드 실행**
   ```bash
   streamlit run app/dashboard.py
   ```

3. **모니터링**
   - 대시보드에서 실시간 로그 확인
   - 이상 탐지 알림 모니터링
   - 그래프로 트렌드 분석

## 설정 커스터마이징

### 로그 포맷 변경

Nginx 로그 포맷이 다른 경우 `src/collector/nginx_parser.py`의 `LOG_PATTERN`을 수정하세요.

### 특징 추가

`src/preprocessing/feature_engineering.py`의 `_extract_window_features` 메서드에 새로운 특징을 추가할 수 있습니다.

### 알림 임계값 조정

`configs/config_alert.yaml`에서 알림 조건을 조정하세요.

## 문제 해결

### 모델이 이상을 제대로 탐지하지 못하는 경우

1. 학습 데이터가 충분한지 확인 (최소 1000개 이상)
2. 특징 추출이 제대로 되는지 확인
3. 임계값 조정 (`configs/config_model.yaml`)

### 로그 파싱 오류

1. 로그 포맷이 표준 Nginx combined format인지 확인
2. `src/collector/nginx_parser.py`의 패턴 확인

### 대시보드가 업데이트되지 않는 경우

1. Streamlit 자동 새로고침 확인
2. 로그 수집이 실행 중인지 확인
