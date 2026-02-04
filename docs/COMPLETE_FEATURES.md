# 완성된 기능 목록

## ✅ 완료된 주요 기능

### 1. Flask REST API 완성
- ✅ 실제 모델과 서비스 연동 완료
- ✅ 이상 탐지 API (`/api/anomaly/detect`)
- ✅ 단일 샘플 이상 탐지 API (`/api/anomaly/detect/single`)
- ✅ 로그 파싱 API (`/api/logs`)
- ✅ 특징 추출 API (`/api/logs/features`)
- ✅ 알림 조회/생성 API (`/api/alerts`)
- ✅ 모델 상태 조회 API (`/api/model/status`)
- ✅ 헬스 체크 API (`/health`)

**사용 방법:**
```bash
python scripts/run_api.py
# 또는
python app/api/api.py
```

**API 문서:** `docs/API.md` 참조

---

### 2. 데이터베이스 연동 (SQLite)
- ✅ 로그 데이터 저장 및 조회
- ✅ 특징 데이터 저장 및 조회
- ✅ 이상 탐지 결과 저장 및 조회
- ✅ 알림 데이터 저장 및 조회
- ✅ 인덱스 최적화로 빠른 조회 성능
- ✅ 통계 조회 기능

**사용 방법:**
```python
from src.database.db_manager import DatabaseManager

db = DatabaseManager()
# 로그 저장
db.insert_log(log_data)
# 로그 조회
logs = db.get_logs(limit=1000)
```

**데이터베이스 위치:** `data/database/logs.db`

---

### 3. 외부 알림 연동
- ✅ Email 알림 지원 (SMTP)
- ✅ 일반 웹훅 알림 지원
- ✅ 심각도별 색상 구분 (low, medium, high)
- ✅ HTML 및 텍스트 이메일 형식 지원
- ✅ 상세 정보 포함 알림

**설정 방법:**
`configs/config_alert.yaml` 파일 수정:
```yaml
alert:
  channels:
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      smtp_user: "your-email@gmail.com"
      smtp_password: "your-app-password"
      from_email: "your-email@gmail.com"
      to_emails:
        - "admin@example.com"
    webhook:
      enabled: true
      url: "https://your-webhook-url.com/alerts"
```

**Gmail 설정 방법:**
1. Google 계정 설정 → 보안 → 2단계 인증 활성화
2. 앱 비밀번호 생성 (https://myaccount.google.com/apppasswords)
3. 생성된 앱 비밀번호를 `smtp_password`에 입력

**자동 알림:** 이상 탐지 시 자동으로 설정된 채널로 알림 전송

---

### 4. 모델 자동 재학습
- ✅ 데이터베이스 기반 재학습
- ✅ 로그 파일 기반 재학습
- ✅ 최근 N일간 데이터 사용 옵션
- ✅ Isolation Forest 및 PyTorch AutoEncoder 지원

**사용 방법:**
```bash
# 데이터베이스에서 최근 7일간 데이터로 재학습
python scripts/retrain_model.py --model-type isolation_forest --days-back 7 --use-database

# 로그 파일로 재학습
python scripts/retrain_model.py --model-type isolation_forest --log-file data/raw_logs/nginx_access.log
```

**크론잡 설정 예제:**
```bash
# 매주 일요일 새벽 3시에 자동 재학습
0 3 * * 0 cd /path/to/project && python scripts/retrain_model.py --use-database
```

---

### 5. API 문서화
- ✅ 상세한 REST API 문서 (`docs/API.md`)
- ✅ 모든 엔드포인트 설명
- ✅ 요청/응답 예제
- ✅ Python 및 cURL 사용 예제
- ✅ 에러 코드 설명

---

## 통합 사용 예제

### 전체 워크플로우

1. **로그 수집 및 저장**
```python
from src.collector.nginx_parser import NginxParser
from src.database.db_manager import DatabaseManager

parser = NginxParser()
db = DatabaseManager()

# 로그 파일 읽기
with open('logs/access.log', 'r') as f:
    logs = parser.parse_batch(f.readlines())

# 데이터베이스에 저장
db.insert_logs_batch(logs)
```

2. **특징 추출 및 이상 탐지**
```python
from src.preprocessing.feature_engineering import FeatureEngineer
from src.anomaly.detector_manager import AnomalyDetectorManager

# 특징 추출
feature_engineer = FeatureEngineer()
features_df = feature_engineer.extract_features(logs)

# 모델 로드
detector = AnomalyDetectorManager()
detector.load_model('models/isolation_forest')

# 이상 탐지
scores, is_anomaly = detector.predict(features_df.values)
```

3. **알림 전송**
```python
from app.services.notification_service import NotificationService

notification = NotificationService()
notification.send_slack_notification(
    message="이상 패턴이 감지되었습니다!",
    severity="high",
    details={"score": 0.85, "count": 5}
)
```

---

## 설정 파일

### 알림 설정 (`configs/config_alert.yaml`)
```yaml
alert:
  enabled: true
  conditions:
    error_rate_threshold: 5.0
    response_time_threshold: 1000
    reconstruction_error_threshold: 0.75
  channels:
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      smtp_user: "your-email@gmail.com"
      smtp_password: "your-app-password"
      from_email: "your-email@gmail.com"
      to_emails:
        - "admin@example.com"
    webhook:
      enabled: false
      url: ""
```

---

## 다음 단계

프로젝트는 이제 프로덕션 환경에서 사용할 수 있는 수준입니다!

1. **모델 학습**: `python scripts/train_isolation_forest.py`
2. **API 서버 실행**: `python scripts/run_api.py`
3. **대시보드 실행**: `streamlit run app/web/main.py`
4. **알림 설정**: `configs/config_alert.yaml` 수정
5. **자동 재학습 설정**: 크론잡 또는 스케줄러 설정

---

## 문제 해결

### 모델이 로드되지 않는 경우
```bash
# 모델 학습
python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log
```

### Slack 알림이 작동하지 않는 경우
1. `configs/config_alert.yaml`에서 `enabled: true` 확인
2. 웹훅 URL이 올바른지 확인
3. 로그 확인: `tail -f logs/app.log`

### 데이터베이스 오류
```bash
# 데이터베이스 파일 삭제 후 재생성
rm data/database/logs.db
python -c "from src.database.db_manager import DatabaseManager; DatabaseManager()"
```
