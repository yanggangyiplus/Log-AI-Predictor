# REST API 문서

Log Pattern Analyzer & Anomaly Predictor의 REST API 사용 가이드

## 기본 정보

- **Base URL**: `http://localhost:5000`
- **Content-Type**: `application/json`

## 엔드포인트 목록

### 1. 헬스 체크

서비스 상태 확인

**요청**
```http
GET /health
```

**응답**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00",
  "modules_loaded": true,
  "model_loaded": true
}
```

---

### 2. 이상 탐지 (배치)

여러 특징 벡터에 대한 이상 탐지

**요청**
```http
POST /api/anomaly/detect
Content-Type: application/json

{
  "features": [
    [0.5, 0.3, 0.2, 0.1, ...],
    [0.6, 0.4, 0.3, 0.2, ...]
  ]
}
```

**응답**
```json
{
  "scores": [0.75, 0.82],
  "is_anomaly": [true, true],
  "timestamp": "2024-01-01T00:00:00"
}
```

**에러 응답**
```json
{
  "error": "모델이 로드되지 않았습니다. 먼저 모델을 학습하세요."
}
```

---

### 3. 이상 탐지 (단일)

단일 특징 벡터에 대한 이상 탐지

**요청**
```http
POST /api/anomaly/detect/single
Content-Type: application/json

{
  "feature": [0.5, 0.3, 0.2, 0.1, ...]
}
```

**응답**
```json
{
  "score": 0.75,
  "is_anomaly": true,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

### 4. 로그 수집 및 파싱

로그 라인을 파싱하여 구조화된 데이터로 변환

**요청**
```http
POST /api/logs
Content-Type: application/json

{
  "logs": [
    "192.168.1.1 - - [01/Jan/2024:00:00:00 +0000] \"GET /api/users HTTP/1.1\" 200 1234",
    "192.168.1.2 - - [01/Jan/2024:00:00:01 +0000] \"POST /api/login HTTP/1.1\" 200 5678"
  ],
  "log_type": "nginx"
}
```

**응답**
```json
{
  "processed": 2,
  "parsed": 2,
  "failed": 0,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

### 5. 특징 추출

파싱된 로그에서 특징 추출

**요청**
```http
POST /api/logs/features
Content-Type: application/json

{
  "logs": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "ip": "192.168.1.1",
      "method": "GET",
      "url_path": "/api/users",
      "status_code": 200,
      "response_time": 0.123
    }
  ],
  "window_seconds": 60
}
```

**응답**
```json
{
  "features": [
    {
      "timestamp": "2024-01-01T00:00:00",
      "rps": 10.5,
      "error_rate_5xx": 0.0,
      "error_rate_4xx": 2.5,
      "avg_response_time": 123.45,
      "unique_ips": 5,
      "unique_paths": 3
    }
  ],
  "window_count": 1,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

### 6. 알림 조회

저장된 알림 목록 조회

**요청**
```http
GET /api/alerts?limit=10
```

**응답**
```json
{
  "alerts": [
    {
      "id": 1,
      "type": "anomaly",
      "message": "이상 패턴이 감지되었습니다.",
      "severity": "high",
      "timestamp": "2024-01-01T00:00:00"
    }
  ],
  "count": 1
}
```

---

### 7. 알림 생성

새로운 알림 생성

**요청**
```http
POST /api/alerts
Content-Type: application/json

{
  "type": "anomaly",
  "message": "이상 패턴이 감지되었습니다.",
  "severity": "high"
}
```

**응답**
```json
{
  "id": 1,
  "timestamp": "2024-01-01T00:00:00"
}
```

---

### 8. 모델 상태 조회

현재 로드된 모델 정보 조회

**요청**
```http
GET /api/model/status
```

**응답**
```json
{
  "loaded": true,
  "model_type": "isolation_forest",
  "model_path": "models/isolation_forest"
}
```

---

## 사용 예제

### Python 예제

```python
import requests
import json

BASE_URL = "http://localhost:5000"

# 1. 헬스 체크
response = requests.get(f"{BASE_URL}/health")
print(response.json())

# 2. 로그 파싱
logs = [
    "192.168.1.1 - - [01/Jan/2024:00:00:00 +0000] \"GET /api/users HTTP/1.1\" 200 1234"
]
response = requests.post(
    f"{BASE_URL}/api/logs",
    json={"logs": logs}
)
parsed_logs = response.json()

# 3. 특징 추출
response = requests.post(
    f"{BASE_URL}/api/logs/features",
    json={
        "logs": parsed_logs,
        "window_seconds": 60
    }
)
features = response.json()

# 4. 이상 탐지
feature_vectors = [f["features"][0] for f in features["features"]]
response = requests.post(
    f"{BASE_URL}/api/anomaly/detect",
    json={"features": feature_vectors}
)
anomaly_results = response.json()
print(f"이상 탐지 결과: {anomaly_results}")
```

### cURL 예제

```bash
# 헬스 체크
curl http://localhost:5000/health

# 이상 탐지
curl -X POST http://localhost:5000/api/anomaly/detect \
  -H "Content-Type: application/json" \
  -d '{
    "features": [[0.5, 0.3, 0.2, 0.1]]
  }'

# 알림 조회
curl http://localhost:5000/api/alerts?limit=10
```

---

## 에러 코드

- `400`: 잘못된 요청 (필수 필드 누락 등)
- `500`: 서버 내부 오류

---

## 주의사항

1. 모델이 로드되지 않은 상태에서는 이상 탐지 API가 동작하지 않습니다.
2. 특징 벡터의 차원은 학습된 모델의 입력 차원과 일치해야 합니다.
3. 대량의 로그를 처리할 때는 배치 단위로 나누어 요청하는 것을 권장합니다.
