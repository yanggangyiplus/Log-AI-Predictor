# 빠른 시작 가이드

## 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 의존성 설치
pip install -r requirements.txt

# 환경 변수 설정 (Email 알림 사용 시 필수)
# 프로젝트 루트에 .env 파일 생성
cat > .env << EOF
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_TO_EMAILS=admin@example.com
EOF
# .env 파일을 편집하여 실제 값 입력
```

## 2. 모델 학습

```bash
# Isolation Forest 모델 학습 (빠름, 권장)
python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log

# 또는 PyTorch AutoEncoder 모델 학습 (더 정확)
python scripts/train_pytorch_autoencoder.py --data data/raw_logs/nginx_access.log --epochs 20
```

## 3. 서비스 실행

### 옵션 1: Streamlit 대시보드 (GUI)
```bash
streamlit run app/web/main.py
```
브라우저에서 `http://localhost:8501` 접속

### 옵션 2: Flask API 서버
```bash
python scripts/run_api.py
```
API 엔드포인트: `http://localhost:5000`

## 4. 알림 설정 (선택사항)

`configs/config_alert.yaml` 파일 수정:
```yaml
alert:
  channels:
    slack:
      enabled: true
      webhook_url: "YOUR_SLACK_WEBHOOK_URL"
```

## 5. 자동 재학습 설정 (선택사항)

크론잡 예제 (매주 일요일 새벽 3시):
```bash
0 3 * * 0 cd /path/to/project && python scripts/retrain_model.py --use-database
```

## 다음 단계

- [API 문서](API.md) 참조
- [완성된 기능 목록](COMPLETE_FEATURES.md) 확인
- [사용 가이드](USAGE.md) 읽기
