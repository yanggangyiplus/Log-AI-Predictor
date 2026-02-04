# 환경 변수 설정 가이드

## 개요

프로젝트에서 민감한 정보(비밀번호, API 키 등)는 환경 변수를 통해 관리합니다. 이를 통해 설정 파일에 비밀번호를 하드코딩하지 않고 안전하게 관리할 수 있습니다.

## 설정 방법

### 1. .env 파일 생성

프로젝트 루트 디렉토리에 `.env` 파일을 생성하세요:

```bash
cp .env.example .env
```

### 2. .env 파일 편집

`.env` 파일을 열고 실제 값으로 수정하세요:

```env
# Email 알림 설정
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password-here
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_TO_EMAILS=admin@example.com,team@example.com

# 웹훅 URL (선택사항)
WEBHOOK_URL=https://your-webhook-url.com/alerts

# 데이터베이스 경로 (선택사항)
DATABASE_PATH=data/database/logs.db

# 로그 파일 경로 (선택사항)
LOG_FILE_PATH=data/raw_logs/nginx_access.log

# 모델 경로 (선택사항)
MODEL_PATH=models/isolation_forest
```

### 3. .env 파일 보안

⚠️ **중요**: `.env` 파일은 절대 Git에 커밋하지 마세요!

- `.gitignore`에 이미 `.env`가 포함되어 있습니다
- `.env.example` 파일만 커밋하여 다른 개발자들이 참고할 수 있도록 합니다

## 환경 변수 목록

### Email 설정

| 변수명 | 설명 | 필수 | 기본값 |
|--------|------|------|--------|
| `SMTP_SERVER` | SMTP 서버 주소 | 아니오 | `smtp.gmail.com` |
| `SMTP_PORT` | SMTP 포트 번호 | 아니오 | `587` |
| `SMTP_USER` | SMTP 인증 이메일 | 예 (Email 사용 시) | - |
| `SMTP_PASSWORD` | SMTP 비밀번호/앱 비밀번호 | 예 (Email 사용 시) | - |
| `SMTP_FROM_EMAIL` | 발신자 이메일 | 아니오 | `SMTP_USER` 값 사용 |
| `SMTP_TO_EMAILS` | 수신자 이메일 (쉼표로 구분) | 예 (Email 사용 시) | - |

### 기타 설정

| 변수명 | 설명 | 필수 | 기본값 |
|--------|------|------|--------|
| `WEBHOOK_URL` | 웹훅 URL | 아니오 | - |
| `DATABASE_PATH` | 데이터베이스 파일 경로 | 아니오 | `data/database/logs.db` |
| `LOG_FILE_PATH` | 로그 파일 경로 | 아니오 | `data/raw_logs/nginx_access.log` |
| `MODEL_PATH` | 모델 파일 경로 | 아니오 | `models/isolation_forest` |

## 사용 예제

### Python 코드에서 사용

```python
from app.utils.env_config import get_email_config, get_database_path

# Email 설정 가져오기
email_config = get_email_config()
print(email_config['smtp_server'])
print(email_config['smtp_user'])

# 데이터베이스 경로 가져오기
db_path = get_database_path()
```

### 설정 파일과의 우선순위

환경 변수와 설정 파일(`configs/config_alert.yaml`)이 모두 있는 경우:

1. **환경 변수가 우선**: 환경 변수가 설정되어 있으면 환경 변수 사용
2. **설정 파일은 기본값**: 환경 변수가 없으면 설정 파일의 값 사용

이를 통해:
- 개발 환경: `.env` 파일 사용
- 프로덕션 환경: 시스템 환경 변수 사용 가능
- 설정 파일: 기본값 및 문서 역할

## Gmail 설정 예제

```env
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=abcd efgh ijkl mnop  # Gmail 앱 비밀번호 (16자리)
SMTP_FROM_EMAIL=your-email@gmail.com
SMTP_TO_EMAILS=admin@example.com,devops@example.com
```

## 프로덕션 환경

프로덕션 환경에서는 시스템 환경 변수로 설정할 수 있습니다:

```bash
# Linux/Mac
export SMTP_USER="your-email@gmail.com"
export SMTP_PASSWORD="your-password"

# 또는 Docker
docker run -e SMTP_USER="..." -e SMTP_PASSWORD="..." ...
```

## 문제 해결

### 환경 변수가 로드되지 않는 경우

1. `.env` 파일이 프로젝트 루트에 있는지 확인
2. `python-dotenv` 패키지가 설치되어 있는지 확인:
   ```bash
   pip install python-dotenv
   ```
3. 애플리케이션 시작 시 환경 변수 로드 확인:
   ```python
   from app.utils.env_config import get_email_config
   print(get_email_config())  # 설정 확인
   ```

### 비밀번호가 노출된 경우

1. 즉시 비밀번호 변경
2. `.env` 파일 확인 및 수정
3. Git 히스토리 확인 (실수로 커밋된 경우):
   ```bash
   git log --all --full-history -- .env
   ```
