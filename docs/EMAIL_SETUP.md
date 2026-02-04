# Email 알림 설정 가이드

## 개요

Log Pattern Analyzer는 SMTP를 통한 Email 알림을 지원합니다. 이상 탐지 시 자동으로 설정된 이메일 주소로 알림을 전송합니다.

## 설정 방법

### 1. 설정 파일 수정

`configs/config_alert.yaml` 파일을 열고 Email 설정을 추가합니다:

```yaml
alert:
  enabled: true
  channels:
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"  # SMTP 서버 주소
      smtp_port: 587                 # 포트 번호 (TLS: 587, SSL: 465)
      smtp_user: "your-email@gmail.com"  # SMTP 인증 이메일
      smtp_password: "your-app-password"  # 앱 비밀번호
      from_email: "your-email@gmail.com"  # 발신자 이메일
      to_emails:                       # 수신자 이메일 리스트
        - "admin@example.com"
        - "team@example.com"
```

### 2. 주요 이메일 서비스 설정

#### Gmail 설정

1. **Google 계정 설정**
   - Google 계정에 로그인
   - [보안 설정](https://myaccount.google.com/security)으로 이동

2. **2단계 인증 활성화**
   - "2단계 인증" 활성화 (필수)

3. **앱 비밀번호 생성**
   - [앱 비밀번호 페이지](https://myaccount.google.com/apppasswords)로 이동
   - "앱 선택" → "기타(맞춤 이름)" → 이름 입력 (예: "Log Analyzer")
   - "생성" 클릭
   - 생성된 16자리 비밀번호를 복사

4. **설정 파일에 입력**
   ```yaml
   smtp_server: "smtp.gmail.com"
   smtp_port: 587
   smtp_user: "your-email@gmail.com"
   smtp_password: "생성된-16자리-앱-비밀번호"
   from_email: "your-email@gmail.com"
   ```

#### Outlook/Hotmail 설정

```yaml
smtp_server: "smtp-mail.outlook.com"
smtp_port: 587
smtp_user: "your-email@outlook.com"
smtp_password: "your-password"
from_email: "your-email@outlook.com"
```

#### 네이버 메일 설정

```yaml
smtp_server: "smtp.naver.com"
smtp_port: 587
smtp_user: "your-email@naver.com"
smtp_password: "your-password"
from_email: "your-email@naver.com"
```

#### 기업용 이메일 (Exchange/Office 365)

```yaml
smtp_server: "smtp.office365.com"
smtp_port: 587
smtp_user: "your-email@company.com"
smtp_password: "your-password"
from_email: "your-email@company.com"
```

### 3. 테스트

설정이 완료되면 테스트해보세요:

```python
from app.services.notification_service import NotificationService

notification = NotificationService()
success = notification.send_email_notification(
    message="테스트 알림입니다.",
    severity="low",
    details={"test": "true"}
)

if success:
    print("이메일 전송 성공!")
else:
    print("이메일 전송 실패. 설정을 확인하세요.")
```

## 이메일 형식

### HTML 형식

이메일은 HTML 형식으로 전송되며, 심각도에 따라 색상이 구분됩니다:

- **높음 (High)**: 빨간색 배경
- **보통 (Medium)**: 주황색 배경
- **낮음 (Low)**: 초록색 배경

### 포함되는 정보

- 알림 제목 (심각도 포함)
- 발생 시간
- 알림 메시지
- 상세 정보 (이상 점수, 탐지 개수 등)

## 문제 해결

### 인증 실패 오류

```
SMTPAuthenticationError: (535, '5.7.8 Username and Password not accepted')
```

**해결 방법:**
1. Gmail의 경우 앱 비밀번호를 사용해야 합니다 (일반 비밀번호 사용 불가)
2. 2단계 인증이 활성화되어 있는지 확인
3. 사용자 이름과 비밀번호가 정확한지 확인

### 연결 시간 초과

```
SMTPException: Connection timed out
```

**해결 방법:**
1. 방화벽에서 SMTP 포트(587 또는 465)가 열려있는지 확인
2. 회사 네트워크의 경우 IT 부서에 문의
3. SMTP 서버 주소가 올바른지 확인

### SSL/TLS 오류

```
SSLError: [SSL: CERTIFICATE_VERIFY_FAILED]
```

**해결 방법:**
1. 포트 번호 확인 (TLS: 587, SSL: 465)
2. `server.starttls()` 사용 여부 확인 (코드에 이미 포함됨)

### 이메일이 스팸으로 분류되는 경우

**해결 방법:**
1. 발신자 이메일 주소를 신뢰할 수 있는 도메인으로 설정
2. SPF 레코드 설정 (도메인 소유 시)
3. 수신자 이메일의 스팸 필터 설정 확인

## 보안 고려사항

1. **비밀번호 보안**
   - 설정 파일에 평문 비밀번호 저장 시 파일 권한 설정 (`chmod 600`)
   - 환경 변수 사용 권장 (향후 개선 예정)

2. **앱 비밀번호 사용**
   - Gmail 등은 앱 비밀번호 사용 권장
   - 일반 비밀번호 노출 시 즉시 변경

3. **네트워크 보안**
   - 가능하면 VPN 또는 보안 네트워크 사용
   - 공용 Wi-Fi에서 SMTP 사용 시 주의

## 예제 설정 파일

완전한 예제 설정:

```yaml
alert:
  enabled: true
  conditions:
    error_rate_threshold: 5.0
    response_time_threshold: 1000
    reconstruction_error_threshold: 0.75
  channels:
    streamlit_popup: true
    log_highlight: true
    email:
      enabled: true
      smtp_server: "smtp.gmail.com"
      smtp_port: 587
      smtp_user: "monitoring@example.com"
      smtp_password: "app-password-here"
      from_email: "monitoring@example.com"
      to_emails:
        - "admin@example.com"
        - "devops@example.com"
        - "oncall@example.com"
    webhook:
      enabled: false
      url: ""
```
