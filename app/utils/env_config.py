"""
환경 변수 설정 모듈
.env 파일에서 환경 변수를 로드하고 관리
"""
import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# .env 파일 경로
ENV_FILE = PROJECT_ROOT / ".env"

# 환경 변수 로드
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)
else:
    # .env 파일이 없으면 기본값 사용
    load_dotenv()


def get_env(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    환경 변수 가져오기
    
    Args:
        key: 환경 변수 키
        default: 기본값
        
    Returns:
        환경 변수 값 또는 기본값
    """
    return os.getenv(key, default)


def get_email_config() -> dict:
    """
    Email 설정 가져오기
    
    Returns:
        Email 설정 딕셔너리
    """
    return {
        'smtp_server': get_env('SMTP_SERVER', 'smtp.gmail.com'),
        'smtp_port': int(get_env('SMTP_PORT', '587')),
        'smtp_user': get_env('SMTP_USER', ''),
        'smtp_password': get_env('SMTP_PASSWORD', ''),
        'from_email': get_env('SMTP_FROM_EMAIL', get_env('SMTP_USER', '')),
        'to_emails': _parse_email_list(get_env('SMTP_TO_EMAILS', '')),
    }


def _parse_email_list(email_string: str) -> List[str]:
    """
    이메일 리스트 문자열을 리스트로 변환
    
    Args:
        email_string: 쉼표로 구분된 이메일 문자열
        
    Returns:
        이메일 리스트
    """
    if not email_string:
        return []
    
    # 쉼표로 분리하고 공백 제거
    emails = [email.strip() for email in email_string.split(',')]
    # 빈 문자열 제거
    return [email for email in emails if email]


def get_webhook_url() -> Optional[str]:
    """
    웹훅 URL 가져오기
    
    Returns:
        웹훅 URL 또는 None
    """
    return get_env('WEBHOOK_URL')


def get_database_path() -> str:
    """
    데이터베이스 경로 가져오기
    
    Returns:
        데이터베이스 파일 경로
    """
    return get_env('DATABASE_PATH', 'data/database/logs.db')


def get_log_file_path() -> str:
    """
    로그 파일 경로 가져오기
    
    Returns:
        로그 파일 경로
    """
    return get_env('LOG_FILE_PATH', 'data/raw_logs/nginx_access.log')


def get_model_path() -> str:
    """
    모델 경로 가져오기
    
    Returns:
        모델 경로
    """
    return get_env('MODEL_PATH', 'models/isolation_forest')
