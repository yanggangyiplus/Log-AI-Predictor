"""
상수 정의
애플리케이션 전역에서 사용하는 상수값들
"""
from pathlib import Path

# 프로젝트 루트 경로
PROJECT_ROOT = Path(__file__).parent.parent.parent

# 설정 파일 경로
CONFIG_COLLECT_PATH = PROJECT_ROOT / "configs" / "config_collect.yaml"
CONFIG_MODEL_PATH = PROJECT_ROOT / "configs" / "config_model.yaml"
CONFIG_ALERT_PATH = PROJECT_ROOT / "configs" / "config_alert.yaml"

# 모델 경로
MODEL_ISOLATION_FOREST = PROJECT_ROOT / "experiments" / "checkpoints" / "isolation_forest"
MODEL_PYTORCH_AUTOENCODER = PROJECT_ROOT / "experiments" / "checkpoints" / "pytorch_autoencoder"

# 데이터 경로
DATA_RAW_LOGS = PROJECT_ROOT / "data" / "raw_logs"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

# 세션 상태 키
SESSION_KEY_COLLECTOR = 'collector'
SESSION_KEY_DETECTOR = 'detector'
SESSION_KEY_FEATURE_ENGINEER = 'feature_engineer'
SESSION_KEY_FAILURE_PREDICTOR = 'failure_predictor'
SESSION_KEY_LOGS_DATA = 'logs_data'
SESSION_KEY_FEATURES_DATA = 'features_data'
SESSION_KEY_ANOMALY_RESULTS = 'anomaly_results'
SESSION_KEY_ALERTS = 'alerts'
SESSION_KEY_CURRENT_MODEL = 'current_model'
SESSION_KEY_COLLECTION_MODE = 'collection_mode'
SESSION_KEY_DATA_UPDATED = 'data_updated'

# 데이터 제한
MAX_LOGS_IN_MEMORY = 1000
MAX_FEATURES_IN_MEMORY = 100
MAX_ALERTS_IN_MEMORY = 50
RECENT_LOGS_DISPLAY = 20
RECENT_ALERTS_DISPLAY = 5

# 특징 추출 설정
FEATURE_WINDOW_SIZE = 60  # 초
FEATURE_RECENT_LOGS = 100  # 특징 추출 시 사용할 최근 로그 수

# 알림 임계값 (기본값)
DEFAULT_ERROR_RATE_THRESHOLD = 5.0  # %
DEFAULT_RESPONSE_TIME_THRESHOLD = 1000  # ms
DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD = 0.75

# 캐시 TTL (초)
CACHE_TTL_ALERT_CONFIG = 300  # 5분
CACHE_TTL_MODEL = 3600  # 1시간

