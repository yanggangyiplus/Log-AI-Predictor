"""
캐시 유틸리티
Streamlit 캐시 관리 함수
"""
import streamlit as st
from pathlib import Path
import yaml
import logging
from typing import Dict, Any, Optional

from .constants import CACHE_TTL_ALERT_CONFIG, CACHE_TTL_MODEL

logger = logging.getLogger(__name__)


@st.cache_data(ttl=CACHE_TTL_ALERT_CONFIG)
def load_alert_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """
    알림 설정 로드 (캐싱 적용)
    
    Args:
        config_path: 설정 파일 경로 (기본값: constants에서 가져옴)
        
    Returns:
        알림 설정 딕셔너리
    """
    if config_path is None:
        from .constants import CONFIG_ALERT_PATH
        config_path = CONFIG_ALERT_PATH
    
    try:
        if not config_path.exists():
            logger.warning("알림 설정 파일이 없습니다. 기본값을 사용합니다.")
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('alert', {})
    except Exception as e:
        logger.error(f"알림 설정 로드 실패: {e}")
        return {}


@st.cache_resource
def get_cached_model(model_path: str):
    """
    모델 캐싱 (리소스 캐시 사용)
    
    Args:
        model_path: 모델 경로
        
    Returns:
        캐시된 모델 객체
    """
    # 실제 모델 로딩은 service layer에서 처리
    # 여기서는 캐시 키만 관리
    return model_path

