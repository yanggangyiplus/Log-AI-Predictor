"""
특징 추출 서비스
특징 추출 및 관리 비즈니스 로직
"""
import sys
from pathlib import Path
from typing import List, Dict
import pandas as pd
import logging

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.preprocessing.feature_engineering import FeatureEngineer
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    logging.error(f"모듈 로드 실패: {e}")

from app.utils.constants import (
    FEATURE_WINDOW_SIZE,
    FEATURE_RECENT_LOGS,
    MAX_FEATURES_IN_MEMORY,
    SESSION_KEY_FEATURE_ENGINEER,
    SESSION_KEY_FEATURES_DATA
)

logger = logging.getLogger(__name__)


class FeatureService:
    """특징 추출 서비스 클래스"""
    
    def __init__(self, session_state):
        """
        초기화
        
        Args:
            session_state: Streamlit session_state 객체
        """
        self.session_state = session_state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if MODULES_LOADED:
            if SESSION_KEY_FEATURE_ENGINEER not in self.session_state:
                self.session_state[SESSION_KEY_FEATURE_ENGINEER] = FeatureEngineer()
            if SESSION_KEY_FEATURES_DATA not in self.session_state:
                self.session_state[SESSION_KEY_FEATURES_DATA] = pd.DataFrame()
    
    def extract_features(self, logs: List[Dict], window_seconds: int = None) -> pd.DataFrame:
        """
        로그에서 특징 추출
        
        Args:
            logs: 로그 리스트
            window_seconds: 시간 윈도우 크기 (초)
            
        Returns:
            특징 DataFrame
        """
        if not MODULES_LOADED:
            return pd.DataFrame()
        
        if not logs:
            return pd.DataFrame()
        
        if window_seconds is None:
            window_seconds = FEATURE_WINDOW_SIZE
        
        try:
            feature_engineer = self.session_state.get(SESSION_KEY_FEATURE_ENGINEER)
            if not feature_engineer:
                feature_engineer = FeatureEngineer()
                self.session_state[SESSION_KEY_FEATURE_ENGINEER] = feature_engineer
            
            # 최근 N개 로그만 사용 (성능 최적화)
            recent_logs = logs[-FEATURE_RECENT_LOGS:] if len(logs) > FEATURE_RECENT_LOGS else logs
            
            features_df = feature_engineer.extract_features(recent_logs, window_seconds)
            
            if not features_df.empty:
                # 기존 데이터와 병합
                existing_features = self.session_state.get(SESSION_KEY_FEATURES_DATA, pd.DataFrame())
                
                if existing_features.empty:
                    self.session_state[SESSION_KEY_FEATURES_DATA] = features_df.copy()
                else:
                    # 중복 제거 후 병합
                    combined = pd.concat([existing_features, features_df], ignore_index=True)
                    self.session_state[SESSION_KEY_FEATURES_DATA] = (
                        combined.drop_duplicates(subset=['timestamp'])
                        .sort_values('timestamp')
                        .reset_index(drop=True)
                    )
                
                # 메모리 관리: 최근 N개만 유지
                features_data = self.session_state[SESSION_KEY_FEATURES_DATA]
                if len(features_data) > MAX_FEATURES_IN_MEMORY:
                    self.session_state[SESSION_KEY_FEATURES_DATA] = (
                        features_data.tail(MAX_FEATURES_IN_MEMORY).reset_index(drop=True)
                    )
            
            return features_df
        except Exception as e:
            logger.error(f"특징 추출 실패: {e}", exc_info=True)
            return pd.DataFrame()
    
    def get_features(self) -> pd.DataFrame:
        """
        저장된 특징 데이터 반환
        
        Returns:
            특징 DataFrame
        """
        return self.session_state.get(SESSION_KEY_FEATURES_DATA, pd.DataFrame())
    
    def get_recent_features(self, count: int = 10) -> pd.DataFrame:
        """
        최근 특징 데이터 반환
        
        Args:
            count: 반환할 개수
            
        Returns:
            특징 DataFrame
        """
        features = self.get_features()
        if features.empty:
            return pd.DataFrame()
        return features.tail(count) if len(features) > count else features

