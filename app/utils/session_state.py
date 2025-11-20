"""
세션 상태 관리 유틸리티
최소 정보만 저장하고 데이터는 캐시로 관리
"""
import streamlit as st
from typing import Any, Dict, Optional
from app.utils.constants import (
    SESSION_KEY_COLLECTOR,
    SESSION_KEY_DETECTOR,
    SESSION_KEY_CURRENT_MODEL,
    SESSION_KEY_COLLECTION_MODE,
    SESSION_KEY_DATA_UPDATED
)


class SessionStateManager:
    """세션 상태 관리 클래스"""
    
    def __init__(self, session_state):
        """
        초기화
        
        Args:
            session_state: Streamlit session_state 객체
        """
        self.session_state = session_state
        self._initialize_minimal_state()
    
    def _initialize_minimal_state(self):
        """최소 세션 상태만 초기화"""
        # 필수 상태만 초기화 (객체 참조)
        if SESSION_KEY_COLLECTOR not in self.session_state:
            self.session_state[SESSION_KEY_COLLECTOR] = None
        if SESSION_KEY_DETECTOR not in self.session_state:
            self.session_state[SESSION_KEY_DETECTOR] = None
        if SESSION_KEY_CURRENT_MODEL not in self.session_state:
            self.session_state[SESSION_KEY_CURRENT_MODEL] = None
        if SESSION_KEY_COLLECTION_MODE not in self.session_state:
            self.session_state[SESSION_KEY_COLLECTION_MODE] = None
        if SESSION_KEY_DATA_UPDATED not in self.session_state:
            self.session_state[SESSION_KEY_DATA_UPDATED] = False
    
    def set_collector(self, collector):
        """수집기 설정"""
        self.session_state[SESSION_KEY_COLLECTOR] = collector
    
    def get_collector(self):
        """수집기 가져오기"""
        return self.session_state.get(SESSION_KEY_COLLECTOR)
    
    def set_detector(self, detector):
        """모델 설정"""
        self.session_state[SESSION_KEY_DETECTOR] = detector
    
    def get_detector(self):
        """모델 가져오기"""
        return self.session_state.get(SESSION_KEY_DETECTOR)
    
    def set_current_model(self, model_path: str):
        """현재 모델 경로 설정"""
        self.session_state[SESSION_KEY_CURRENT_MODEL] = model_path
    
    def get_current_model(self) -> Optional[str]:
        """현재 모델 경로 가져오기"""
        return self.session_state.get(SESSION_KEY_CURRENT_MODEL)
    
    def set_collection_mode(self, mode: str):
        """수집 모드 설정"""
        self.session_state[SESSION_KEY_COLLECTION_MODE] = mode
    
    def get_collection_mode(self) -> Optional[str]:
        """수집 모드 가져오기"""
        return self.session_state.get(SESSION_KEY_COLLECTION_MODE)
    
    def mark_data_updated(self):
        """데이터 업데이트 플래그 설정"""
        self.session_state[SESSION_KEY_DATA_UPDATED] = True
    
    def clear_data_updated(self):
        """데이터 업데이트 플래그 초기화"""
        self.session_state[SESSION_KEY_DATA_UPDATED] = False
    
    def is_data_updated(self) -> bool:
        """데이터 업데이트 여부 확인"""
        return self.session_state.get(SESSION_KEY_DATA_UPDATED, False)
    
    def cleanup(self):
        """세션 상태 정리 (메모리 관리)"""
        # 큰 데이터는 제거하고 최소 정보만 유지
        keys_to_remove = [
            'logs_data',
            'features_data',
            'anomaly_results',
            'alerts'
        ]
        
        for key in keys_to_remove:
            if key in self.session_state:
                del self.session_state[key]

