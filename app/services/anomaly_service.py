"""
이상 탐지 서비스
이상 탐지 및 예측 비즈니스 로직
"""
import sys
from pathlib import Path
from typing import Tuple, List
import numpy as np
import pandas as pd
import logging

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.constants import SESSION_KEY_DETECTOR

logger = logging.getLogger(__name__)


class AnomalyService:
    """이상 탐지 서비스 클래스"""
    
    def __init__(self, session_state):
        """
        초기화
        
        Args:
            session_state: Streamlit session_state 객체
        """
        self.session_state = session_state
    
    def predict(self, features_df: pd.DataFrame) -> Tuple[List[float], List[bool]]:
        """
        이상 탐지 예측
        
        Args:
            features_df: 특징 DataFrame
            
        Returns:
            (점수 리스트, 이상 여부 리스트)
        """
        detector = self.session_state.get(SESSION_KEY_DETECTOR)
        if not detector:
            return [], []
        
        if features_df.empty:
            return [], []
        
        try:
            # 숫자형 컬럼만 선택
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            numeric_cols = [col for col in numeric_cols if col != 'timestamp']
            
            if len(numeric_cols) == 0:
                return [], []
            
            X = features_df[numeric_cols].values
            
            # 모델이 학습되었는지 확인
            if hasattr(detector, 'detector') and hasattr(detector.detector, 'is_trained'):
                if not detector.detector.is_trained:
                    logger.warning("모델이 학습되지 않았습니다.")
                    return [], []
            
            # 이상 탐지 실행
            anomaly_scores, is_anomaly = detector.predict(X)
            
            # 리스트로 변환
            if isinstance(anomaly_scores, np.ndarray):
                anomaly_scores = anomaly_scores.tolist()
            if isinstance(is_anomaly, np.ndarray):
                is_anomaly = is_anomaly.tolist()
            
            return anomaly_scores, is_anomaly
        except Exception as e:
            logger.error(f"이상 탐지 실패: {e}", exc_info=True)
            return [], []
    
    def get_anomaly_summary(self, anomaly_scores: List[float], 
                           is_anomaly: List[bool]) -> dict:
        """
        이상 탐지 결과 요약
        
        Args:
            anomaly_scores: 점수 리스트
            is_anomaly: 이상 여부 리스트
            
        Returns:
            요약 딕셔너리
        """
        if not anomaly_scores or not is_anomaly:
            return {
                'count': 0,
                'total': 0,
                'percentage': 0.0,
                'has_anomaly': False
            }
        
        anomaly_count = sum(is_anomaly)
        total_count = len(is_anomaly)
        anomaly_percentage = (anomaly_count / total_count * 100) if total_count > 0 else 0
        
        return {
            'count': anomaly_count,
            'total': total_count,
            'percentage': anomaly_percentage,
            'has_anomaly': anomaly_count > 0
        }
    
    def get_recent_anomaly_score(self, anomaly_scores: List[float], 
                                 is_anomaly: List[bool]) -> dict:
        """
        최근 이상 탐지 점수 반환
        
        Args:
            anomaly_scores: 점수 리스트
            is_anomaly: 이상 여부 리스트
            
        Returns:
            최근 점수 정보 딕셔너리
        """
        if not anomaly_scores or not is_anomaly:
            return {
                'score': 0.0,
                'is_anomaly': False,
                'avg_score': 0.0
            }
        
        recent_score = float(anomaly_scores[-1])
        recent_is_anomaly = bool(is_anomaly[-1])
        avg_score = float(np.mean(anomaly_scores))
        
        return {
            'score': recent_score,
            'is_anomaly': recent_is_anomaly,
            'avg_score': avg_score
        }

