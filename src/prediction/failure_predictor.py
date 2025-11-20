"""
장애 예측 모듈
과거 장애 직전 패턴과 유사한 패턴을 찾아 장애 예측
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class FailurePredictor:
    """장애 예측 클래스"""
    
    def __init__(self, n_neighbors: int = 5, similarity_threshold: float = 0.8):
        """
        초기화
        
        Args:
            n_neighbors: KNN에서 사용할 이웃 수
            similarity_threshold: 유사도 임계값 (0-1)
        """
        self.n_neighbors = n_neighbors
        self.similarity_threshold = similarity_threshold
        self.failure_patterns: List[np.ndarray] = []
        self.knn_model: Optional[NearestNeighbors] = None
        self.is_trained = False
    
    def add_failure_pattern(self, pattern: np.ndarray, minutes_before_failure: int = 10):
        """
        장애 직전 패턴 추가
        
        Args:
            pattern: 장애 직전 특징 벡터 (시간 윈도우별 특징)
            minutes_before_failure: 장애 발생 몇 분 전 패턴인지
        """
        self.failure_patterns.append({
            'pattern': pattern,
            'minutes_before': minutes_before_failure
        })
        logger.info(f"장애 패턴 추가: {pattern.shape}")
    
    def train(self):
        """KNN 모델 학습"""
        if not self.failure_patterns:
            logger.warning("학습할 장애 패턴이 없습니다.")
            return
        
        # 모든 패턴을 하나의 배열로 변환
        # 각 패턴을 평탄화하여 사용
        patterns_flat = []
        for item in self.failure_patterns:
            pattern = item['pattern']
            if isinstance(pattern, pd.DataFrame):
                pattern = pattern.values
            if len(pattern.shape) > 1:
                # 여러 시간 윈도우를 평탄화
                pattern_flat = pattern.flatten()
            else:
                pattern_flat = pattern
            patterns_flat.append(pattern_flat)
        
        X_train = np.array(patterns_flat)
        
        # KNN 모델 학습
        self.knn_model = NearestNeighbors(
            n_neighbors=min(self.n_neighbors, len(self.failure_patterns)),
            metric='cosine'  # 코사인 유사도 사용
        )
        self.knn_model.fit(X_train)
        
        self.is_trained = True
        logger.info(f"KNN 모델 학습 완료: {len(self.failure_patterns)}개 패턴")
    
    def predict(self, current_pattern: np.ndarray) -> Tuple[float, bool]:
        """
        현재 패턴이 장애 직전 패턴과 유사한지 예측
        
        Args:
            current_pattern: 현재 시간 윈도우 특징 벡터
            
        Returns:
            (유사도, 장애 위험 여부)
        """
        if not self.is_trained:
            logger.warning("모델이 학습되지 않았습니다.")
            return 0.0, False
        
        # 패턴 평탄화
        if isinstance(current_pattern, pd.DataFrame):
            current_pattern = current_pattern.values
        if len(current_pattern.shape) > 1:
            current_flat = current_pattern.flatten()
        else:
            current_flat = current_pattern
        
        current_flat = current_flat.reshape(1, -1)
        
        # 가장 유사한 패턴 찾기
        distances, indices = self.knn_model.kneighbors(current_flat)
        
        # 코사인 거리를 유사도로 변환 (1 - distance)
        similarity = 1 - distances[0][0]
        
        # 임계값 이상이면 장애 위험
        is_risk = similarity >= self.similarity_threshold
        
        return similarity, is_risk
    
    def predict_batch(self, patterns: List[np.ndarray]) -> List[Tuple[float, bool]]:
        """
        여러 패턴에 대한 예측
        
        Args:
            patterns: 패턴 리스트
            
        Returns:
            (유사도, 위험 여부) 튜플 리스트
        """
        results = []
        for pattern in patterns:
            similarity, is_risk = self.predict(pattern)
            results.append((similarity, is_risk))
        return results
    
    def add_historical_failure(self, features_df: pd.DataFrame, failure_time: datetime, 
                               window_minutes: int = 10):
        """
        과거 장애 데이터에서 패턴 추출하여 추가
        
        Args:
            features_df: 특징 DataFrame (timestamp 인덱스)
            failure_time: 장애 발생 시간
            window_minutes: 장애 직전 몇 분간의 패턴을 사용할지
        """
        if 'timestamp' not in features_df.columns and features_df.index.name != 'timestamp':
            logger.error("타임스탬프 정보가 필요합니다.")
            return
        
        # 장애 직전 시간 윈도우
        start_time = failure_time - timedelta(minutes=window_minutes)
        
        # 해당 시간대의 특징 추출
        if features_df.index.name == 'timestamp' or isinstance(features_df.index, pd.DatetimeIndex):
            window_features = features_df[(features_df.index >= start_time) & 
                                        (features_df.index < failure_time)]
        else:
            window_features = features_df[
                (features_df['timestamp'] >= start_time) & 
                (features_df['timestamp'] < failure_time)
            ]
        
        if len(window_features) > 0:
            # 타임스탬프 제외하고 숫자형 특징만 사용
            numeric_cols = window_features.select_dtypes(include=[np.number]).columns
            pattern = window_features[numeric_cols].values
            self.add_failure_pattern(pattern, minutes_before_failure=window_minutes)
            logger.info(f"과거 장애 패턴 추가: {failure_time}, 윈도우 크기: {len(window_features)}")
        else:
            logger.warning(f"장애 시간 {failure_time} 주변에 데이터가 없습니다.")

