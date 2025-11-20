"""
Isolation Forest 기반 이상 탐지 모듈
트리 기반 이상 탐지 알고리즘
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from typing import Tuple, Optional
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class IsolationForestAnomalyDetector:
    """Isolation Forest 기반 이상 탐지 클래스"""
    
    def __init__(self, n_estimators: int = 100, contamination: float = 0.1, 
                 random_state: int = 42):
        """
        초기화
        
        Args:
            n_estimators: 트리 개수
            contamination: 이상치 비율 추정값
            random_state: 랜덤 시드
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.random_state = random_state
        self.model: Optional[IsolationForest] = None
        self.is_trained = False
    
    def train(self, X_train: np.ndarray):
        """
        모델 학습
        
        Args:
            X_train: 학습 데이터
        """
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train)
        self.is_trained = True
        
        logger.info(f"Isolation Forest 모델 학습 완료")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 예측
        
        Args:
            X: 입력 데이터
            
        Returns:
            (anomaly scores, 이상 여부 배열)
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        # 이상 점수 계산 (낮을수록 이상)
        anomaly_scores = self.model.decision_function(X)
        
        # 이상 여부 판단 (-1: 이상, 1: 정상)
        predictions = self.model.predict(X)
        is_anomaly = predictions == -1
        
        return anomaly_scores, is_anomaly
    
    def predict_single(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        단일 샘플 예측
        
        Args:
            x: 단일 샘플 (1D 배열)
            
        Returns:
            (anomaly score, 이상 여부)
        """
        x = x.reshape(1, -1)
        scores, is_anomaly = self.predict(x)
        return scores[0], is_anomaly[0]
    
    def save_model(self, filepath: str):
        """
        모델 저장
        
        Args:
            filepath: 저장 경로
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'n_estimators': self.n_estimators,
                'contamination': self.contamination,
                'random_state': self.random_state
            }, f)
        
        logger.info(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str):
        """
        모델 로드
        
        Args:
            filepath: 로드 경로
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.n_estimators = data['n_estimators']
        self.contamination = data['contamination']
        self.random_state = data['random_state']
        self.is_trained = True
        
        logger.info(f"모델 로드 완료: {filepath}")

