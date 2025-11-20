"""
AutoEncoder 기반 이상 탐지 모듈
정상 데이터만 학습하여 재구성 오류로 이상치 탐지
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Tuple, Optional, List
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class AutoEncoderAnomalyDetector:
    """AutoEncoder 기반 이상 탐지 클래스"""
    
    def __init__(self, input_dim: int = 50, encoding_dim: int = 16, 
                 hidden_layers: List[int] = [32, 24], activation: str = 'relu'):
        """
        초기화
        
        Args:
            input_dim: 입력 특징 차원
            encoding_dim: 인코딩 차원 (압축된 표현)
            hidden_layers: 히든 레이어 크기 리스트
            activation: 활성화 함수
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_layers = hidden_layers
        self.activation = activation
        self.model: Optional[keras.Model] = None
        self.threshold: float = 0.0
        self.is_trained = False
        
    def build_model(self):
        """AutoEncoder 모델 구축"""
        # 입력 레이어
        input_layer = layers.Input(shape=(self.input_dim,), name='input')
        
        # 인코더
        x = input_layer
        for i, hidden_size in enumerate(self.hidden_layers):
            x = layers.Dense(hidden_size, activation=self.activation, 
                           name=f'encoder_{i+1}')(x)
        
        # 인코딩 레이어 (압축된 표현)
        encoded = layers.Dense(self.encoding_dim, activation=self.activation, 
                              name='encoded')(x)
        
        # 디코더
        x = encoded
        for i, hidden_size in enumerate(reversed(self.hidden_layers)):
            x = layers.Dense(hidden_size, activation=self.activation, 
                           name=f'decoder_{i+1}')(x)
        
        # 출력 레이어 (재구성)
        decoded = layers.Dense(self.input_dim, activation='sigmoid', 
                             name='decoded')(x)
        
        # 모델 생성
        self.model = keras.Model(input_layer, decoded, name='autoencoder')
        
        # 컴파일
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        logger.info(f"AutoEncoder 모델 구축 완료: 입력={self.input_dim}, 인코딩={self.encoding_dim}")
    
    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32, 
              validation_split: float = 0.2, verbose: int = 1):
        """
        모델 학습 (정상 데이터만 사용)
        
        Args:
            X_train: 학습 데이터 (정상 데이터만)
            epochs: 에폭 수
            batch_size: 배치 크기
            validation_split: 검증 데이터 비율
            verbose: 출력 레벨
        """
        if self.model is None:
            self.build_model()
        
        # 데이터 정규화 (0-1 범위)
        self.data_min = X_train.min(axis=0)
        self.data_max = X_train.max(axis=0)
        X_normalized = (X_train - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        # 학습
        history = self.model.fit(
            X_normalized, X_normalized,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=verbose,
            shuffle=True
        )
        
        # 임계값 설정 (학습 데이터의 재구성 오류 기반)
        train_reconstruction = self.model.predict(X_normalized, verbose=0)
        train_errors = np.mean(np.square(X_normalized - train_reconstruction), axis=1)
        
        # 95 백분위수를 임계값으로 설정
        self.threshold = np.percentile(train_errors, 95)
        
        self.is_trained = True
        logger.info(f"모델 학습 완료. 임계값: {self.threshold:.4f}")
        
        return history
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 예측
        
        Args:
            X: 입력 데이터
            
        Returns:
            (재구성 오류 배열, 이상 여부 배열)
        """
        if not self.is_trained:
            raise ValueError("모델이 학습되지 않았습니다. train() 메서드를 먼저 호출하세요.")
        
        # 데이터 정규화
        X_normalized = (X - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        # 재구성
        reconstruction = self.model.predict(X_normalized, verbose=0)
        
        # 재구성 오류 계산 (MSE)
        reconstruction_errors = np.mean(np.square(X_normalized - reconstruction), axis=1)
        
        # 이상 여부 판단
        is_anomaly = reconstruction_errors > self.threshold
        
        return reconstruction_errors, is_anomaly
    
    def predict_single(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        단일 샘플 예측
        
        Args:
            x: 단일 샘플 (1D 배열)
            
        Returns:
            (재구성 오류, 이상 여부)
        """
        x = x.reshape(1, -1)
        errors, is_anomaly = self.predict(x)
        return errors[0], is_anomaly[0]
    
    def save_model(self, filepath: str):
        """
        모델 저장
        
        Args:
            filepath: 저장 경로 (확장자 없이)
        """
        if self.model is None:
            raise ValueError("저장할 모델이 없습니다.")
        
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # 모델 저장
        self.model.save(f"{filepath}_model.h5")
        
        # 메타데이터 저장
        metadata = {
            'input_dim': self.input_dim,
            'encoding_dim': self.encoding_dim,
            'hidden_layers': self.hidden_layers,
            'activation': self.activation,
            'threshold': self.threshold,
            'data_min': self.data_min,
            'data_max': self.data_max
        }
        
        with open(f"{filepath}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str):
        """
        모델 로드
        
        Args:
            filepath: 로드 경로 (확장자 없이)
        """
        # 모델 로드
        self.model = keras.models.load_model(f"{filepath}_model.h5")
        
        # 메타데이터 로드
        with open(f"{filepath}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_layers = metadata['hidden_layers']
        self.activation = metadata['activation']
        self.threshold = metadata['threshold']
        self.data_min = metadata['data_min']
        self.data_max = metadata['data_max']
        self.is_trained = True
        
        logger.info(f"모델 로드 완료: {filepath}")

