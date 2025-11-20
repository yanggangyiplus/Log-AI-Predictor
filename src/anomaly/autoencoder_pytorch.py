"""
PyTorch 기반 AutoEncoder 이상 탐지 모듈
TensorFlow 대신 PyTorch 사용 (Mac에서 더 안정적)
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, Optional, List
import logging
import os
import pickle

logger = logging.getLogger(__name__)


class PyTorchAutoEncoder(nn.Module):
    """PyTorch AutoEncoder 네트워크"""
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_layers: List[int], activation: str = 'relu'):
        super(PyTorchAutoEncoder, self).__init__()
        
        # 활성화 함수 선택
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()
        
        # 인코더
        encoder_layers = []
        prev_dim = input_dim
        for hidden_size in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_size))
            encoder_layers.append(self.activation)
            prev_dim = hidden_size
        
        encoder_layers.append(nn.Linear(prev_dim, encoding_dim))
        encoder_layers.append(self.activation)
        self.encoder = nn.Sequential(*encoder_layers)
        
        # 디코더
        decoder_layers = []
        prev_dim = encoding_dim
        for hidden_size in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_size))
            decoder_layers.append(self.activation)
            prev_dim = hidden_size
        
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        decoder_layers.append(nn.Sigmoid())  # 출력은 0-1 범위
        self.decoder = nn.Sequential(*decoder_layers)
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class PyTorchAutoEncoderAnomalyDetector:
    """PyTorch 기반 AutoEncoder 이상 탐지 클래스"""
    
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
        
        # 디바이스 설정 (CPU 사용, Mac에서 안정적)
        self.device = torch.device('cpu')
        
        self.model: Optional[PyTorchAutoEncoder] = None
        self.threshold: float = 0.0
        self.is_trained = False
        self.data_min = None
        self.data_max = None
        
    def build_model(self):
        """AutoEncoder 모델 구축"""
        self.model = PyTorchAutoEncoder(
            self.input_dim,
            self.encoding_dim,
            self.hidden_layers,
            self.activation
        ).to(self.device)
        
        logger.info(f"PyTorch AutoEncoder 모델 구축 완료: 입력={self.input_dim}, 인코딩={self.encoding_dim}")
    
    def train(self, X_train: np.ndarray, epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001, verbose: int = 1):
        """
        모델 학습 (정상 데이터만 사용)
        
        Args:
            X_train: 학습 데이터 (정상 데이터만)
            epochs: 에폭 수
            batch_size: 배치 크기
            learning_rate: 학습률
            verbose: 출력 레벨
        """
        if self.model is None:
            self.build_model()
        
        # 데이터 정규화 (0-1 범위)
        self.data_min = X_train.min(axis=0)
        self.data_max = X_train.max(axis=0)
        X_normalized = (X_train - self.data_min) / (self.data_max - self.data_min + 1e-8)
        
        # NumPy를 PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        # 옵티마이저 및 손실 함수
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        
        # 학습
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            # 배치로 학습
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                
                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / (len(X_tensor) // batch_size + 1)
            if verbose > 0 and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
        
        # 임계값 설정 (학습 데이터의 재구성 오류 기반)
        self.model.eval()
        with torch.no_grad():
            train_reconstruction = self.model(X_tensor)
            train_errors = ((X_tensor - train_reconstruction) ** 2).mean(dim=1).cpu().numpy()
        
        # 95 백분위수를 임계값으로 설정
        self.threshold = np.percentile(train_errors, 95)
        
        self.is_trained = True
        logger.info(f"모델 학습 완료. 임계값: {self.threshold:.4f}")
    
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
        
        # NumPy를 PyTorch 텐서로 변환
        X_tensor = torch.FloatTensor(X_normalized).to(self.device)
        
        # 재구성
        self.model.eval()
        with torch.no_grad():
            reconstruction = self.model(X_tensor)
            
            # 재구성 오류 계산 (MSE)
            reconstruction_errors = ((X_tensor - reconstruction) ** 2).mean(dim=1).cpu().numpy()
        
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
        torch.save(self.model.state_dict(), f"{filepath}_pytorch.pth")
        
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
        
        with open(f"{filepath}_pytorch_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"모델 저장 완료: {filepath}")
    
    def load_model(self, filepath: str):
        """
        모델 로드
        
        Args:
            filepath: 로드 경로 (확장자 없이)
        """
        # 메타데이터 로드
        with open(f"{filepath}_pytorch_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.input_dim = metadata['input_dim']
        self.encoding_dim = metadata['encoding_dim']
        self.hidden_layers = metadata['hidden_layers']
        self.activation = metadata['activation']
        self.threshold = metadata['threshold']
        self.data_min = metadata['data_min']
        self.data_max = metadata['data_max']
        
        # 모델 구축 및 가중치 로드
        self.build_model()
        self.model.load_state_dict(torch.load(f"{filepath}_pytorch.pth", map_location=self.device))
        self.model.eval()
        self.is_trained = True
        
        logger.info(f"모델 로드 완료: {filepath}")

