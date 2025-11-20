"""
이상 탐지 관리자
여러 이상 탐지 모델을 통합 관리
"""
import yaml
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
import logging
import os

# TensorFlow는 지연 로딩 (느린 초기화 방지)
# from .autoencoder import AutoEncoderAnomalyDetector
from .isolation_forest import IsolationForestAnomalyDetector

logger = logging.getLogger(__name__)


class AnomalyDetectorManager:
    """이상 탐지 관리자 클래스"""
    
    def __init__(self, config_path: str = "configs/config_model.yaml"):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.detector_type = self.config.get('anomaly_detector', 'autoencoder')
        self.detector: Optional[object] = None
        self._initialize_detector()
    
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('model', {})
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}
    
    def _initialize_detector(self):
        """탐지기 초기화"""
        if self.detector_type == 'autoencoder':
            # PyTorch AutoEncoder 사용 (Mac에서 더 안정적)
            try:
                from .autoencoder_pytorch import PyTorchAutoEncoderAnomalyDetector
                ae_config = self.config.get('autoencoder', {})
                self.detector = PyTorchAutoEncoderAnomalyDetector(
                    input_dim=ae_config.get('input_dim', 50),
                    encoding_dim=ae_config.get('encoding_dim', 16),
                    hidden_layers=ae_config.get('hidden_layers', [32, 24]),
                    activation=ae_config.get('activation', 'relu')
                )
                logger.info("PyTorch AutoEncoder 사용 (Mac 친화적)")
            except ImportError:
                # PyTorch가 없으면 TensorFlow 시도 (지연 로딩)
                logger.warning("PyTorch를 찾을 수 없습니다. TensorFlow를 시도합니다...")
                try:
                    from .autoencoder import AutoEncoderAnomalyDetector
                    ae_config = self.config.get('autoencoder', {})
                    self.detector = AutoEncoderAnomalyDetector(
                        input_dim=ae_config.get('input_dim', 50),
                        encoding_dim=ae_config.get('encoding_dim', 16),
                        hidden_layers=ae_config.get('hidden_layers', [32, 24]),
                        activation=ae_config.get('activation', 'relu')
                    )
                    logger.info("TensorFlow AutoEncoder 사용")
                except ImportError:
                    raise ImportError("PyTorch 또는 TensorFlow가 필요합니다. pip install torch 또는 pip install tensorflow")
        elif self.detector_type == 'pytorch_autoencoder':
            # 명시적으로 PyTorch AutoEncoder 사용
            from .autoencoder_pytorch import PyTorchAutoEncoderAnomalyDetector
            ae_config = self.config.get('autoencoder', {})
            self.detector = PyTorchAutoEncoderAnomalyDetector(
                input_dim=ae_config.get('input_dim', 50),
                encoding_dim=ae_config.get('encoding_dim', 16),
                hidden_layers=ae_config.get('hidden_layers', [32, 24]),
                activation=ae_config.get('activation', 'relu')
            )
        elif self.detector_type == 'isolation_forest':
            if_config = self.config.get('isolation_forest', {})
            self.detector = IsolationForestAnomalyDetector(
                n_estimators=if_config.get('n_estimators', 100),
                contamination=if_config.get('contamination', 0.1),
                random_state=if_config.get('random_state', 42)
            )
        else:
            raise ValueError(f"지원하지 않는 탐지기 타입: {self.detector_type}")
        
        logger.info(f"이상 탐지기 초기화 완료: {self.detector_type}")
    
    def train(self, X_train: np.ndarray, **kwargs):
        """
        모델 학습
        
        Args:
            X_train: 학습 데이터
            **kwargs: 모델별 추가 파라미터
        """
        if self.detector_type == 'autoencoder':
            ae_config = self.config.get('autoencoder', {})
            self.detector.train(
                X_train,
                epochs=kwargs.get('epochs', ae_config.get('epochs', 50)),
                batch_size=kwargs.get('batch_size', ae_config.get('batch_size', 32)),
                validation_split=kwargs.get('validation_split', ae_config.get('validation_split', 0.2))
            )
        else:
            self.detector.train(X_train)
        
        logger.info("모델 학습 완료")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        이상 탐지 예측
        
        Args:
            X: 입력 데이터
            
        Returns:
            (점수 배열, 이상 여부 배열)
        """
        return self.detector.predict(X)
    
    def predict_single(self, x: np.ndarray) -> Tuple[float, bool]:
        """
        단일 샘플 예측
        
        Args:
            x: 단일 샘플
            
        Returns:
            (점수, 이상 여부)
        """
        return self.detector.predict_single(x)
    
    def save_model(self, filepath: str):
        """
        모델 저장
        
        Args:
            filepath: 저장 경로
        """
        self.detector.save_model(filepath)
    
    def load_model(self, filepath: str):
        """
        모델 로드
        
        Args:
            filepath: 로드 경로
        """
        # 파일 경로로 모델 타입 자동 감지
        if os.path.exists(f"{filepath}_pytorch.pth"):
            # PyTorch AutoEncoder 모델
            if self.detector_type != 'pytorch_autoencoder':
                from .autoencoder_pytorch import PyTorchAutoEncoderAnomalyDetector
                ae_config = self.config.get('autoencoder', {})
                self.detector = PyTorchAutoEncoderAnomalyDetector(
                    input_dim=ae_config.get('input_dim', 50),
                    encoding_dim=ae_config.get('encoding_dim', 16),
                    hidden_layers=ae_config.get('hidden_layers', [32, 24]),
                    activation=ae_config.get('activation', 'relu')
                )
                self.detector_type = 'pytorch_autoencoder'
        elif os.path.exists(filepath) and 'isolation' in filepath.lower():
            # Isolation Forest 모델
            if self.detector_type != 'isolation_forest':
                if_config = self.config.get('isolation_forest', {})
                self.detector = IsolationForestAnomalyDetector(
                    n_estimators=if_config.get('n_estimators', 100),
                    contamination=if_config.get('contamination', 0.1),
                    random_state=if_config.get('random_state', 42)
                )
                self.detector_type = 'isolation_forest'
        elif os.path.exists(f"{filepath}_model.h5"):
            # TensorFlow AutoEncoder 모델
            if self.detector_type != 'autoencoder':
                from .autoencoder import AutoEncoderAnomalyDetector
                ae_config = self.config.get('autoencoder', {})
                self.detector = AutoEncoderAnomalyDetector(
                    input_dim=ae_config.get('input_dim', 50),
                    encoding_dim=ae_config.get('encoding_dim', 16),
                    hidden_layers=ae_config.get('hidden_layers', [32, 24]),
                    activation=ae_config.get('activation', 'relu')
                )
                self.detector_type = 'autoencoder'
        
        # 모델 파일 존재 확인
        if not os.path.exists(filepath):
            # PyTorch나 TensorFlow 모델인지 확인
            if not os.path.exists(f"{filepath}_pytorch.pth") and not os.path.exists(f"{filepath}_model.h5"):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")
        
        self.detector.load_model(filepath)
    
    def _get_pytorch_detector(self):
        """PyTorch detector 클래스 반환 (타입 체크용)"""
        try:
            from .autoencoder_pytorch import PyTorchAutoEncoderAnomalyDetector
            return PyTorchAutoEncoderAnomalyDetector
        except ImportError:
            return None

