#!/usr/bin/env python3
"""
PyTorch 기반 AutoEncoder 모델 학습 스크립트
Mac에서 TensorFlow 대신 PyTorch 사용 (더 안정적)
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.feature_engineering import FeatureEngineer
from src.anomaly.autoencoder_pytorch import PyTorchAutoEncoderAnomalyDetector
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_logs_from_file(filepath: str) -> list:
    """로그 파일에서 데이터 로드"""
    logs = []
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        from src.collector.nginx_parser import NginxParser
        parser = NginxParser()
        logs = parser.parse_batch(lines)
        
        logger.info(f"로그 파일 로드 완료: {len(logs)}개")
    except Exception as e:
        logger.error(f"로그 파일 로드 실패: {e}")
    
    return logs


def main():
    parser = argparse.ArgumentParser(description='PyTorch AutoEncoder 모델 학습 (Mac 친화적)')
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw_logs/nginx_access.log',
        help='학습 데이터 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/pytorch_autoencoder',
        help='모델 저장 경로'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=20,
        help='학습 에폭 수 (기본값: 20)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='배치 크기 (기본값: 32)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.001,
        help='학습률 (기본값: 0.001)'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("PyTorch AutoEncoder 모델 학습 시작")
    logger.info("=" * 60)
    
    # PyTorch 설치 확인
    try:
        import torch
        logger.info(f" PyTorch 버전: {torch.__version__}")
    except ImportError:
        logger.error(" PyTorch가 설치되지 않았습니다.")
        logger.info("설치 방법: pip install torch")
        return
    
    # 데이터 로드
    logger.info("1단계: 데이터 로드 중...")
    logs = load_logs_from_file(args.data)
    
    if not logs:
        logger.error("학습 데이터가 없습니다.")
        return
    
    if len(logs) < 100:
        logger.warning(f"데이터가 너무 적습니다 ({len(logs)}개). 최소 100개 이상 권장합니다.")
    
    # 특징 추출
    logger.info("2단계: 특징 추출 중...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.extract_features(logs)
    
    if features_df.empty:
        logger.error("추출된 특징이 없습니다.")
        return
    
    logger.info(f"특징 추출 완료: {len(features_df)}개 윈도우")
    
    # 숫자형 특징만 선택
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'timestamp']
    
    X_train = features_df[numeric_cols].values
    
    logger.info(f"학습 데이터 형태: {X_train.shape}")
    
    if X_train.shape[0] < 5:
        logger.error(f"학습 데이터가 너무 적습니다 ({X_train.shape[0]}개). 더 많은 로그가 필요합니다.")
        return
    
    # 모델 초기화 및 학습
    logger.info("3단계: 모델 학습 시작...")
    logger.info(f"   - 에폭 수: {args.epochs}")
    logger.info(f"   - 배치 크기: {args.batch_size}")
    logger.info(f"   - 학습률: {args.learning_rate}")
    logger.info(f"   - 예상 시간: 약 {args.epochs * 2}초")
    
    try:
        detector = PyTorchAutoEncoderAnomalyDetector(
            input_dim=X_train.shape[1],
            encoding_dim=16,
            hidden_layers=[32, 24],
            activation='relu'
        )
        
        detector.train(
            X_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            verbose=1
        )
        
        # 모델 저장
        logger.info("4단계: 모델 저장 중...")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        detector.save_model(args.output)
        
        logger.info("=" * 60)
        logger.info(" PyTorch AutoEncoder 모델 학습 완료!")
        logger.info(f"   저장 위치: {args.output}_pytorch.pth")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"모델 학습 중 오류 발생: {e}")
        logger.exception(e)


if __name__ == '__main__':
    main()

