#!/usr/bin/env python3
"""
모델 자동 재학습 스크립트
주기적으로 새로운 데이터로 모델을 재학습
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import logging

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.feature_engineering import FeatureEngineer
from src.anomaly.detector_manager import AnomalyDetectorManager
from src.database.db_manager import DatabaseManager
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config_model.yaml"):
    """설정 파일 로드"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('model', {})
    except Exception as e:
        logger.error(f"설정 파일 로드 실패: {e}")
        return {}


def retrain_model(model_type: str = "isolation_forest", 
                 days_back: int = 7,
                 use_database: bool = True,
                 log_file: str = None):
    """
    모델 재학습
    
    Args:
        model_type: 모델 타입 ("isolation_forest" 또는 "pytorch_autoencoder")
        days_back: 최근 며칠간의 데이터 사용
        use_database: 데이터베이스 사용 여부
        log_file: 로그 파일 경로 (데이터베이스 미사용 시)
    """
    logger.info(f"모델 재학습 시작: {model_type}")
    
    # 데이터 로드
    logs = []
    
    if use_database:
        logger.info("데이터베이스에서 데이터 로드 중...")
        db_manager = DatabaseManager()
        
        # 최근 N일간의 로그 조회
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        logs = db_manager.get_logs(start_time=start_time, end_time=end_time, limit=100000)
        logger.info(f"데이터베이스에서 {len(logs)}개 로그 로드 완료")
    else:
        if not log_file:
            logger.error("로그 파일 경로가 필요합니다.")
            return False
        
        logger.info(f"로그 파일에서 데이터 로드 중: {log_file}")
        from src.collector.nginx_parser import NginxParser
        parser = NginxParser()
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        logs = parser.parse_batch(lines)
        logger.info(f"로그 파일에서 {len(logs)}개 로그 로드 완료")
    
    if not logs:
        logger.error("학습 데이터가 없습니다.")
        return False
    
    # 특징 추출
    logger.info("특징 추출 중...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.extract_features(logs)
    
    if features_df.empty:
        logger.error("추출된 특징이 없습니다.")
        return False
    
    logger.info(f"특징 추출 완료: {len(features_df)}개 윈도우")
    
    # 숫자형 특징만 선택
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col != 'timestamp']
    
    X_train = features_df[numeric_cols].values
    
    logger.info(f"학습 데이터 형태: {X_train.shape}")
    
    # 모델 학습
    logger.info(f"{model_type} 모델 학습 시작...")
    detector_manager = AnomalyDetectorManager()
    
    # 모델 타입 설정
    if model_type == "isolation_forest":
        detector_manager.detector_type = "isolation_forest"
        detector_manager._initialize_detector()
    elif model_type == "pytorch_autoencoder":
        detector_manager.detector_type = "pytorch_autoencoder"
        detector_manager._initialize_detector()
    
    # 학습 수행
    config = load_config()
    if model_type == "pytorch_autoencoder":
        ae_config = config.get('autoencoder', {})
        detector_manager.train(
            X_train,
            epochs=ae_config.get('epochs', 20),
            batch_size=ae_config.get('batch_size', 32)
        )
    else:
        detector_manager.train(X_train)
    
    # 모델 저장
    if model_type == "isolation_forest":
        model_path = "models/isolation_forest"
    else:
        model_path = "models/pytorch_autoencoder"
    
    logger.info(f"모델 저장 중: {model_path}")
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    detector_manager.save_model(model_path)
    
    logger.info(f"✅ 모델 재학습 완료: {model_type}")
    return True


def main():
    parser = argparse.ArgumentParser(description='모델 자동 재학습 스크립트')
    parser.add_argument(
        '--model-type',
        type=str,
        default='isolation_forest',
        choices=['isolation_forest', 'pytorch_autoencoder'],
        help='모델 타입'
    )
    parser.add_argument(
        '--days-back',
        type=int,
        default=7,
        help='최근 며칠간의 데이터 사용 (기본값: 7일)'
    )
    parser.add_argument(
        '--use-database',
        action='store_true',
        help='데이터베이스 사용 여부'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        default=None,
        help='로그 파일 경로 (데이터베이스 미사용 시)'
    )
    
    args = parser.parse_args()
    
    success = retrain_model(
        model_type=args.model_type,
        days_back=args.days_back,
        use_database=args.use_database,
        log_file=args.log_file
    )
    
    if success:
        logger.info("모델 재학습이 성공적으로 완료되었습니다.")
        sys.exit(0)
    else:
        logger.error("모델 재학습에 실패했습니다.")
        sys.exit(1)


if __name__ == '__main__':
    main()
