#!/usr/bin/env python3
"""
Isolation Forest 모델 학습 스크립트 (빠른 테스트용)
"""
import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pickle

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing.feature_engineering import FeatureEngineer
from src.anomaly.isolation_forest import IsolationForestAnomalyDetector
import logging
import yaml

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
    parser = argparse.ArgumentParser(description='Isolation Forest 모델 학습 (빠른 테스트용)')
    parser.add_argument(
        '--data',
        type=str,
        default='data/raw_logs/nginx_access.log',
        help='학습 데이터 파일 경로'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='models/isolation_forest',
        help='모델 저장 경로'
    )
    
    args = parser.parse_args()
    
    # 데이터 로드
    logger.info("데이터 로드 중...")
    logs = load_logs_from_file(args.data)
    
    if not logs:
        logger.error("학습 데이터가 없습니다.")
        return
    
    # 특징 추출
    logger.info("특징 추출 중...")
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
    
    # Isolation Forest 모델 학습 (매우 빠름!)
    logger.info("Isolation Forest 모델 학습 시작...")
    detector = IsolationForestAnomalyDetector(
        n_estimators=100,
        contamination=0.1,
        random_state=42
    )
    
    detector.train(X_train)
    
    # 모델 저장
    logger.info(f"모델 저장 중: {args.output}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    detector.save_model(args.output)
    
    logger.info(" 모델 학습 완료! (Isolation Forest는 매우 빠릅니다)")


if __name__ == '__main__':
    main()


