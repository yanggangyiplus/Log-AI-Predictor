#!/usr/bin/env python3
"""
이상 탐지기 실행 스크립트
실시간으로 로그를 수집하고 이상 탐지 수행
"""
import sys
import argparse
import time
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collector.collector_manager import CollectorManager
from src.preprocessing.feature_engineering import FeatureEngineer
from src.anomaly.detector_manager import AnomalyDetectorManager
import logging
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='이상 탐지기 실행')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='모델 파일 경로'
    )
    parser.add_argument(
        '--config-collect',
        type=str,
        default='configs/config_collect.yaml',
        help='수집 설정 파일 경로'
    )
    parser.add_argument(
        '--config-model',
        type=str,
        default='configs/config_model.yaml',
        help='모델 설정 파일 경로'
    )
    
    args = parser.parse_args()
    
    # 모델 로드
    logger.info(f"모델 로드 중: {args.model}")
    detector = AnomalyDetectorManager(config_path=args.config_model)
    detector.load_model(args.model)
    
    # 특징 엔지니어 초기화
    feature_engineer = FeatureEngineer()
    
    # 로그 버퍼
    log_buffer = []
    
    def on_log(parsed_log):
        """로그 수집 콜백"""
        log_buffer.append(parsed_log)
        # 최근 100개만 유지
        if len(log_buffer) > 100:
            log_buffer.pop(0)
        
        # 충분한 로그가 모이면 특징 추출 및 이상 탐지
        if len(log_buffer) >= 10:
            features_df = feature_engineer.extract_features(log_buffer[-60:])  # 최근 60개
            
            if not features_df.empty:
                numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                numeric_cols = [col for col in numeric_cols if col != 'timestamp']
                
                if len(numeric_cols) > 0:
                    X = features_df[numeric_cols].values
                    scores, is_anomaly = detector.predict(X)
                    
                    # 이상 탐지 결과 출력
                    for i, (score, anomaly) in enumerate(zip(scores, is_anomaly)):
                        if anomaly:
                            logger.warning(
                                f" 이상 탐지! 점수: {score:.4f}, "
                                f"시간: {features_df.iloc[i]['timestamp']}"
                            )
                        else:
                            logger.debug(f"정상: 점수 {score:.4f}")
    
    # 수집기 시작
    collector = CollectorManager(config_path=args.config_collect)
    collector.add_callback(on_log)
    collector.start_realtime_collection()
    
    logger.info("이상 탐지기 실행 중... (Ctrl+C로 종료)")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("중지 요청")
        collector.stop()


if __name__ == '__main__':
    main()

