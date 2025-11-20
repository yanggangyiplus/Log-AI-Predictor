#!/usr/bin/env python3
"""
로그 수집기 실행 스크립트
실시간 또는 배치 모드로 로그 수집 실행
"""
import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.collector.collector_manager import CollectorManager
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='로그 수집기 실행')
    parser.add_argument(
        '--mode',
        choices=['realtime', 'batch'],
        default='realtime',
        help='수집 모드 (realtime/batch)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config_collect.yaml',
        help='설정 파일 경로'
    )
    
    args = parser.parse_args()
    
    # 수집기 초기화
    collector = CollectorManager(config_path=args.config)
    
    def on_log(parsed_log):
        """로그 수집 콜백"""
        logger.info(f"로그 수집: {parsed_log.get('ip')} - {parsed_log.get('status_code')}")
    
    collector.add_callback(on_log)
    
    if args.mode == 'realtime':
        logger.info("실시간 수집 모드 시작")
        collector.start_realtime_collection()
        
        try:
            # 무한 대기
            import time
            while True:
                time.sleep(1)
                stats = collector.get_stats()
                logger.debug(f"수집 통계: {stats}")
        except KeyboardInterrupt:
            logger.info("수집 중지 요청")
            collector.stop()
    else:
        logger.info("배치 수집 모드 시작")
        logs = collector.collect_batch()
        logger.info(f"수집 완료: {len(logs)}개 로그")


if __name__ == '__main__':
    main()

