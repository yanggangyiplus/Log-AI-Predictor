"""
로그 수집 관리자
실시간/배치 모드 로그 수집을 통합 관리
"""
import yaml
import time
from pathlib import Path
from typing import List, Dict, Callable, Optional
from datetime import datetime
import logging

from .tail_collector import TailCollector
from .polling_collector import PollingCollector
from .nginx_parser import NginxParser
from .apache_parser import ApacheParser

logger = logging.getLogger(__name__)


class CollectorManager:
    """로그 수집 관리자 클래스"""
    
    def __init__(self, config_path: str = "configs/config_collect.yaml"):
        """
        초기화
        
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        # 로그 타입에 따라 파서 선택
        log_type = self.config.get('log_type', 'nginx').lower()
        if log_type == 'apache':
            log_format = self.config.get('apache_format', 'combined')
            self.parser = ApacheParser(log_format=log_format)
        else:
            self.parser = NginxParser()
        self.collector: Optional[TailCollector] = None
        self.collected_logs: List[Dict] = []
        self.callbacks: List[Callable] = []
        
    def _load_config(self, config_path: str) -> Dict:
        """설정 파일 로드"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            return config.get('collector', {})
        except Exception as e:
            logger.error(f"설정 파일 로드 실패: {e}")
            return {}
    
    def add_callback(self, callback: Callable):
        """
        로그 수집 시 호출될 콜백 함수 등록
        
        Args:
            callback: 콜백 함수 (파싱된 로그 딕셔너리를 인자로 받음)
        """
        self.callbacks.append(callback)
    
    def start_realtime_collection(self, use_polling: bool = True):
        """
        실시간 수집 시작
        
        Args:
            use_polling: True면 polling 방식 (Streamlit 호환), False면 스레드 방식
        """
        log_path = self.config.get('log_path', 'data/raw_logs/nginx_access.log')
        
        if use_polling:
            # Polling 방식 (스레드 없음, Streamlit 호환)
            self.collector = PollingCollector(log_path)
            self.collector.start()
            logger.info("실시간 로그 수집 시작 (Polling 방식)")
        else:
            # 스레드 방식 (기존 방식, 호환성 유지)
            def on_new_line(line: str):
                """새 로그 라인 수집 시 호출"""
                parsed = self.parser.parse_line(line)
                if parsed:
                    self.collected_logs.append(parsed)
                    # 콜백 호출
                    for callback in self.callbacks:
                        try:
                            callback(parsed)
                        except Exception as e:
                            logger.error(f"콜백 실행 오류: {e}")
            
            self.collector = TailCollector(log_path, callback=on_new_line)
            self.collector.start()
            logger.info("실시간 로그 수집 시작 (스레드 방식)")
    
    def poll_new_logs(self) -> List[Dict]:
        """
        새로운 로그를 polling 방식으로 가져오기
        (Streamlit에서 주기적으로 호출)
        
        Returns:
            파싱된 로그 리스트
        """
        if not self.collector or not isinstance(self.collector, PollingCollector):
            return []
        
        new_lines = self.collector.poll()
        parsed_logs = []
        
        for line in new_lines:
            parsed = self.parser.parse_line(line)
            if parsed:
                self.collected_logs.append(parsed)
                parsed_logs.append(parsed)
                # 콜백 호출
                for callback in self.callbacks:
                    try:
                        callback(parsed)
                    except Exception as e:
                        logger.error(f"콜백 실행 오류: {e}")
        
        return parsed_logs
    
    def collect_batch(self) -> List[Dict]:
        """
        배치 모드로 로그 수집
        
        Returns:
            파싱된 로그 리스트
        """
        log_path = self.config.get('log_path', 'data/raw_logs/nginx_access.log')
        
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 최근 N분간의 로그만 수집 (선택사항)
            parsed_logs = self.parser.parse_batch(lines)
            
            # 콜백 호출
            for log in parsed_logs:
                for callback in self.callbacks:
                    try:
                        callback(log)
                    except Exception as e:
                        logger.error(f"콜백 실행 오류: {e}")
            
            return parsed_logs
            
        except Exception as e:
            logger.error(f"배치 수집 오류: {e}")
            return []
    
    def stop(self):
        """수집 중지"""
        if self.collector:
            self.collector.stop()
        logger.info("로그 수집 중지")
    
    def is_running(self) -> bool:
        """
        수집이 실행 중인지 확인
        
        Returns:
            실행 중이면 True, 아니면 False
        """
        if self.collector:
            return self.collector.is_running()
        return False
    
    def get_stats(self) -> Dict:
        """
        수집 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        parser_stats = self.parser.get_stats()
        return {
            'parser': parser_stats,
            'collected_count': len(self.collected_logs),
            'is_running': self.collector.is_running() if self.collector else False
        }

