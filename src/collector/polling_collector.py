"""
Polling 방식 로그 수집 모듈
Streamlit과 호환되는 스레드 없는 수집 방식
"""
import os
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PollingCollector:
    """Polling 방식 로그 수집 클래스 (스레드 없음)"""
    
    def __init__(self, log_path: str):
        """
        초기화
        
        Args:
            log_path: 모니터링할 로그 파일 경로
        """
        self.log_path = log_path
        self.file_position = 0
        self.running = False
        
        # 로그 파일 존재 확인
        if not os.path.exists(log_path):
            logger.warning(f"로그 파일이 존재하지 않습니다: {log_path}")
            # 빈 파일 생성
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            open(log_path, 'a').close()
    
    def start(self):
        """수집 시작 (파일 위치만 초기화)"""
        if self.running:
            logger.warning("이미 수집이 실행 중입니다.")
            return
        
        self.running = True
        
        # 파일 위치 초기화 (파일 끝부터 읽기)
        try:
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(0, 2)  # 파일 끝으로 이동
                self.file_position = f.tell()
        except Exception as e:
            logger.error(f"파일 위치 초기화 실패: {e}")
            self.file_position = 0
        
        logger.info(f"Polling 수집 시작: {self.log_path}")
    
    def stop(self):
        """수집 중지"""
        self.running = False
        logger.info("Polling 수집 중지")
    
    def poll(self) -> List[str]:
        """
        새로운 로그 라인을 polling 방식으로 가져오기
        (스레드 없이 직접 호출)
        
        Returns:
            새로운 로그 라인 리스트
        """
        if not self.running:
            return []
        
        new_lines = []
        
        try:
            if not os.path.exists(self.log_path):
                return []
            
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 이전 위치로 이동
                f.seek(self.file_position)
                
                # 새로운 라인 읽기
                for line in f:
                    line = line.strip()
                    if line:
                        new_lines.append(line)
                
                # 파일 위치 업데이트
                self.file_position = f.tell()
                
        except Exception as e:
            logger.error(f"파일 읽기 오류: {e}")
        
        return new_lines
    
    def is_running(self) -> bool:
        """수집 중인지 확인"""
        return self.running

