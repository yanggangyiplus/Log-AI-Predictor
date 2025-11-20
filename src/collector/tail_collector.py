"""
실시간 로그 수집 모듈
tail -f 방식으로 로그 파일을 실시간 모니터링
"""
import time
import os
import threading
from typing import Callable, Optional
from queue import Queue
import logging

logger = logging.getLogger(__name__)


class TailCollector:
    """실시간 로그 수집 클래스"""
    
    def __init__(self, log_path: str, callback: Optional[Callable] = None):
        """
        초기화
        
        Args:
            log_path: 모니터링할 로그 파일 경로
            callback: 새 로그 라인이 수집될 때 호출될 콜백 함수
        """
        self.log_path = log_path
        self.callback = callback
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.queue = Queue()
        self.file_position = 0
        
        # 로그 파일 존재 확인
        if not os.path.exists(log_path):
            logger.warning(f"로그 파일이 존재하지 않습니다: {log_path}")
            # 빈 파일 생성
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            open(log_path, 'a').close()
    
    def start(self):
        """수집 시작"""
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
        
        # 백그라운드 스레드 시작
        self.thread = threading.Thread(target=self._collect_loop, daemon=True)
        self.thread.start()
        logger.info(f"로그 수집 시작: {self.log_path}")
    
    def stop(self):
        """수집 중지"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=5)
        logger.info("로그 수집 중지")
    
    def _collect_loop(self):
        """수집 루프 (백그라운드 스레드에서 실행)"""
        while self.running:
            try:
                self._read_new_lines()
                time.sleep(0.1)  # CPU 사용률 조절
            except Exception as e:
                logger.error(f"수집 루프 오류: {e}")
                time.sleep(1)
    
    def _read_new_lines(self):
        """
        파일에서 새로운 라인 읽기
        """
        try:
            if not os.path.exists(self.log_path):
                return
            
            with open(self.log_path, 'r', encoding='utf-8', errors='ignore') as f:
                # 이전 위치로 이동
                f.seek(self.file_position)
                
                # 새로운 라인 읽기
                new_lines = []
                for line in f:
                    line = line.strip()
                    if line:
                        new_lines.append(line)
                
                # 파일 위치 업데이트
                self.file_position = f.tell()
                
                # 새 라인이 있으면 처리
                if new_lines:
                    if self.callback:
                        for line in new_lines:
                            try:
                                self.callback(line)
                            except Exception as e:
                                logger.error(f"콜백 실행 오류: {e}")
                    else:
                        # 콜백이 없으면 큐에 추가
                        for line in new_lines:
                            self.queue.put(line)
                            
        except Exception as e:
            logger.error(f"파일 읽기 오류: {e}")
    
    def get_new_lines(self, timeout: float = 0.1) -> list:
        """
        큐에서 새로운 로그 라인 가져오기
        
        Args:
            timeout: 타임아웃 (초)
            
        Returns:
            로그 라인 리스트
        """
        lines = []
        try:
            while True:
                line = self.queue.get(timeout=timeout)
                lines.append(line)
        except:
            pass
        return lines
    
    def is_running(self) -> bool:
        """수집 중인지 확인"""
        return self.running

