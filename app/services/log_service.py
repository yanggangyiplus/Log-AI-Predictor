"""
로그 수집 서비스
로그 수집 및 관리 비즈니스 로직
"""
import sys
from pathlib import Path
from typing import List, Dict, Optional, Callable
import logging

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.collector.collector_manager import CollectorManager
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    logging.error(f"모듈 로드 실패: {e}")

from app.utils.constants import (
    MAX_LOGS_IN_MEMORY,
    SESSION_KEY_COLLECTOR,
    SESSION_KEY_LOGS_DATA
)

logger = logging.getLogger(__name__)


class LogService:
    """로그 수집 서비스 클래스"""
    
    def __init__(self, session_state):
        """
        초기화
        
        Args:
            session_state: Streamlit session_state 객체
        """
        self.session_state = session_state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화 (최소 정보만)"""
        if SESSION_KEY_COLLECTOR not in self.session_state:
            self.session_state[SESSION_KEY_COLLECTOR] = None
        # 로그 데이터는 캐시로 관리 (세션 상태에 직접 저장하지 않음)
        if SESSION_KEY_LOGS_DATA not in self.session_state:
            self.session_state[SESSION_KEY_LOGS_DATA] = []
    
    def start_collection(self, mode: str = "realtime") -> tuple[bool, str]:
        """
        로그 수집 시작
        
        Args:
            mode: 수집 모드 ("realtime" 또는 "batch")
            
        Returns:
            (성공 여부, 메시지)
        """
        if not MODULES_LOADED:
            return False, "모듈 로드 실패. 터미널에서 오류를 확인하세요."
        
        if self.session_state[SESSION_KEY_COLLECTOR] is not None:
            return False, "이미 수집이 실행 중입니다."
        
        try:
            collector = CollectorManager()
            
            # 콜백 등록 (polling 방식에서는 사용 안 함)
            collector.add_callback(self._on_new_log)
            
            if mode == "realtime":
                # Polling 방식으로 실시간 수집 시작 (스레드 없음)
                collector.start_realtime_collection(use_polling=True)
            else:
                logs = collector.collect_batch()
                self.session_state[SESSION_KEY_LOGS_DATA].extend(logs)
            
            self.session_state[SESSION_KEY_COLLECTOR] = collector
            return True, "수집 시작됨"
        except Exception as e:
            logger.error(f"수집 시작 실패: {e}", exc_info=True)
            return False, f"수집 시작 실패: {e}"
    
    def stop_collection(self) -> tuple[bool, str]:
        """
        로그 수집 중지
        
        Returns:
            (성공 여부, 메시지)
        """
        if self.session_state[SESSION_KEY_COLLECTOR]:
            try:
                self.session_state[SESSION_KEY_COLLECTOR].stop()
                self.session_state[SESSION_KEY_COLLECTOR] = None
                return True, "수집 중지됨"
            except Exception as e:
                logger.error(f"수집 중지 실패: {e}", exc_info=True)
                return False, f"수집 중지 실패: {e}"
        return False, "수집기가 실행 중이 아닙니다."
    
    def is_running(self) -> bool:
        """
        수집이 실행 중인지 확인
        
        Returns:
            실행 중이면 True
        """
        collector = self.session_state.get(SESSION_KEY_COLLECTOR)
        if collector:
            return collector.is_running()
        return False
    
    def get_stats(self) -> Dict:
        """
        수집 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        collector = self.session_state.get(SESSION_KEY_COLLECTOR)
        if collector:
            return collector.get_stats()
        return {
            'parser': {'success_rate': 0.0},
            'collected_count': 0,
            'is_running': False
        }
    
    def get_recent_logs(self, count: int = 20) -> List[Dict]:
        """
        최근 로그 반환
        
        Args:
            count: 반환할 로그 개수
            
        Returns:
            로그 리스트
        """
        logs = self.session_state.get(SESSION_KEY_LOGS_DATA, [])
        return logs[-count:] if len(logs) > count else logs
    
    def get_all_logs(self) -> List[Dict]:
        """
        모든 로그 반환
        
        Returns:
            로그 리스트
        """
        return self.session_state.get(SESSION_KEY_LOGS_DATA, [])
    
    def _on_new_log(self, parsed_log: Dict):
        """
        새 로그 수집 시 콜백 (내부 사용)
        
        Args:
            parsed_log: 파싱된 로그 딕셔너리
        """
        # 세션 상태에 저장 (메모리 관리 적용)
        logs = self.session_state.get(SESSION_KEY_LOGS_DATA, [])
        logs.append(parsed_log)
        
        # 메모리 관리: 최근 N개만 유지
        if len(logs) > MAX_LOGS_IN_MEMORY:
            logs = logs[-MAX_LOGS_IN_MEMORY:]
        
        self.session_state[SESSION_KEY_LOGS_DATA] = logs
        
        # 데이터 업데이트 플래그 설정
        self.session_state['data_updated'] = True
    
    def poll_new_logs(self) -> List[Dict]:
        """
        새로운 로그를 polling 방식으로 가져오기
        (실시간 수집의 스레드 문제 해결)
        
        Returns:
            새로운 로그 리스트
        """
        collector = self.session_state.get(SESSION_KEY_COLLECTOR)
        if not collector or not collector.is_running():
            return []
        
        try:
            # CollectorManager의 poll_new_logs 메서드 사용
            new_logs = collector.poll_new_logs()
            
            # 새 로그를 세션 상태에 추가
            for log in new_logs:
                self._on_new_log(log)
            
            return new_logs
        except Exception as e:
            logger.error(f"Polling 중 오류: {e}", exc_info=True)
        
        return []

