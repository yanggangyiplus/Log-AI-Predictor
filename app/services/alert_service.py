"""
알림 서비스
알림 생성 및 관리 비즈니스 로직
"""
import sys
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import pandas as pd
import logging

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.constants import (
    MAX_ALERTS_IN_MEMORY,
    SESSION_KEY_ALERTS,
    DEFAULT_ERROR_RATE_THRESHOLD,
    DEFAULT_RESPONSE_TIME_THRESHOLD,
    DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD
)
from app.utils.cache import load_alert_config
from app.services.notification_service import NotificationService

logger = logging.getLogger(__name__)


class AlertService:
    """알림 서비스 클래스"""
    
    def __init__(self, session_state):
        """
        초기화
        
        Args:
            session_state: Streamlit session_state 객체
        """
        self.session_state = session_state
        self.notification_service = NotificationService()
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if SESSION_KEY_ALERTS not in self.session_state:
            self.session_state[SESSION_KEY_ALERTS] = []
    
    def check_alerts(self, features_df: pd.DataFrame, 
                    anomaly_scores: List[float], 
                    is_anomaly: List[bool]) -> List[Dict]:
        """
        알림 조건 확인
        
        Args:
            features_df: 특징 DataFrame
            anomaly_scores: 이상 탐지 점수 리스트
            is_anomaly: 이상 여부 리스트
            
        Returns:
            알림 리스트
        """
        alert_config = load_alert_config()
        conditions = alert_config.get('conditions', {})
        alerts = []
        
        if features_df.empty:
            return alerts
        
        # 최근 데이터 확인
        recent_features = features_df.iloc[-1] if len(features_df) > 0 else None
        
        if recent_features is not None:
            # 5xx 에러 비율 확인
            error_rate = recent_features.get('error_rate_5xx', 0)
            threshold = conditions.get('error_rate_threshold', DEFAULT_ERROR_RATE_THRESHOLD)
            if error_rate > threshold:
                alerts.append({
                    'type': 'error_rate',
                    'message': f'5xx 에러 비율이 {error_rate:.2f}%로 임계값을 초과했습니다.',
                    'severity': 'high',
                    'timestamp': datetime.now()
                })
            
            # 응답 시간 확인
            avg_response_time = recent_features.get('avg_response_time', 0)
            threshold = conditions.get('response_time_threshold', DEFAULT_RESPONSE_TIME_THRESHOLD)
            if avg_response_time > threshold:
                alerts.append({
                    'type': 'response_time',
                    'message': f'평균 응답 시간이 {avg_response_time:.2f}ms로 임계값을 초과했습니다.',
                    'severity': 'medium',
                    'timestamp': datetime.now()
                })
        
        # 이상 탐지 결과 확인
        if len(anomaly_scores) > 0:
            recent_anomaly_score = anomaly_scores[-1] if isinstance(anomaly_scores, list) else anomaly_scores
            recent_is_anomaly = is_anomaly[-1] if isinstance(is_anomaly, list) else is_anomaly
            
            threshold = conditions.get('reconstruction_error_threshold', DEFAULT_RECONSTRUCTION_ERROR_THRESHOLD)
            
            if recent_is_anomaly or (isinstance(recent_anomaly_score, (int, float)) and recent_anomaly_score > threshold):
                alerts.append({
                    'type': 'anomaly',
                    'message': f'이상 패턴이 감지되었습니다. (점수: {recent_anomaly_score:.4f})',
                    'severity': 'high',
                    'timestamp': datetime.now()
                })
        
        # 알림 추가 및 외부 알림 전송
        if alerts:
            existing_alerts = self.session_state.get(SESSION_KEY_ALERTS, [])
            
            for alert in alerts:
                existing_alerts.append(alert)
                
                # 외부 알림 전송 (Slack, Webhook 등)
                try:
                    self.notification_service.send_notification(
                        alert_type=alert.get('type', 'unknown'),
                        message=alert.get('message', ''),
                        severity=alert.get('severity', 'medium'),
                        details={
                            'type': alert.get('type'),
                            'timestamp': alert.get('timestamp').isoformat() if hasattr(alert.get('timestamp'), 'isoformat') else str(alert.get('timestamp'))
                        }
                    )
                except Exception as e:
                    logger.error(f"외부 알림 전송 실패: {e}", exc_info=True)
            
            # 메모리 관리: 최근 N개만 유지
            if len(existing_alerts) > MAX_ALERTS_IN_MEMORY:
                existing_alerts = existing_alerts[-MAX_ALERTS_IN_MEMORY:]
            
            self.session_state[SESSION_KEY_ALERTS] = existing_alerts
        
        return alerts
    
    def get_alerts(self) -> List[Dict]:
        """
        모든 알림 반환
        
        Returns:
            알림 리스트
        """
        return self.session_state.get(SESSION_KEY_ALERTS, [])
    
    def get_recent_alerts(self, count: int = 5) -> List[Dict]:
        """
        최근 알림 반환
        
        Args:
            count: 반환할 알림 개수
            
        Returns:
            알림 리스트
        """
        alerts = self.get_alerts()
        return alerts[-count:] if len(alerts) > count else alerts
    
    def get_anomaly_count(self) -> int:
        """
        이상 탐지 알림 개수 반환
        
        Returns:
            이상 탐지 알림 개수
        """
        alerts = self.get_alerts()
        return sum(1 for alert in alerts if alert.get('type') == 'anomaly')

