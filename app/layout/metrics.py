"""
메트릭 카드 레이아웃 컴포넌트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from app.services.log_service import LogService
from app.services.alert_service import AlertService
from app.services.feature_service import FeatureService


def render_metrics(log_service: LogService, alert_service: AlertService, 
                  feature_service: FeatureService):
    """
    메트릭 카드 렌더링
    
    Args:
        log_service: 로그 서비스 인스턴스
        alert_service: 알림 서비스 인스턴스
        feature_service: 특징 서비스 인스턴스
    """
    st.subheader("실시간 통계")
    
    if log_service.is_running():
        stats = log_service.get_stats()
        logs = log_service.get_all_logs()
        alerts = alert_service.get_alerts()
        features = feature_service.get_features()
        
        # 첫 번째 행: 주요 통계
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_logs = len(logs)
            st.metric("수집된 로그", f"{total_logs:,}")
        
        with col2:
            if stats.get('parser'):
                success_rate = stats['parser']['success_rate']
                st.metric("파싱 성공률", f"{success_rate:.1f}%", 
                         delta=f"{success_rate:.1f}%" if success_rate > 90 else None)
            else:
                st.metric("파싱 성공률", "0%")
        
        with col3:
            anomaly_count = alert_service.get_anomaly_count()
            recent_alerts = alert_service.get_recent_alerts(10)
            recent_anomaly = sum(1 for alert in recent_alerts if alert.get('type') == 'anomaly')
            st.metric("이상 탐지", f"{anomaly_count}", 
                     delta=f"+{recent_anomaly}" if recent_anomaly > 0 else None, 
                     delta_color="inverse")
        
        with col4:
            window_count = len(features)
            st.metric("시간 윈도우", f"{window_count}")
        
        with col5:
            if len(logs) > 0:
                # 최근 100개 로그에서 5xx 에러율 계산
                recent_logs = logs[-100:]
                error_5xx_count = sum(1 for log in recent_logs 
                                    if isinstance(log.get('status_code'), int) 
                                    and log.get('status_code', 0) >= 500)
                error_rate = (error_5xx_count / len(recent_logs) * 100) if recent_logs else 0
                st.metric("5xx 에러율", f"{error_rate:.1f}%", 
                         delta=f"{error_rate:.1f}%" if error_rate > 5 else None, 
                         delta_color="inverse")
            else:
                st.metric("5xx 에러율", "0%")
        
        st.markdown("---")
        
        # 진행 상황 표시
        if len(logs) > 0:
            from app.utils.constants import MAX_LOGS_IN_MEMORY
            progress_value = min(len(logs) / MAX_LOGS_IN_MEMORY, 1.0)
            st.progress(progress_value, text=f"로그 수집 진행: {len(logs)}/{MAX_LOGS_IN_MEMORY}")
    else:
        # 수집 중이 아닐 때 기본 통계 표시
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("수집된 로그", "0")
        with col2:
            st.metric("파싱 성공률", "0%")
        with col3:
            st.metric("이상 탐지", "0")
        with col4:
            st.metric("시간 윈도우", "0")
        with col5:
            st.metric("5xx 에러율", "0%")
        st.info("사이드바에서 '수집 시작'을 클릭하세요")
        st.markdown("---")

