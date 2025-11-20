"""
알림 레이아웃 컴포넌트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from datetime import datetime
from app.services.alert_service import AlertService
from app.utils.constants import RECENT_ALERTS_DISPLAY
from app.utils.formatter import format_timestamp


def render_alerts(alert_service: AlertService):
    """
    알림 섹션 렌더링
    
    Args:
        alert_service: 알림 서비스 인스턴스
    """
    alerts = alert_service.get_recent_alerts(RECENT_ALERTS_DISPLAY)
    
    if alerts:
        st.subheader("실시간 알림")
        alert_container = st.container()
        with alert_container:
            for alert in alerts:
                timestamp_str = format_timestamp(
                    alert.get('timestamp', datetime.now())
                )
                severity = alert.get('severity', 'info')
                message = alert.get('message', '')
                
                if severity == 'high':
                    st.error(f"[{timestamp_str}] {message}")
                elif severity == 'medium':
                    st.warning(f"[{timestamp_str}] {message}")
                else:
                    st.info(f"[{timestamp_str}] {message}")
        st.markdown("---")
    else:
        st.info("현재 이상 탐지 없음 - 시스템 정상 작동 중")
        st.markdown("---")

