"""
Streamlit 메인 애플리케이션
UI 라우팅 및 통합
"""
import streamlit as st
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가 (app 모듈 import 전에 필수)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 로깅 설정
from app.utils.logger_config import app_logger as logger

# 서비스 레이어 import
from app.services.log_service import LogService
from app.services.model_service import ModelService
from app.services.feature_service import FeatureService
from app.services.anomaly_service import AnomalyService
from app.services.alert_service import AlertService

# 레이아웃 컴포넌트 import
from app.layout import (
    render_sidebar,
    render_metrics,
    render_charts,
    render_alerts,
    render_logs_table,
    render_downloads
)

# 페이지 설정
st.set_page_config(
    page_title="Log Pattern Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 서비스 인스턴스 생성
log_service = LogService(st.session_state)
model_service = ModelService(st.session_state)
feature_service = FeatureService(st.session_state)
anomaly_service = AnomalyService(st.session_state)
alert_service = AlertService(st.session_state)

# 사이드바 렌더링
render_sidebar(log_service, model_service)

# 메인 대시보드
st.title("Log Pattern Analyzer")

# 현재 모델 정보 표시
col_model_info, col_status_info = st.columns([3, 1])
with col_model_info:
    if model_service.is_model_loaded():
        model_name = model_service.get_model_name()
        st.success(f"현재 모델 **{model_name}**")
    else:
        st.info("모델을 로드해주세요 (사이드바)")
with col_status_info:
    if log_service.is_running():
        st.success("수집 중")
    else:
        st.info("대기 중")

st.markdown("---")

# 메트릭 카드 렌더링
render_metrics(log_service, alert_service, feature_service)

# 알림 렌더링
render_alerts(alert_service)

# 실시간 업데이트 처리
if log_service.is_running():
    # 새로고침 버튼
    if st.button("새로고침", use_container_width=True):
        st.rerun()
    
    # Polling 방식으로 새 로그 가져오기 (스레드 문제 해결)
    # Streamlit rerun 시마다 호출되어 새로운 로그를 가져옴
    with st.spinner("새 로그 확인 중..."):
        log_service.poll_new_logs()
    
    # 로그 데이터 처리
    logs = log_service.get_all_logs()
    
    if len(logs) > 0:
        try:
            # 특징 추출
            with st.spinner("특징 추출 중..."):
                features_df = feature_service.extract_features(logs)
            
            if not features_df.empty:
                anomaly_scores = []
                is_anomaly = []
                
                # 이상 탐지 (모델이 있을 때만)
                detector = model_service.get_current_model()
                if detector:
                    try:
                        anomaly_scores, is_anomaly = anomaly_service.predict(features_df)
                        
                        # 알림 확인
                        alert_service.check_alerts(features_df, anomaly_scores, is_anomaly)
                        
                        # 이상 탐지 결과 요약 표시
                        summary = anomaly_service.get_anomaly_summary(anomaly_scores, is_anomaly)
                        if summary['has_anomaly']:
                            st.error(
                                f"이상 탐지: {summary['count']}/{summary['total']}개 윈도우 "
                                f"({summary['percentage']:.1f}%)에서 이상 패턴 발견!"
                            )
                        else:
                            st.success(f"정상: {summary['total']}개 윈도우 모두 정상")
                    except Exception as e:
                        st.warning("이상 탐지 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
                        logger.error(f"이상 탐지 오류: {e}", exc_info=True)
                else:
                    st.info("모델을 로드하면 이상 탐지가 실행됩니다.")
                
                st.markdown("---")
                
                # 차트 렌더링
                render_charts(features_df, anomaly_scores, is_anomaly, detector)
                
                st.markdown("---")
                
                # 로그 테이블 렌더링
                recent_logs = log_service.get_recent_logs()
                render_logs_table(recent_logs)
                
                st.markdown("---")
                
                # 다운로드 섹션 렌더링
                current_model = model_service.get_current_model_path()
                alerts = alert_service.get_alerts()
                render_downloads(logs, features_df, alerts, current_model)
            else:
                st.info("특징 추출 중... 로그가 더 필요합니다.")
        except Exception as e:
            st.error("데이터 처리 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.")
            logger.error(f"데이터 처리 오류: {e}", exc_info=True)
    else:
        st.info("수집된 로그가 없습니다. 사이드바에서 '수집 시작'을 클릭하세요.")
else:
    st.info("사이드바에서 수집을 시작하세요")

