"""
Streamlit 기반 실시간 모니터링 대시보드
로그 분석 및 장애 예측 결과를 시각화
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import sys
import os
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 모듈 import (에러 처리 추가)
try:
    from src.collector.collector_manager import CollectorManager
    from src.preprocessing.feature_engineering import FeatureEngineer
    from src.anomaly.detector_manager import AnomalyDetectorManager
    from src.prediction.failure_predictor import FailurePredictor
    import yaml
    MODULES_LOADED = True
except Exception as e:
    MODULES_LOADED = False
    import logging
    logging.error(f"모듈 로드 실패: {e}")

import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 페이지 설정
st.set_page_config(
    page_title="Log Pattern Analyzer",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 세션 상태 초기화
if 'collector' not in st.session_state:
    st.session_state.collector = None
if 'detector' not in st.session_state:
    st.session_state.detector = None
if MODULES_LOADED:
    if 'feature_engineer' not in st.session_state:
        st.session_state.feature_engineer = FeatureEngineer()
    if 'failure_predictor' not in st.session_state:
        st.session_state.failure_predictor = FailurePredictor()
if 'logs_data' not in st.session_state:
    st.session_state.logs_data = []
if 'features_data' not in st.session_state:
    st.session_state.features_data = pd.DataFrame()
if 'anomaly_results' not in st.session_state:
    st.session_state.anomaly_results = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []
if 'current_model' not in st.session_state:
    st.session_state.current_model = None


@st.cache_data(ttl=300)  # 5분간 캐싱
def load_alert_config():
    """알림 설정 로드 (캐싱 적용)"""
    try:
        config_path = Path('configs/config_alert.yaml')
        if not config_path.exists():
            logger.warning("알림 설정 파일이 없습니다. 기본값을 사용합니다.")
            return {}
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config.get('alert', {})
    except Exception as e:
        logger.error(f"알림 설정 로드 실패: {e}")
        return {}


def check_alerts(features_df, anomaly_scores, is_anomaly):
    """알림 조건 확인"""
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
        if error_rate > conditions.get('error_rate_threshold', 5.0):
            alerts.append({
                'type': 'error_rate',
                'message': f' 5xx 에러 비율이 {error_rate:.2f}%로 임계값을 초과했습니다.',
                'severity': 'high',
                'timestamp': datetime.now()
            })
        
        # 응답 시간 확인
        avg_response_time = recent_features.get('avg_response_time', 0)
        if avg_response_time > conditions.get('response_time_threshold', 1000):
            alerts.append({
                'type': 'response_time',
                'message': f' 평균 응답 시간이 {avg_response_time:.2f}ms로 임계값을 초과했습니다.',
                'severity': 'medium',
                'timestamp': datetime.now()
            })
    
    # 이상 탐지 결과 확인
    if len(anomaly_scores) > 0:
        recent_anomaly_score = anomaly_scores[-1] if isinstance(anomaly_scores, list) else anomaly_scores
        recent_is_anomaly = is_anomaly[-1] if isinstance(is_anomaly, list) else is_anomaly
        
        threshold = alert_config.get('conditions', {}).get('reconstruction_error_threshold', 0.75)
        
        if recent_is_anomaly or (isinstance(recent_anomaly_score, (int, float)) and recent_anomaly_score > threshold):
            alerts.append({
                'type': 'anomaly',
                'message': f' 이상 패턴이 감지되었습니다. (점수: {recent_anomaly_score:.4f})',
                'severity': 'high',
                'timestamp': datetime.now()
            })
    
    return alerts


def on_new_log(parsed_log):
    """새 로그 수집 시 콜백"""
    st.session_state.logs_data.append(parsed_log)
    # 최근 1000개만 유지 (메모리 관리)
    if len(st.session_state.logs_data) > 1000:
        st.session_state.logs_data = st.session_state.logs_data[-1000:]
    
    # 세션 상태 업데이트 플래그 (성능 최적화)
    if 'data_updated' not in st.session_state:
        st.session_state.data_updated = True


# 사이드바
with st.sidebar:
    st.title("설정")
    
    # 수집 모드 선택
    collection_mode = st.radio(
        "수집 모드",
        ["실시간", "배치"],
        index=0
    )
    
    if st.button("수집 시작", type="primary"):
        if not MODULES_LOADED:
            st.error("모듈 로드 실패. 터미널에서 오류를 확인하세요.")
        elif st.session_state.collector is None:
            try:
                collector = CollectorManager()
                collector.add_callback(on_new_log)
                
                if collection_mode == "실시간":
                    collector.start_realtime_collection()
                else:
                    logs = collector.collect_batch()
                    st.session_state.logs_data.extend(logs)
                
                st.session_state.collector = collector
                st.success("수집 시작됨")
            except Exception as e:
                st.error(f"수집 시작 실패: {e}")
                st.exception(e)
        else:
            st.warning("이미 수집이 실행 중입니다.")
    
    if st.button("수집 중지"):
        if st.session_state.collector:
            st.session_state.collector.stop()
            st.session_state.collector = None
            st.success("수집 중지됨")
    
    st.divider()
    
    # 모델 선택 및 로드
    st.subheader("모델 선택")
    
    # 모델 선택 라디오 버튼
    model_choice = st.radio(
        "사용할 모델 선택",
        ["Isolation Forest (빠르고 안정적)", "PyTorch AutoEncoder (더 정확)"],
        index=0,
        help="Isolation Forest: 빠르고 안정적, PyTorch AutoEncoder: 더 정확한 이상 탐지"
    )
    
    # 선택에 따라 모델 경로 설정
    if model_choice == "Isolation Forest (빠르고 안정적)":
        model_path = "models/isolation_forest"
        model_info = "빠른 학습 및 예측, 안정적"
    else:
        model_path = "models/pytorch_autoencoder"
        model_info = "복잡한 패턴 학습 가능, 더 정확"
    
    st.info(model_info)
    
    # 모델 로드 버튼
    col_load, col_status = st.columns([2, 1])
    with col_load:
        if st.button("모델 로드", type="primary", use_container_width=True):
            if not MODULES_LOADED:
                st.error("모듈 로드 실패")
            else:
                try:
                    with st.spinner("모델 로드 중..."):
                        detector = AnomalyDetectorManager()
                        detector.load_model(model_path)
                        st.session_state.detector = detector
                        st.session_state.current_model = model_path
                        st.success("모델 로드 완료")
                except FileNotFoundError:
                    st.error(f"모델 파일을 찾을 수 없습니다 {model_path}")
                    st.info("먼저 모델을 학습시켜주세요")
                    if "isolation"in model_path.lower():
                        st.code(f"python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log --output {model_path}")
                    else:
                        st.code(f"python scripts/train_pytorch_autoencoder.py --data data/raw_logs/nginx_access.log --output {model_path}")
                except Exception as e:
                    st.error(f"모델 로드 실패: {e}")
                    st.exception(e)
    
    with col_status:
        if st.session_state.get('detector'):
            st.success("로드됨")
        else:
            st.info("⏳ 대기 중")
    
    st.divider()
    
    # 통계
    st.subheader("통계")
    if st.session_state.collector:
        stats = st.session_state.collector.get_stats()
        st.metric("수집된 로그", stats['collected_count'])
        st.metric("파싱 성공률", f"{stats['parser']['success_rate']:.1f}%")
    else:
        st.info("수집기를 시작하세요")


# 메인 대시보드
st.title("Log Pattern Analyzer")

# 현재 모델 정보 표시
col_model_info, col_status_info = st.columns([3, 1])
with col_model_info:
    if st.session_state.get('detector'):
        model_name = "Isolation Forest"if "isolation"in st.session_state.get('current_model', '').lower() else "PyTorch AutoEncoder"
        st.success(f"현재 모델 **{model_name}**")
    else:
        st.info("⏳ 모델을 로드해주세요 (사이드바)")
with col_status_info:
    if st.session_state.collector and st.session_state.collector.is_running():
        st.success("수집 중")
    else:
        st.info("대기 중")

st.markdown("---")

# 실시간 통계 카드 (메인 화면 상단)
st.subheader("실시간 통계")

if st.session_state.collector and st.session_state.collector.is_running():
    stats = st.session_state.collector.get_stats()
    
    # 첫 번째 행: 주요 통계
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_logs = len(st.session_state.logs_data)
        # delta는 이전 값과 비교하여 표시 (현재는 단순 증가만 표시)
        st.metric("수집된 로그", f"{total_logs:,}")
    
    with col2:
        if stats.get('parser'):
            success_rate = stats['parser']['success_rate']
            st.metric("파싱 성공률", f"{success_rate:.1f}%", delta=f"{success_rate:.1f}%"if success_rate > 90 else None)
        else:
            st.metric("파싱 성공률", "0%")
    
    with col3:
        anomaly_count = sum(1 for alert in st.session_state.alerts if alert.get('type') == 'anomaly')
        # 최근 알림 중 이상 탐지 개수
        recent_anomaly = sum(1 for alert in st.session_state.alerts[-10:] if alert.get('type') == 'anomaly')
        st.metric("이상 탐지", f"{anomaly_count}", delta=f"+{recent_anomaly}"if recent_anomaly > 0 else None, delta_color="inverse")
    
    with col4:
        if len(st.session_state.features_data) > 0:
            window_count = len(st.session_state.features_data)
            st.metric("시간 윈도우", f"{window_count}")
        else:
            st.metric("시간 윈도우", "0")
    
    with col5:
        if len(st.session_state.logs_data) > 0:
            # 최근 100개 로그에서 5xx 에러율 계산 (성능 최적화)
            recent_logs = st.session_state.logs_data[-100:]
            error_5xx_count = sum(1 for log in recent_logs if isinstance(log.get('status_code'), int) and log.get('status_code', 0) >= 500)
            error_rate = (error_5xx_count / len(recent_logs) * 100) if recent_logs else 0
            st.metric("5xx 에러율", f"{error_rate:.1f}%", delta=f"{error_rate:.1f}%"if error_rate > 5 else None, delta_color="inverse")
        else:
            st.metric("5xx 에러율", "0%")
    
    st.markdown("---")
    
    # 진행 상황 표시
    if len(st.session_state.logs_data) > 0:
        progress_value = min(len(st.session_state.logs_data) / 1000, 1.0)  # 최대 1000개 기준
        st.progress(progress_value, text=f"로그 수집 진행: {len(st.session_state.logs_data)}/1000")
else:
    # 수집 중이 아닐 때도 기본 통계 표시
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

# 알림 표시 (더 눈에 띄게)
if st.session_state.alerts:
    st.subheader("실시간 알림")
    alert_container = st.container()
    with alert_container:
        for alert in st.session_state.alerts[-5:]:  # 최근 5개만 표시
            timestamp_str = alert.get('timestamp', datetime.now()).strftime('%H:%M:%S')
            if alert['severity'] == 'high':
                st.error(f"[{timestamp_str}] {alert['message']}")
            elif alert['severity'] == 'medium':
                st.warning(f"[{timestamp_str}] {alert['message']}")
            else:
                st.info(f"[{timestamp_str}] {alert['message']}")
    st.markdown("---")
else:
    st.info("현재 이상 탐지 없음 - 시스템 정상 작동 중")
    st.markdown("---")

# 실시간 업데이트 (수동 새로고침 방식으로 변경)
if st.session_state.collector and st.session_state.collector.is_running():
    # 새로고침 버튼
    if st.button("새로고침", use_container_width=True):
        st.rerun()
    
    # 로그 데이터 처리
    if len(st.session_state.logs_data) > 0:
        try:
            # 특징 추출 (최근 100개 로그 사용, 성능 최적화)
            with st.spinner("특징 추출 중..."):
                features_df = st.session_state.feature_engineer.extract_features(
                    st.session_state.logs_data[-100:]
                )
            
            if not features_df.empty:
                # 기존 데이터와 병합 (더 효율적인 방법)
                if st.session_state.features_data.empty:
                    st.session_state.features_data = features_df.copy()
                else:
                    # 중복 제거 후 병합
                    combined = pd.concat([st.session_state.features_data, features_df], ignore_index=True)
                    st.session_state.features_data = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
                
                # 최근 100개 윈도우만 유지 (메모리 관리)
                if len(st.session_state.features_data) > 100:
                    st.session_state.features_data = st.session_state.features_data.tail(100).reset_index(drop=True)
                
                anomaly_scores = []
                is_anomaly = []
                
                # 이상 탐지 (모델이 있을 때만)
                if st.session_state.detector:
                    try:
                        if hasattr(st.session_state.detector, 'detector') and st.session_state.detector.detector.is_trained:
                            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
                            numeric_cols = [col for col in numeric_cols if col != 'timestamp']
                            
                            if len(numeric_cols) > 0:
                                X = features_df[numeric_cols].values
                                
                                # 이상 탐지 실행
                                with st.spinner("이상 탐지 실행 중..."):
                                    anomaly_scores, is_anomaly = st.session_state.detector.predict(X)
                                
                                # 알림 확인
                                new_alerts = check_alerts(features_df, anomaly_scores, is_anomaly)
                                if new_alerts:
                                    st.session_state.alerts.extend(new_alerts)
                                
                                # 최근 50개 알림만 유지 (메모리 관리)
                                if len(st.session_state.alerts) > 50:
                                    st.session_state.alerts = st.session_state.alerts[-50:]
                        else:
                            st.info("모델이 학습되지 않았습니다. 모델을 먼저 로드해주세요.")
                    except Exception as e:
                        st.warning(f"이상 탐지 중 오류: {e}")
                        logger.error(f"이상 탐지 오류: {e}", exc_info=True)
                
                # 이상 탐지 결과 요약 표시
                if len(anomaly_scores) > 0:
                    anomaly_count = sum(is_anomaly)
                    total_count = len(is_anomaly)
                    anomaly_percentage = (anomaly_count / total_count * 100) if total_count > 0 else 0
                    
                    if anomaly_count > 0:
                        st.error(f"이상 탐지: {anomaly_count}/{total_count}개 윈도우 ({anomaly_percentage:.1f}%)에서 이상 패턴 발견!")
                    else:
                        st.success(f"정상: {total_count}개 윈도우 모두 정상")
                
                st.markdown("---")
                
                # 시각화 섹션
                st.subheader("실시간 시각화")
                
                # 그래프 표시 (2x2 그리드)
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("###  요청 수 (RPS)")
                    if 'rps' in features_df.columns:
                        fig = px.line(
                            features_df,
                            x='timestamp',
                            y='rps',
                            title="초당 요청 수",
                            labels={'rps': '요청/초', 'timestamp': '시간'}
                        )
                        fig.update_traces(line_color='#1f77b4', line_width=2)
                        fig.update_layout(
                            height=300,
                            showlegend=False,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("데이터 수집 중...")
                
                with col2:
                    st.markdown("###  에러 비율")
                    if 'error_rate_5xx' in features_df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=features_df['timestamp'],
                            y=features_df['error_rate_5xx'],
                            mode='lines',
                            name='5xx 에러',
                            fill='tozeroy',
                            line=dict(color='red', width=2)
                        ))
                        fig.update_layout(
                            title="5xx 에러 비율 (%)",
                            xaxis_title="시간",
                            yaxis_title="에러 비율 (%)",
                            height=300,
                            showlegend=False,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("데이터 수집 중...")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.markdown("###  응답 시간")
                    if 'avg_response_time' in features_df.columns:
                        fig = px.line(
                            features_df,
                            x='timestamp',
                            y='avg_response_time',
                            title="평균 응답 시간 (ms)",
                            labels={'avg_response_time': '응답 시간 (ms)', 'timestamp': '시간'}
                        )
                        fig.update_traces(line_color='#2ca02c', line_width=2)
                        fig.update_layout(
                            height=300,
                            showlegend=False,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("데이터 수집 중...")
                
                with col4:
                    st.markdown("###  이상 탐지 점수")
                    if len(anomaly_scores) > 0:
                        # 이상 여부에 따라 색상 변경
                        colors = ['red' if is_anom else 'green' for is_anom in is_anomaly]
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=features_df['timestamp'],
                            y=anomaly_scores,
                            mode='lines+markers',
                            name='이상 점수',
                            marker=dict(color=colors, size=8),
                            line=dict(color='orange', width=2)
                        ))
                        
                        # 임계값 라인 추가 (가능한 경우)
                        if hasattr(st.session_state.detector, 'detector') and hasattr(st.session_state.detector.detector, 'threshold'):
                            threshold = st.session_state.detector.detector.threshold
                            fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                                        annotation_text=f"임계값: {threshold:.4f}")
                        
                        fig.update_layout(
                            title="이상 탐지 점수",
                            xaxis_title="시간",
                            yaxis_title="이상 점수",
                            height=300,
                            hovermode='x unified',
                            showlegend=False,
                            margin=dict(l=0, r=0, t=30, b=0)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # 최근 이상 점수 표시
                        if len(anomaly_scores) > 0 and len(is_anomaly) > 0:
                            recent_score = float(anomaly_scores[-1])
                            recent_is_anomaly = bool(is_anomaly[-1])
                            if recent_is_anomaly:
                                st.error(f"최근 점수: {recent_score:.4f} (이상 탐지됨)")
                            else:
                                st.success(f"최근 점수: {recent_score:.4f} (정상)")
                            
                            # 평균 점수도 표시
                            avg_score = float(np.mean(anomaly_scores))
                            st.caption(f"평균 점수: {avg_score:.4f}")
                    else:
                        st.info("모델을 로드하면 이상 탐지 점수가 표시됩니다.")
                
                st.markdown("---")
                
                # 최근 로그 테이블 (색상 코딩)
                st.subheader("최근 로그 (최대 20개)")
                # DataFrame 생성 최적화
                if len(st.session_state.logs_data) > 0:
                    recent_logs_list = st.session_state.logs_data[-20:]
                    recent_logs_df = pd.DataFrame(recent_logs_list)
                else:
                    recent_logs_df = pd.DataFrame()
                
                if not recent_logs_df.empty:
                    display_cols = ['timestamp', 'ip', 'method', 'url_path', 'status_code']
                    available_cols = [col for col in display_cols if col in recent_logs_df.columns]
                    
                    # 상태 코드에 따라 색상 스타일 적용
                    def highlight_status(val):
                        if isinstance(val, (int, float)):
                            if val >= 500:
                                return 'background-color: #ffcccc'  # 빨간색
                            elif val >= 400:
                                return 'background-color: #fff4cc'  # 노란색
                            else:
                                return 'background-color: #ccffcc'  # 초록색
                        return ''
                    
                    styled_df = recent_logs_df[available_cols].style.applymap(highlight_status, subset=['status_code'])
                    st.dataframe(styled_df, use_container_width=True, height=300)
                    
                    # 통계 요약
                    if 'status_code' in recent_logs_df.columns:
                        status_counts = recent_logs_df['status_code'].value_counts()
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        with col_stat1:
                            success = status_counts.get(200, 0) + status_counts.get(201, 0)
                            st.metric("성공", f"{success}개")
                        with col_stat2:
                            error_4xx = sum(status_counts.get(code, 0) for code in [400, 404])
                            st.metric("4xx", f"{error_4xx}개", delta=f"{error_4xx}개"if error_4xx > 0 else None, delta_color="inverse")
                        with col_stat3:
                            error_5xx = sum(status_counts.get(code, 0) for code in [500, 502, 503])
                            st.metric("5xx", f"{error_5xx}개", delta=f"{error_5xx}개"if error_5xx > 0 else None, delta_color="inverse")
                
                st.markdown("---")
                
                # 데이터 다운로드 섹션
                st.subheader("데이터 다운로드")
                col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
                
                with col_dl1:
                    # 로그 데이터 CSV 다운로드
                    if len(st.session_state.logs_data) > 0:
                        try:
                            logs_df = pd.DataFrame(st.session_state.logs_data)
                            # 타임스탬프를 문자열로 변환 (CSV 호환성)
                            if 'timestamp' in logs_df.columns:
                                logs_df['timestamp'] = logs_df['timestamp'].astype(str)
                            csv_logs = logs_df.to_csv(index=False).encode('utf-8-sig')  # Excel 호환성
                            st.download_button(
                                label="로그 데이터 (CSV)",
                                data=csv_logs,
                                file_name=f"logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"다운로드 준비 실패: {e}")
                    else:
                        st.button("로그 데이터 (CSV)", disabled=True, use_container_width=True)
                
                with col_dl2:
                    # 특징 데이터 CSV 다운로드
                    if len(st.session_state.features_data) > 0:
                        try:
                            # 타임스탬프를 문자열로 변환
                            features_copy = st.session_state.features_data.copy()
                            if 'timestamp' in features_copy.columns:
                                features_copy['timestamp'] = features_copy['timestamp'].astype(str)
                            csv_features = features_copy.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="특징 데이터 (CSV)",
                                data=csv_features,
                                file_name=f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"다운로드 준비 실패: {e}")
                    else:
                        st.button("특징 데이터 (CSV)", disabled=True, use_container_width=True)
                
                with col_dl3:
                    # 알림 데이터 JSON 다운로드
                    if len(st.session_state.alerts) > 0:
                        try:
                            import json
                            # 타임스탬프를 문자열로 변환
                            alerts_for_json = []
                            for alert in st.session_state.alerts:
                                alert_copy = alert.copy()
                                if 'timestamp' in alert_copy and isinstance(alert_copy['timestamp'], datetime):
                                    alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
                                alerts_for_json.append(alert_copy)
                            alerts_json = json.dumps(alerts_for_json, ensure_ascii=False, indent=2).encode('utf-8')
                            st.download_button(
                                label="알림 데이터 (JSON)",
                                data=alerts_json,
                                file_name=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json",
                                use_container_width=True
                            )
                        except Exception as e:
                            st.error(f"다운로드 준비 실패: {e}")
                    else:
                        st.button("알림 데이터 (JSON)", disabled=True, use_container_width=True)
                
                with col_dl4:
                    # 전체 리포트 다운로드 (JSON)
                    if len(st.session_state.logs_data) > 0 or len(st.session_state.features_data) > 0:
                        report = {
                            'timestamp': datetime.now().isoformat(),
                            'statistics': {
                                'total_logs': len(st.session_state.logs_data),
                                'total_windows': len(st.session_state.features_data),
                                'total_alerts': len(st.session_state.alerts),
                                'anomaly_count': sum(1 for alert in st.session_state.alerts if alert.get('type') == 'anomaly')
                            },
                            'model': st.session_state.get('current_model', 'None'),
                            'logs_count': len(st.session_state.logs_data),
                            'features_count': len(st.session_state.features_data),
                            'alerts_count': len(st.session_state.alerts)
                        }
                        import json
                        report_json = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
                        st.download_button(
                            label="전체 리포트 (JSON)",
                            data=report_json,
                            file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    else:
                        st.button("전체 리포트 (JSON)", disabled=True, use_container_width=True)
            else:
                st.info("특징 추출 중... 로그가 더 필요합니다.")
        except Exception as e:
            st.error(f"데이터 처리 중 오류: {e}")
            st.exception(e)
    else:
        st.info("수집된 로그가 없습니다. 사이드바에서 '수집 시작'을 클릭하세요.")

else:
    st.info("사이드바에서 수집을 시작하세요")

