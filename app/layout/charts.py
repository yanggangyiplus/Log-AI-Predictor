"""
차트 레이아웃 컴포넌트
Plotly 그래프 생성 및 렌더링
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import List


@st.cache_data(ttl=60)  # 1분 캐싱
def create_rps_chart(features_df: pd.DataFrame) -> go.Figure:
    """
    RPS 차트 생성 (캐싱 적용)
    
    Args:
        features_df: 특징 DataFrame
        
    Returns:
        Plotly Figure 객체
    """
    if features_df.empty or 'rps' not in features_df.columns:
        return None
    
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
    return fig


@st.cache_data(ttl=60)
def create_error_rate_chart(features_df: pd.DataFrame) -> go.Figure:
    """
    에러 비율 차트 생성 (캐싱 적용)
    
    Args:
        features_df: 특징 DataFrame
        
    Returns:
        Plotly Figure 객체
    """
    if features_df.empty or 'error_rate_5xx' not in features_df.columns:
        return None
    
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
    return fig


@st.cache_data(ttl=60)
def create_response_time_chart(features_df: pd.DataFrame) -> go.Figure:
    """
    응답 시간 차트 생성 (캐싱 적용)
    
    Args:
        features_df: 특징 DataFrame
        
    Returns:
        Plotly Figure 객체
    """
    if features_df.empty or 'avg_response_time' not in features_df.columns:
        return None
    
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
    return fig


def create_anomaly_score_chart(features_df: pd.DataFrame, 
                               anomaly_scores: List[float],
                               is_anomaly: List[bool],
                               threshold: float = None) -> go.Figure:
    """
    이상 탐지 점수 차트 생성
    
    Args:
        features_df: 특징 DataFrame
        anomaly_scores: 이상 탐지 점수 리스트
        is_anomaly: 이상 여부 리스트
        threshold: 임계값 (선택사항)
        
    Returns:
        Plotly Figure 객체
    """
    if features_df.empty or len(anomaly_scores) == 0:
        return None
    
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
    
    # 임계값 라인 추가
    if threshold is not None:
        fig.add_hline(
            y=threshold, 
            line_dash="dash", 
            line_color="red", 
            annotation_text=f"임계값: {threshold:.4f}"
        )
    
    fig.update_layout(
        title="이상 탐지 점수",
        xaxis_title="시간",
        yaxis_title="이상 점수",
        height=300,
        hovermode='x unified',
        showlegend=False,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    return fig


def render_charts(features_df: pd.DataFrame, anomaly_scores: List[float], 
                  is_anomaly: List[bool], detector=None):
    """
    차트 섹션 렌더링
    
    Args:
        features_df: 특징 DataFrame
        anomaly_scores: 이상 탐지 점수 리스트
        is_anomaly: 이상 여부 리스트
        detector: 모델 객체 (임계값 가져오기용)
    """
    st.subheader("실시간 시각화")
    
    # 그래프 표시 (2x2 그리드)
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 요청 수 (RPS)")
        fig_rps = create_rps_chart(features_df)
        if fig_rps:
            st.plotly_chart(fig_rps, use_container_width=True)
        else:
            st.info("데이터 수집 중...")
    
    with col2:
        st.markdown("### 에러 비율")
        fig_error = create_error_rate_chart(features_df)
        if fig_error:
            st.plotly_chart(fig_error, use_container_width=True)
        else:
            st.info("데이터 수집 중...")
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### 응답 시간")
        fig_response = create_response_time_chart(features_df)
        if fig_response:
            st.plotly_chart(fig_response, use_container_width=True)
        else:
            st.info("데이터 수집 중...")
    
    with col4:
        st.markdown("### 이상 탐지 점수")
        if len(anomaly_scores) > 0:
            # 임계값 가져오기
            threshold = None
            if detector and hasattr(detector, 'detector') and hasattr(detector.detector, 'threshold'):
                threshold = detector.detector.threshold
            
            fig_anomaly = create_anomaly_score_chart(
                features_df, anomaly_scores, is_anomaly, threshold
            )
            if fig_anomaly:
                st.plotly_chart(fig_anomaly, use_container_width=True)
            
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

