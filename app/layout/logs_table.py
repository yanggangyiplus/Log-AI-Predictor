"""
로그 테이블 레이아웃 컴포넌트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
from typing import List, Dict
from app.utils.formatter import highlight_status_code
from app.utils.constants import RECENT_LOGS_DISPLAY


def render_logs_table(logs: List[Dict]):
    """
    로그 테이블 렌더링
    
    Args:
        logs: 로그 리스트
    """
    st.subheader(f"최근 로그 (최대 {RECENT_LOGS_DISPLAY}개)")
    
    if not logs:
        st.info("수집된 로그가 없습니다.")
        return
    
    # DataFrame 생성
    recent_logs_list = logs[-RECENT_LOGS_DISPLAY:]
    recent_logs_df = pd.DataFrame(recent_logs_list)
    
    if recent_logs_df.empty:
        st.info("로그 데이터가 없습니다.")
        return
    
    # 표시할 컬럼 선택
    display_cols = ['timestamp', 'ip', 'method', 'url_path', 'status_code']
    available_cols = [col for col in display_cols if col in recent_logs_df.columns]
    
    # 상태 코드에 따라 색상 스타일 적용
    styled_df = recent_logs_df[available_cols].style.applymap(
        highlight_status_code, subset=['status_code']
    )
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
            st.metric("4xx", f"{error_4xx}개", 
                     delta=f"{error_4xx}개" if error_4xx > 0 else None, 
                     delta_color="inverse")
        with col_stat3:
            error_5xx = sum(status_counts.get(code, 0) for code in [500, 502, 503])
            st.metric("5xx", f"{error_5xx}개", 
                     delta=f"{error_5xx}개" if error_5xx > 0 else None, 
                     delta_color="inverse")

