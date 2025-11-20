"""
다운로드 레이아웃 컴포넌트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict
from app.utils.formatter import (
    prepare_dataframe_for_csv,
    prepare_alerts_for_json,
    create_report_data
)


def render_downloads(logs: List[Dict], features_df: pd.DataFrame, 
                    alerts: List[Dict], current_model: str):
    """
    다운로드 섹션 렌더링
    
    Args:
        logs: 로그 리스트
        features_df: 특징 DataFrame
        alerts: 알림 리스트
        current_model: 현재 모델 경로
    """
    st.subheader("데이터 다운로드")
    col_dl1, col_dl2, col_dl3, col_dl4 = st.columns(4)
    
    with col_dl1:
        # 로그 데이터 CSV 다운로드
        if len(logs) > 0:
            try:
                logs_df = pd.DataFrame(logs)
                logs_df = prepare_dataframe_for_csv(logs_df)
                csv_logs = logs_df.to_csv(index=False).encode('utf-8-sig')
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
        if len(features_df) > 0:
            try:
                features_copy = prepare_dataframe_for_csv(features_df.copy())
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
        if len(alerts) > 0:
            try:
                alerts_for_json = prepare_alerts_for_json(alerts)
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
        if len(logs) > 0 or len(features_df) > 0:
            try:
                report = create_report_data(
                    len(logs), len(features_df), len(alerts), current_model
                )
                # anomaly_count 계산
                anomaly_count = sum(1 for alert in alerts if alert.get('type') == 'anomaly')
                report['statistics']['anomaly_count'] = anomaly_count
                
                report_json = json.dumps(report, ensure_ascii=False, indent=2).encode('utf-8')
                st.download_button(
                    label="전체 리포트 (JSON)",
                    data=report_json,
                    file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"다운로드 준비 실패: {e}")
        else:
            st.button("전체 리포트 (JSON)", disabled=True, use_container_width=True)

