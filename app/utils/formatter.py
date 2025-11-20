"""
포맷터 유틸리티
데이터 포맷팅 및 변환 함수
"""
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Any


def format_timestamp(dt: datetime) -> str:
    """
    타임스탬프를 문자열로 포맷팅
    
    Args:
        dt: datetime 객체
        
    Returns:
        포맷된 문자열
    """
    return dt.strftime('%H:%M:%S')


def format_datetime_iso(dt: datetime) -> str:
    """
    datetime을 ISO 형식 문자열로 변환
    
    Args:
        dt: datetime 객체
        
    Returns:
        ISO 형식 문자열
    """
    return dt.isoformat()


def prepare_dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    """
    DataFrame을 CSV 다운로드용으로 준비 (타임스탬프 문자열 변환)
    
    Args:
        df: 원본 DataFrame
        
    Returns:
        변환된 DataFrame
    """
    df_copy = df.copy()
    if 'timestamp' in df_copy.columns:
        df_copy['timestamp'] = df_copy['timestamp'].astype(str)
    return df_copy


def prepare_alerts_for_json(alerts: List[Dict]) -> List[Dict]:
    """
    알림 리스트를 JSON 다운로드용으로 준비
    
    Args:
        alerts: 알림 리스트
        
    Returns:
        변환된 알림 리스트
    """
    alerts_for_json = []
    for alert in alerts:
        alert_copy = alert.copy()
        if 'timestamp' in alert_copy and isinstance(alert_copy['timestamp'], datetime):
            alert_copy['timestamp'] = alert_copy['timestamp'].isoformat()
        alerts_for_json.append(alert_copy)
    return alerts_for_json


def highlight_status_code(val: Any) -> str:
    """
    상태 코드에 따라 CSS 스타일 반환
    
    Args:
        val: 상태 코드 값
        
    Returns:
        CSS 스타일 문자열
    """
    if isinstance(val, (int, float)):
        if val >= 500:
            return 'background-color: #ffcccc'  # 빨간색
        elif val >= 400:
            return 'background-color: #fff4cc'  # 노란색
        else:
            return 'background-color: #ccffcc'  # 초록색
    return ''


def create_report_data(logs_count: int, features_count: int, 
                      alerts_count: int, current_model: str) -> Dict:
    """
    리포트 데이터 생성
    
    Args:
        logs_count: 로그 개수
        features_count: 특징 개수
        alerts_count: 알림 개수
        current_model: 현재 모델 경로
        
    Returns:
        리포트 딕셔너리
    """
    anomaly_count = 0  # 실제로는 alerts에서 계산해야 함
    
    return {
        'timestamp': datetime.now().isoformat(),
        'statistics': {
            'total_logs': logs_count,
            'total_windows': features_count,
            'total_alerts': alerts_count,
            'anomaly_count': anomaly_count
        },
        'model': current_model or 'None',
        'logs_count': logs_count,
        'features_count': features_count,
        'alerts_count': alerts_count
    }

