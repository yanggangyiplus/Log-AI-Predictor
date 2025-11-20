"""
Feature Engineering 모듈
로그 데이터를 머신러닝 모델이 학습할 수 있는 숫자형 특징으로 변환
"""
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Feature Engineering 클래스"""
    
    def __init__(self, window_size: int = 60):
        """
        초기화
        
        Args:
            window_size: 시간 윈도우 크기 (초)
        """
        self.window_size = window_size
        self.historical_stats = defaultdict(list)
    
    def extract_features(self, logs: List[Dict], window_seconds: int = 60) -> pd.DataFrame:
        """
        로그 리스트에서 특징 추출
        
        Args:
            logs: 파싱된 로그 딕셔너리 리스트
            window_seconds: 시간 윈도우 크기 (초)
            
        Returns:
            특징 DataFrame
        """
        if not logs:
            return pd.DataFrame()
        
        df = pd.DataFrame(logs)
        
        # 타임스탬프를 인덱스로 설정
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
        
        # 시간 윈도우별로 그룹화
        features_list = []
        
        # 시간 윈도우 생성
        if len(df) > 0:
            start_time = df.index.min()
            end_time = df.index.max()
            
            current_time = start_time
            while current_time <= end_time:
                window_end = current_time + timedelta(seconds=window_seconds)
                window_logs = df[(df.index >= current_time) & (df.index < window_end)]
                
                if len(window_logs) > 0:
                    features = self._extract_window_features(window_logs, current_time)
                    features_list.append(features)
                
                current_time = window_end
        
        if not features_list:
            return pd.DataFrame()
        
        features_df = pd.DataFrame(features_list)
        return features_df
    
    def _extract_window_features(self, window_logs: pd.DataFrame, timestamp: datetime) -> Dict:
        """
        시간 윈도우 내 로그에서 특징 추출
        
        Args:
            window_logs: 윈도우 내 로그 DataFrame
            timestamp: 윈도우 시작 타임스탬프
            
        Returns:
            특징 딕셔너리
        """
        total_requests = len(window_logs)
        
        # 기본 통계
        features = {
            'timestamp': timestamp,
            'total_requests': total_requests,
            'rps': total_requests / self.window_size,  # 초당 요청 수
        }
        
        if total_requests == 0:
            # 요청이 없으면 기본값 반환
            features.update({
                'error_rate_5xx': 0.0,
                'error_rate_4xx': 0.0,
                'error_rate_404': 0.0,
                'avg_response_time': 0.0,
                'max_response_time': 0.0,
                'min_response_time': 0.0,
                'unique_ips': 0,
                'unique_paths': 0,
                'top_path_count': 0,
            })
            return features
        
        # 상태 코드별 통계
        status_codes = window_logs['status_code']
        features['error_rate_5xx'] = (status_codes >= 500).sum() / total_requests * 100
        features['error_rate_4xx'] = ((status_codes >= 400) & (status_codes < 500)).sum() / total_requests * 100
        features['error_rate_404'] = (status_codes == 404).sum() / total_requests * 100
        
        # 응답 시간 통계
        if 'response_time' in window_logs.columns:
            response_times = window_logs['response_time'].dropna()
            if len(response_times) > 0:
                features['avg_response_time'] = response_times.mean()
                features['max_response_time'] = response_times.max()
                features['min_response_time'] = response_times.min()
                features['std_response_time'] = response_times.std()
            else:
                features['avg_response_time'] = 0.0
                features['max_response_time'] = 0.0
                features['min_response_time'] = 0.0
                features['std_response_time'] = 0.0
        else:
            # 응답 시간이 없으면 body_bytes_sent를 대체 지표로 사용
            bytes_sent = window_logs['body_bytes_sent']
            features['avg_response_time'] = bytes_sent.mean() / 1000  # 대략적인 추정
            features['max_response_time'] = bytes_sent.max() / 1000
            features['min_response_time'] = bytes_sent.min() / 1000
            features['std_response_time'] = bytes_sent.std() / 1000
        
        # IP 통계
        features['unique_ips'] = window_logs['ip'].nunique()
        features['top_ip_count'] = window_logs['ip'].value_counts().iloc[0] if len(window_logs) > 0 else 0
        
        # URL 경로 통계
        if 'url_path' in window_logs.columns:
            features['unique_paths'] = window_logs['url_path'].nunique()
            top_path = window_logs['url_path'].value_counts()
            features['top_path_count'] = top_path.iloc[0] if len(top_path) > 0 else 0
            features['top_path_ratio'] = features['top_path_count'] / total_requests if total_requests > 0 else 0
        else:
            features['unique_paths'] = 0
            features['top_path_count'] = 0
            features['top_path_ratio'] = 0
        
        # 메서드 통계
        if 'method' in window_logs.columns:
            method_counts = window_logs['method'].value_counts()
            features['get_ratio'] = method_counts.get('GET', 0) / total_requests if total_requests > 0 else 0
            features['post_ratio'] = method_counts.get('POST', 0) / total_requests if total_requests > 0 else 0
        
        # 변화율 계산 (이전 윈도우와 비교)
        features.update(self._calculate_change_rates(features))
        
        return features
    
    def _calculate_change_rates(self, current_features: Dict) -> Dict:
        """
        이전 윈도우 대비 변화율 계산
        
        Args:
            current_features: 현재 윈도우 특징
            
        Returns:
            변화율 특징 딕셔너리
        """
        change_features = {}
        
        # 이전 통계가 있으면 변화율 계산
        if len(self.historical_stats) > 0:
            prev_features = self.historical_stats['features'][-1] if self.historical_stats.get('features') else {}
            
            for key in ['rps', 'error_rate_5xx', 'error_rate_404', 'avg_response_time']:
                if key in prev_features and prev_features[key] > 0:
                    change = ((current_features.get(key, 0) - prev_features[key]) / prev_features[key]) * 100
                    change_features[f'{key}_change'] = change
                else:
                    change_features[f'{key}_change'] = 0.0
        else:
            # 첫 번째 윈도우면 변화율 0
            for key in ['rps', 'error_rate_5xx', 'error_rate_404', 'avg_response_time']:
                change_features[f'{key}_change'] = 0.0
        
        # 현재 특징을 히스토리에 저장
        if 'features' not in self.historical_stats:
            self.historical_stats['features'] = []
        self.historical_stats['features'].append(current_features.copy())
        
        # 히스토리 크기 제한 (메모리 관리)
        if len(self.historical_stats['features']) > 100:
            self.historical_stats['features'] = self.historical_stats['features'][-100:]
        
        return change_features
    
    def normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        특징 정규화 (0-1 범위로 스케일링)
        
        Args:
            features_df: 특징 DataFrame
            
        Returns:
            정규화된 특징 DataFrame
        """
        if features_df.empty:
            return features_df
        
        # 정규화할 컬럼 선택 (숫자형 컬럼만)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'timestamp']
        
        normalized_df = features_df.copy()
        
        for col in numeric_cols:
            col_min = normalized_df[col].min()
            col_max = normalized_df[col].max()
            if col_max > col_min:
                normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
            else:
                normalized_df[col] = 0.0
        
        return normalized_df

