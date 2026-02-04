"""
데이터베이스 관리 모듈
SQLite 기반 로그 저장 및 조회
"""
import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path
import logging
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseManager:
    """데이터베이스 관리 클래스"""
    
    def __init__(self, db_path: str = "data/database/logs.db"):
        """
        초기화
        
        Args:
            db_path: 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self._ensure_db_directory()
        self._initialize_database()
    
    def _ensure_db_directory(self):
        """데이터베이스 디렉토리 생성"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _initialize_database(self):
        """데이터베이스 테이블 초기화"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # 로그 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                ip TEXT,
                method TEXT,
                url_path TEXT,
                status_code INTEGER,
                response_time REAL,
                body_bytes_sent INTEGER,
                user_agent TEXT,
                raw_log TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 특징 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS features (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                window_start DATETIME NOT NULL,
                rps REAL,
                error_rate_5xx REAL,
                error_rate_4xx REAL,
                error_rate_404 REAL,
                avg_response_time REAL,
                max_response_time REAL,
                min_response_time REAL,
                unique_ips INTEGER,
                unique_paths INTEGER,
                features_json TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 이상 탐지 결과 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomaly_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                window_start DATETIME NOT NULL,
                anomaly_score REAL,
                is_anomaly BOOLEAN,
                model_type TEXT,
                features_id INTEGER,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (features_id) REFERENCES features(id)
            )
        ''')
        
        # 알림 테이블
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                alert_type TEXT NOT NULL,
                message TEXT NOT NULL,
                severity TEXT,
                metadata TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # 인덱스 생성 (조회 성능 향상)
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON logs(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_features_timestamp ON features(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_anomaly_timestamp ON anomaly_results(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')
        
        conn.commit()
        conn.close()
        logger.info(f"데이터베이스 초기화 완료: {self.db_path}")
    
    def insert_log(self, log_data: Dict) -> int:
        """
        로그 데이터 삽입
        
        Args:
            log_data: 로그 딕셔너리
            
        Returns:
            삽입된 로그의 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO logs (
                    timestamp, ip, method, url_path, status_code,
                    response_time, body_bytes_sent, user_agent, raw_log
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                log_data.get('timestamp'),
                log_data.get('ip'),
                log_data.get('method'),
                log_data.get('url_path'),
                log_data.get('status_code'),
                log_data.get('response_time'),
                log_data.get('body_bytes_sent'),
                log_data.get('user_agent'),
                log_data.get('raw_log', '')
            ))
            
            log_id = cursor.lastrowid
            conn.commit()
            return log_id
        except Exception as e:
            logger.error(f"로그 삽입 실패: {e}", exc_info=True)
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def insert_logs_batch(self, logs: List[Dict]) -> int:
        """
        여러 로그를 배치로 삽입
        
        Args:
            logs: 로그 딕셔너리 리스트
            
        Returns:
            삽입된 로그 개수
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.executemany('''
                INSERT INTO logs (
                    timestamp, ip, method, url_path, status_code,
                    response_time, body_bytes_sent, user_agent, raw_log
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', [
                (
                    log.get('timestamp'),
                    log.get('ip'),
                    log.get('method'),
                    log.get('url_path'),
                    log.get('status_code'),
                    log.get('response_time'),
                    log.get('body_bytes_sent'),
                    log.get('user_agent'),
                    log.get('raw_log', '')
                )
                for log in logs
            ])
            
            count = cursor.rowcount
            conn.commit()
            return count
        except Exception as e:
            logger.error(f"배치 로그 삽입 실패: {e}", exc_info=True)
            conn.rollback()
            return 0
        finally:
            conn.close()
    
    def insert_features(self, features_data: Dict) -> int:
        """
        특징 데이터 삽입
        
        Args:
            features_data: 특징 딕셔너리
            
        Returns:
            삽입된 특징의 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # features_json에 모든 특징 저장
            features_json = json.dumps(features_data)
            
            cursor.execute('''
                INSERT INTO features (
                    timestamp, window_start, rps, error_rate_5xx,
                    error_rate_4xx, error_rate_404, avg_response_time,
                    max_response_time, min_response_time, unique_ips,
                    unique_paths, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features_data.get('timestamp'),
                features_data.get('timestamp'),  # window_start
                features_data.get('rps'),
                features_data.get('error_rate_5xx'),
                features_data.get('error_rate_4xx'),
                features_data.get('error_rate_404'),
                features_data.get('avg_response_time'),
                features_data.get('max_response_time'),
                features_data.get('min_response_time'),
                features_data.get('unique_ips'),
                features_data.get('unique_paths'),
                features_json
            ))
            
            feature_id = cursor.lastrowid
            conn.commit()
            return feature_id
        except Exception as e:
            logger.error(f"특징 삽입 실패: {e}", exc_info=True)
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def insert_anomaly_result(self, anomaly_data: Dict) -> int:
        """
        이상 탐지 결과 삽입
        
        Args:
            anomaly_data: 이상 탐지 결과 딕셔너리
            
        Returns:
            삽입된 결과의 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO anomaly_results (
                    timestamp, window_start, anomaly_score, is_anomaly,
                    model_type, features_id
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                anomaly_data.get('timestamp'),
                anomaly_data.get('window_start'),
                anomaly_data.get('anomaly_score'),
                anomaly_data.get('is_anomaly'),
                anomaly_data.get('model_type'),
                anomaly_data.get('features_id')
            ))
            
            result_id = cursor.lastrowid
            conn.commit()
            return result_id
        except Exception as e:
            logger.error(f"이상 탐지 결과 삽입 실패: {e}", exc_info=True)
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def insert_alert(self, alert_data: Dict) -> int:
        """
        알림 데이터 삽입
        
        Args:
            alert_data: 알림 딕셔너리
            
        Returns:
            삽입된 알림의 ID
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            metadata_json = json.dumps(alert_data.get('metadata', {}))
            
            cursor.execute('''
                INSERT INTO alerts (
                    timestamp, alert_type, message, severity, metadata
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                alert_data.get('timestamp', datetime.now()),
                alert_data.get('type', 'unknown'),
                alert_data.get('message', ''),
                alert_data.get('severity', 'medium'),
                metadata_json
            ))
            
            alert_id = cursor.lastrowid
            conn.commit()
            return alert_id
        except Exception as e:
            logger.error(f"알림 삽입 실패: {e}", exc_info=True)
            conn.rollback()
            return None
        finally:
            conn.close()
    
    def get_logs(self, start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 limit: int = 1000) -> List[Dict]:
        """
        로그 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 조회 개수
            
        Returns:
            로그 리스트
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM logs WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        logs = [dict(row) for row in rows]
        conn.close()
        
        return logs
    
    def get_features(self, start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """
        특징 데이터 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 조회 개수
            
        Returns:
            특징 DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM features WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_anomaly_results(self, start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None,
                           limit: int = 1000) -> pd.DataFrame:
        """
        이상 탐지 결과 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 조회 개수
            
        Returns:
            이상 탐지 결과 DataFrame
        """
        conn = sqlite3.connect(self.db_path)
        
        query = 'SELECT * FROM anomaly_results WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_alerts(self, start_time: Optional[datetime] = None,
                  end_time: Optional[datetime] = None,
                  limit: int = 100) -> List[Dict]:
        """
        알림 조회
        
        Args:
            start_time: 시작 시간
            end_time: 종료 시간
            limit: 최대 조회 개수
            
        Returns:
            알림 리스트
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM alerts WHERE 1=1'
        params = []
        
        if start_time:
            query += ' AND timestamp >= ?'
            params.append(start_time)
        
        if end_time:
            query += ' AND timestamp <= ?'
            params.append(end_time)
        
        query += ' ORDER BY timestamp DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        alerts = [dict(row) for row in rows]
        conn.close()
        
        return alerts
    
    def get_statistics(self) -> Dict:
        """
        데이터베이스 통계 조회
        
        Returns:
            통계 딕셔너리
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # 로그 개수
        cursor.execute('SELECT COUNT(*) FROM logs')
        stats['total_logs'] = cursor.fetchone()[0]
        
        # 특징 개수
        cursor.execute('SELECT COUNT(*) FROM features')
        stats['total_features'] = cursor.fetchone()[0]
        
        # 이상 탐지 결과 개수
        cursor.execute('SELECT COUNT(*) FROM anomaly_results')
        stats['total_anomaly_results'] = cursor.fetchone()[0]
        
        # 알림 개수
        cursor.execute('SELECT COUNT(*) FROM alerts')
        stats['total_alerts'] = cursor.fetchone()[0]
        
        # 최근 로그 시간
        cursor.execute('SELECT MAX(timestamp) FROM logs')
        stats['latest_log_time'] = cursor.fetchone()[0]
        
        # 최근 특징 시간
        cursor.execute('SELECT MAX(timestamp) FROM features')
        stats['latest_feature_time'] = cursor.fetchone()[0]
        
        conn.close()
        
        return stats
