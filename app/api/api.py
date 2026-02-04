"""
REST API 엔드포인트
외부 시스템과의 통합을 위한 API 제공
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from flask import Flask, request, jsonify
from datetime import datetime
import logging
import numpy as np
from typing import List, Dict

# 프로젝트 모듈 import
try:
    from src.anomaly.detector_manager import AnomalyDetectorManager
    from src.collector.nginx_parser import NginxParser
    from src.preprocessing.feature_engineering import FeatureEngineer
    from app.utils.constants import MODEL_ISOLATION_FOREST
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    logging.error(f"모듈 로드 실패: {e}")

logger = logging.getLogger(__name__)

app = Flask(__name__)

# 전역 상태 (모델 및 파서 인스턴스)
detector_manager: AnomalyDetectorManager = None
nginx_parser: NginxParser = None
feature_engineer: FeatureEngineer = None
alerts: List[Dict] = []


def _initialize_services():
    """서비스 초기화"""
    global detector_manager, nginx_parser, feature_engineer
    
    if not MODULES_LOADED:
        logger.error("필수 모듈을 로드할 수 없습니다.")
        return False
    
    try:
        # 모델 로드 (기본값: Isolation Forest)
        if detector_manager is None:
            detector_manager = AnomalyDetectorManager()
            try:
                detector_manager.load_model(str(MODEL_ISOLATION_FOREST))
                logger.info("모델 로드 완료")
            except FileNotFoundError:
                logger.warning("모델 파일을 찾을 수 없습니다. 학습이 필요합니다.")
        
        # 파서 초기화
        if nginx_parser is None:
            nginx_parser = NginxParser()
        
        # 특징 엔지니어 초기화
        if feature_engineer is None:
            feature_engineer = FeatureEngineer()
        
        return True
    except Exception as e:
        logger.error(f"서비스 초기화 실패: {e}", exc_info=True)
        return False


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'modules_loaded': MODULES_LOADED,
        'model_loaded': detector_manager is not None
    })


@app.route('/api/anomaly/detect', methods=['POST'])
def detect_anomaly():
    """
    이상 탐지 API
    
    Request Body:
        {
            "features": [[feature_vector1], [feature_vector2], ...]
        }
    
    Response:
        {
            "scores": [score1, score2, ...],
            "is_anomaly": [bool1, bool2, ...],
            "timestamp": "2024-01-01T00:00:00"
        }
    """
    try:
        if not _initialize_services():
            return jsonify({'error': '서비스 초기화 실패'}), 500
        
        if detector_manager is None:
            return jsonify({'error': '모델이 로드되지 않았습니다. 먼저 모델을 학습하세요.'}), 400
        
        data = request.json
        if not data or 'features' not in data:
            return jsonify({'error': 'features 필드가 필요합니다.'}), 400
        
        features = data.get('features', [])
        if not features:
            return jsonify({'error': 'features 배열이 비어있습니다.'}), 400
        
        # NumPy 배열로 변환
        features_array = np.array(features)
        
        # 모델 예측 수행
        scores, is_anomaly = detector_manager.predict(features_array)
        
        # NumPy 배열을 리스트로 변환
        scores_list = scores.tolist() if isinstance(scores, np.ndarray) else scores
        is_anomaly_list = is_anomaly.tolist() if isinstance(is_anomaly, np.ndarray) else is_anomaly
        
        return jsonify({
            'scores': scores_list,
            'is_anomaly': is_anomaly_list,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"이상 탐지 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/anomaly/detect/single', methods=['POST'])
def detect_anomaly_single():
    """
    단일 샘플 이상 탐지 API
    
    Request Body:
        {
            "feature": [feature_value1, feature_value2, ...]
        }
    
    Response:
        {
            "score": 0.75,
            "is_anomaly": true,
            "timestamp": "2024-01-01T00:00:00"
        }
    """
    try:
        if not _initialize_services():
            return jsonify({'error': '서비스 초기화 실패'}), 500
        
        if detector_manager is None:
            return jsonify({'error': '모델이 로드되지 않았습니다.'}), 400
        
        data = request.json
        if not data or 'feature' not in data:
            return jsonify({'error': 'feature 필드가 필요합니다.'}), 400
        
        feature = np.array(data['feature'])
        
        # 단일 샘플 예측
        score, is_anomaly = detector_manager.predict_single(feature)
        
        return jsonify({
            'score': float(score),
            'is_anomaly': bool(is_anomaly),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"단일 샘플 이상 탐지 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['POST'])
def ingest_logs():
    """
    로그 수집 및 파싱 API
    
    Request Body:
        {
            "logs": ["log_line1", "log_line2", ...],
            "log_type": "nginx" (선택사항, 기본값: nginx)
        }
    
    Response:
        {
            "processed": 100,
            "parsed": 95,
            "failed": 5,
            "timestamp": "2024-01-01T00:00:00"
        }
    """
    try:
        if not _initialize_services():
            return jsonify({'error': '서비스 초기화 실패'}), 500
        
        data = request.json
        if not data or 'logs' not in data:
            return jsonify({'error': 'logs 필드가 필요합니다.'}), 400
        
        logs = data.get('logs', [])
        log_type = data.get('log_type', 'nginx')
        
        if not logs:
            return jsonify({'error': 'logs 배열이 비어있습니다.'}), 400
        
        # 로그 파싱
        parsed_logs = []
        failed_count = 0
        
        for log_line in logs:
            parsed_log = nginx_parser.parse_line(log_line)
            if parsed_log:
                parsed_logs.append(parsed_log)
            else:
                failed_count += 1
        
        return jsonify({
            'processed': len(logs),
            'parsed': len(parsed_logs),
            'failed': failed_count,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"로그 수집 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs/features', methods=['POST'])
def extract_features():
    """
    로그에서 특징 추출 API
    
    Request Body:
        {
            "logs": [parsed_log1, parsed_log2, ...],
            "window_seconds": 60 (선택사항, 기본값: 60)
        }
    
    Response:
        {
            "features": [...],
            "window_count": 10,
            "timestamp": "2024-01-01T00:00:00"
        }
    """
    try:
        if not _initialize_services():
            return jsonify({'error': '서비스 초기화 실패'}), 500
        
        data = request.json
        if not data or 'logs' not in data:
            return jsonify({'error': 'logs 필드가 필요합니다.'}), 400
        
        logs = data.get('logs', [])
        window_seconds = data.get('window_seconds', 60)
        
        if not logs:
            return jsonify({'error': 'logs 배열이 비어있습니다.'}), 400
        
        # 특징 추출
        features_df = feature_engineer.extract_features(logs, window_seconds)
        
        # DataFrame을 딕셔너리 리스트로 변환
        features_list = features_df.to_dict('records')
        
        return jsonify({
            'features': features_list,
            'window_count': len(features_list),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"특징 추출 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """
    알림 조회 API
    
    Query Parameters:
        limit: 반환할 알림 개수 (기본값: 10)
    
    Response:
        {
            "alerts": [...],
            "count": 10
        }
    """
    try:
        limit = request.args.get('limit', 10, type=int)
        return jsonify({
            'alerts': alerts[-limit:] if len(alerts) > limit else alerts,
            'count': len(alerts)
        })
    except Exception as e:
        logger.error(f"알림 조회 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['POST'])
def create_alert():
    """
    알림 생성 API
    
    Request Body:
        {
            "type": "anomaly",
            "message": "이상 패턴 감지됨",
            "severity": "high"
        }
    
    Response:
        {
            "id": 1,
            "timestamp": "2024-01-01T00:00:00"
        }
    """
    try:
        data = request.json
        if not data:
            return jsonify({'error': '요청 본문이 필요합니다.'}), 400
        
        alert = {
            'id': len(alerts) + 1,
            'type': data.get('type', 'unknown'),
            'message': data.get('message', ''),
            'severity': data.get('severity', 'medium'),
            'timestamp': datetime.now().isoformat()
        }
        
        alerts.append(alert)
        
        return jsonify({
            'id': alert['id'],
            'timestamp': alert['timestamp']
        }), 201
    except Exception as e:
        logger.error(f"알림 생성 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/model/status', methods=['GET'])
def get_model_status():
    """
    모델 상태 조회 API
    
    Response:
        {
            "loaded": true,
            "model_type": "isolation_forest",
            "model_path": "models/isolation_forest"
        }
    """
    try:
        if detector_manager is None:
            return jsonify({
                'loaded': False,
                'model_type': None,
                'model_path': None
            })
        
        return jsonify({
            'loaded': True,
            'model_type': detector_manager.detector_type,
            'model_path': str(MODEL_ISOLATION_FOREST)
        })
    except Exception as e:
        logger.error(f"모델 상태 조회 API 오류: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 서비스 초기화
    _initialize_services()
    app.run(host='0.0.0.0', port=5000, debug=True)

