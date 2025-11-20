"""
REST API 엔드포인트
외부 시스템과의 통합을 위한 API 제공
"""
from flask import Flask, request, jsonify
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

app = Flask(__name__)

# 전역 상태 (실제로는 데이터베이스나 캐시 사용 권장)
anomaly_results = []
alerts = []


@app.route('/health', methods=['GET'])
def health_check():
    """헬스 체크 엔드포인트"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
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
            "is_anomaly": [bool1, bool2, ...]
        }
    """
    try:
        data = request.json
        features = data.get('features', [])
        
        # TODO: 실제 모델로 예측 수행
        # detector = AnomalyDetectorManager()
        # scores, is_anomaly = detector.predict(np.array(features))
        
        # 임시 응답
        scores = [0.5] * len(features)
        is_anomaly = [False] * len(features)
        
        return jsonify({
            'scores': scores,
            'is_anomaly': is_anomaly,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"이상 탐지 API 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/alerts', methods=['GET'])
def get_alerts():
    """알림 조회 API"""
    try:
        limit = request.args.get('limit', 10, type=int)
        return jsonify({
            'alerts': alerts[-limit:],
            'count': len(alerts)
        })
    except Exception as e:
        logger.error(f"알림 조회 API 오류: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/logs', methods=['POST'])
def ingest_logs():
    """
    로그 수집 API
    
    Request Body:
        {
            "logs": ["log_line1", "log_line2", ...]
        }
    """
    try:
        data = request.json
        logs = data.get('logs', [])
        
        # TODO: 실제 로그 파싱 및 처리
        # parser = NginxParser()
        # parsed_logs = parser.parse_batch(logs)
        
        return jsonify({
            'processed': len(logs),
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"로그 수집 API 오류: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

