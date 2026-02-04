#!/usr/bin/env python3
"""
Flask API 서버 실행 스크립트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.api.api import app
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


if __name__ == '__main__':
    logger.info("Flask API 서버 시작...")
    logger.info("API 문서: http://localhost:5000/health")
    logger.info("API 엔드포인트: http://localhost:5000/api/")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
