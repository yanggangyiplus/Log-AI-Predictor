#!/bin/bash
# Streamlit 대시보드 실행 스크립트

# 프로젝트 루트로 이동
cd "$(dirname "$0")/.."

# Streamlit 실행
streamlit run app/dashboard.py --server.port 8501 --server.address 0.0.0.0

