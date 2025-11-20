"""
사이드바 레이아웃 컴포넌트
"""
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
from app.services.log_service import LogService
from app.services.model_service import ModelService


def render_sidebar(log_service: LogService, model_service: ModelService):
    """
    사이드바 렌더링
    
    Args:
        log_service: 로그 서비스 인스턴스
        model_service: 모델 서비스 인스턴스
    """
    with st.sidebar:
        st.title("설정")
        
        # 수집 모드 선택
        collection_mode = st.radio(
            "수집 모드",
            ["실시간", "배치"],
            index=0
        )
        
        # 수집 시작 버튼
        if st.button("수집 시작", type="primary"):
            mode = "realtime" if collection_mode == "실시간" else "batch"
            success, message = log_service.start_collection(mode)
            if success:
                st.success(message)
            else:
                st.error(message)
        
        # 수집 중지 버튼
        if st.button("수집 중지"):
            success, message = log_service.stop_collection()
            if success:
                st.success(message)
            else:
                st.warning(message)
        
        st.divider()
        
        # 모델 선택 및 로드
        st.subheader("모델 선택")
        
        model_choice = st.radio(
            "사용할 모델 선택",
            ["Isolation Forest (빠르고 안정적)", "PyTorch AutoEncoder (더 정확)"],
            index=0,
            help="Isolation Forest: 빠르고 안정적, PyTorch AutoEncoder: 더 정확한 이상 탐지"
        )
        
        # 모델 타입 결정
        if model_choice == "Isolation Forest (빠르고 안정적)":
            model_type = "isolation_forest"
            model_info = "빠른 학습 및 예측, 안정적"
        else:
            model_type = "pytorch_autoencoder"
            model_info = "복잡한 패턴 학습 가능, 더 정확"
        
        st.info(model_info)
        
        # 모델 로드 버튼
        col_load, col_status = st.columns([2, 1])
        with col_load:
            if st.button("모델 로드", type="primary", use_container_width=True):
                with st.spinner("모델 로드 중..."):
                    success, message = model_service.load_model(model_type)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
                        # 모델 파일이 없을 때 안내
                        if "찾을 수 없습니다" in message:
                            st.info("먼저 모델을 학습시켜주세요")
                            if model_type == "isolation_forest":
                                st.code("python scripts/train_isolation_forest.py --data data/raw_logs/nginx_access.log --output models/isolation_forest")
                            else:
                                st.code("python scripts/train_pytorch_autoencoder.py --data data/raw_logs/nginx_access.log --output models/pytorch_autoencoder")
        
        with col_status:
            if model_service.is_model_loaded():
                st.success("로드됨")
            else:
                st.info("대기 중")
        
        st.divider()
        
        # 통계
        st.subheader("통계")
        if log_service.is_running():
            stats = log_service.get_stats()
            st.metric("수집된 로그", stats['collected_count'])
            st.metric("파싱 성공률", f"{stats['parser']['success_rate']:.1f}%")
        else:
            st.info("수집기를 시작하세요")

