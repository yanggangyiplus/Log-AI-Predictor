"""
모델 관리 서비스
모델 로딩 및 관리 비즈니스 로직
"""
import sys
from pathlib import Path
from typing import Optional, Tuple
import logging

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.anomaly.detector_manager import AnomalyDetectorManager
    MODULES_LOADED = True
except ImportError as e:
    MODULES_LOADED = False
    logging.error(f"모듈 로드 실패: {e}")

from app.utils.constants import (
    MODEL_ISOLATION_FOREST,
    MODEL_PYTORCH_AUTOENCODER,
    SESSION_KEY_DETECTOR,
    SESSION_KEY_CURRENT_MODEL
)

logger = logging.getLogger(__name__)


class ModelService:
    """모델 관리 서비스 클래스"""
    
    def __init__(self, session_state):
        """
        초기화
        
        Args:
            session_state: Streamlit session_state 객체
        """
        self.session_state = session_state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """세션 상태 초기화"""
        if SESSION_KEY_DETECTOR not in self.session_state:
            self.session_state[SESSION_KEY_DETECTOR] = None
        if SESSION_KEY_CURRENT_MODEL not in self.session_state:
            self.session_state[SESSION_KEY_CURRENT_MODEL] = None
    
    def load_model(self, model_type: str = "isolation_forest") -> Tuple[bool, str]:
        """
        모델 로드
        
        Args:
            model_type: 모델 타입 ("isolation_forest" 또는 "pytorch_autoencoder")
            
        Returns:
            (성공 여부, 메시지)
        """
        if not MODULES_LOADED:
            return False, "모듈 로드 실패"
        
        try:
            # 모델 경로 결정
            if model_type == "isolation_forest":
                model_path = str(MODEL_ISOLATION_FOREST)
            elif model_type == "pytorch_autoencoder":
                model_path = str(MODEL_PYTORCH_AUTOENCODER)
            else:
                return False, f"지원하지 않는 모델 타입: {model_type}"
            
            # 모델 파일 존재 확인
            model_path_obj = Path(model_path)
            if not model_path_obj.exists():
                # PyTorch 모델인 경우 추가 확장자 확인
                if model_type == "pytorch_autoencoder":
                    pytorch_model_file = model_path_obj.parent / f"{model_path_obj.name}_pytorch.pth"
                    if not pytorch_model_file.exists():
                        return False, f"모델 파일을 찾을 수 없습니다: {pytorch_model_file}"
                else:
                    return False, f"모델 파일을 찾을 수 없습니다: {model_path}"
            
            # 모델 로드
            detector = AnomalyDetectorManager()
            detector.load_model(model_path)
            
            self.session_state[SESSION_KEY_DETECTOR] = detector
            self.session_state[SESSION_KEY_CURRENT_MODEL] = model_path
            
            return True, "모델 로드 완료"
        except FileNotFoundError as e:
            logger.error(f"모델 파일 없음: {e}")
            return False, f"모델 파일을 찾을 수 없습니다: {model_path}"
        except Exception as e:
            logger.error(f"모델 로드 실패: {e}", exc_info=True)
            return False, f"모델 로드 실패: {e}"
    
    def get_current_model(self) -> Optional[object]:
        """
        현재 로드된 모델 반환
        
        Returns:
            모델 객체 또는 None
        """
        return self.session_state.get(SESSION_KEY_DETECTOR)
    
    def get_current_model_path(self) -> Optional[str]:
        """
        현재 모델 경로 반환
        
        Returns:
            모델 경로 또는 None
        """
        return self.session_state.get(SESSION_KEY_CURRENT_MODEL)
    
    def get_model_name(self) -> str:
        """
        현재 모델 이름 반환
        
        Returns:
            모델 이름 문자열
        """
        model_path = self.get_current_model_path()
        if not model_path:
            return "없음"
        
        if "isolation" in model_path.lower():
            return "Isolation Forest"
        elif "pytorch" in model_path.lower():
            return "PyTorch AutoEncoder"
        else:
            return "알 수 없음"
    
    def is_model_loaded(self) -> bool:
        """
        모델이 로드되었는지 확인
        
        Returns:
            로드되었으면 True
        """
        return self.session_state.get(SESSION_KEY_DETECTOR) is not None

