"""
프로젝트 경로 설정 유틸리티
공통 경로 초기화 로직
"""
import sys
from pathlib import Path


def setup_project_root(current_file: str) -> Path:
    """
    프로젝트 루트를 sys.path에 추가
    
    Args:
        current_file: 현재 파일 경로 (__file__)
        
    Returns:
        프로젝트 루트 Path 객체
    """
    # 현재 파일의 경로를 절대 경로로 변환
    current_path = Path(current_file).resolve()
    
    # app/web/main.py 또는 app/api/api.py 등에서 호출되는 경우를 고려
    # 프로젝트 루트는 app/ 폴더의 부모 디렉토리
    if 'app/web' in str(current_path) or 'app/api' in str(current_path):
        # app/web/main.py -> app/web/ -> app/ -> 프로젝트 루트
        project_root = current_path.parent.parent.parent
    elif 'app' in str(current_path):
        # app/utils/path_setup.py -> app/utils/ -> app/ -> 프로젝트 루트
        project_root = current_path.parent.parent.parent
    else:
        # 기타 경우: 상위 디렉토리로 이동하여 프로젝트 루트 찾기
        project_root = current_path.parent
        while project_root.name != 'Log-AI-Predictor' and project_root.parent != project_root:
            project_root = project_root.parent
    
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    return project_root

