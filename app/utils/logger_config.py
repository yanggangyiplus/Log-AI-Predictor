"""
로깅 설정 유틸리티
애플리케이션 전역 로깅 설정
"""
import logging
import sys


# 로깅 설정 (한 번만 실행되도록 체크)
_logging_configured = False


def get_app_logger(name: str = None) -> logging.Logger:
    """
    애플리케이션 로거 반환
    
    Args:
        name: 로거 이름 (기본값: 호출 모듈명)
        
    Returns:
        설정된 Logger 객체
    """
    global _logging_configured
    
    if not _logging_configured:
        # 기본 로깅 설정 (한 번만 실행)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        _logging_configured = True
    
    if name is None:
        # 호출한 모듈의 이름 자동 감지
        import inspect
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get('__name__', 'app')
    
    return logging.getLogger(name)


# 기본 로거 (app.main 등에서 사용)
app_logger = get_app_logger('app')

