"""
Nginx 로그 파서 모듈
Nginx access log를 파싱하여 구조화된 데이터로 변환
"""
import re
from datetime import datetime
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)


class NginxParser:
    """Nginx access log 파서 클래스"""
    
    # Nginx 로그 패턴 (표준 combined format)
    LOG_PATTERN = re.compile(
        r'(?P<ip>\S+) - (?P<remote_user>\S+) \[(?P<time_local>[^\]]+)\] '
        r'"(?P<method>\S+) (?P<url>\S+) (?P<protocol>\S+)" '
        r'(?P<status>\d+) (?P<body_bytes_sent>\d+) '
        r'"(?P<http_referer>[^"]*)" "(?P<http_user_agent>[^"]*)"'
    )
    
    def __init__(self):
        """파서 초기화"""
        self.parsed_count = 0
        self.error_count = 0
    
    def parse_line(self, line: str) -> Optional[Dict]:
        """
        단일 로그 라인 파싱
        
        Args:
            line: 로그 라인 문자열
            
        Returns:
            파싱된 로그 딕셔너리 또는 None (파싱 실패 시)
        """
        try:
            match = self.LOG_PATTERN.match(line.strip())
            if not match:
                self.error_count += 1
                logger.warning(f"로그 파싱 실패: {line[:100]}")
                return None
            
            groups = match.groupdict()
            
            # 타임스탬프 파싱
            timestamp = self._parse_timestamp(groups['time_local'])
            
            # URL에서 경로 추출
            url_path = self._extract_path(groups['url'])
            
            # 응답 시간 추출 (로그에 포함된 경우)
            response_time = self._extract_response_time(line)
            
            parsed_log = {
                'ip': groups['ip'],
                'remote_user': groups['remote_user'] if groups['remote_user'] != '-' else None,
                'timestamp': timestamp,
                'method': groups['method'],
                'url': groups['url'],
                'url_path': url_path,
                'protocol': groups['protocol'],
                'status_code': int(groups['status']),
                'body_bytes_sent': int(groups['body_bytes_sent']),
                'http_referer': groups['http_referer'] if groups['http_referer'] != '-' else None,
                'user_agent': groups['http_user_agent'],
                'response_time': response_time,
                'raw_line': line
            }
            
            self.parsed_count += 1
            return parsed_log
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"로그 파싱 중 오류 발생: {e}, 라인: {line[:100]}")
            return None
    
    def parse_batch(self, lines: List[str]) -> List[Dict]:
        """
        여러 로그 라인 일괄 파싱
        
        Args:
            lines: 로그 라인 리스트
            
        Returns:
            파싱된 로그 딕셔너리 리스트
        """
        parsed_logs = []
        for line in lines:
            parsed = self.parse_line(line)
            if parsed:
                parsed_logs.append(parsed)
        return parsed_logs
    
    def _parse_timestamp(self, time_str: str) -> datetime:
        """
        타임스탬프 문자열을 datetime 객체로 변환
        
        Args:
            time_str: 타임스탬프 문자열 (예: "12/Oct/2024:06:25:24 +0000")
            
        Returns:
            datetime 객체
        """
        try:
            # Nginx 로그 형식: 12/Oct/2024:06:25:24 +0000
            dt = datetime.strptime(time_str, '%d/%b/%Y:%H:%M:%S %z')
            return dt
        except Exception as e:
            logger.warning(f"타임스탬프 파싱 실패: {time_str}, 오류: {e}")
            return datetime.now()
    
    def _extract_path(self, url: str) -> str:
        """
        URL에서 경로 부분만 추출
        
        Args:
            url: 전체 URL 문자열
            
        Returns:
            경로 문자열
        """
        try:
            # 쿼리 파라미터 제거
            path = url.split('?')[0]
            return path
        except:
            return url
    
    def _extract_response_time(self, line: str) -> Optional[float]:
        """
        로그 라인에서 응답 시간 추출 (선택사항)
        일부 Nginx 설정에서는 $request_time 변수를 사용
        
        Args:
            line: 로그 라인
            
        Returns:
            응답 시간 (초) 또는 None
        """
        # request_time이 포함된 경우 추출 시도
        # 예: "GET /api/login HTTP/1.1" 500 234 "-" "Mozilla/5.0" 0.234
        try:
            parts = line.split()
            if len(parts) > 10:
                # 마지막 부분이 숫자면 응답 시간으로 간주
                last_part = parts[-1]
                if last_part.replace('.', '').isdigit():
                    return float(last_part)
        except:
            pass
        return None
    
    def get_stats(self) -> Dict:
        """
        파싱 통계 반환
        
        Returns:
            통계 딕셔너리
        """
        total = self.parsed_count + self.error_count
        success_rate = (self.parsed_count / total * 100) if total > 0 else 0
        
        return {
            'parsed_count': self.parsed_count,
            'error_count': self.error_count,
            'total_count': total,
            'success_rate': success_rate
        }

