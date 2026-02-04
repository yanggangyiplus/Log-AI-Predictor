"""
알림 서비스
외부 알림 시스템 연동 (Email, Webhook 등)
"""
import sys
from pathlib import Path
from typing import Dict, Optional, List
import logging
import requests
import json
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.utils import formatdate

# 프로젝트 루트를 경로에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from app.utils.cache import load_alert_config
from app.utils.env_config import get_email_config, get_webhook_url

logger = logging.getLogger(__name__)


class NotificationService:
    """알림 서비스 클래스"""
    
    def __init__(self):
        """초기화"""
        self.config = load_alert_config()
        self.channels = self.config.get('channels', {})
        # 환경 변수에서 Email 설정 로드
        self.email_env_config = get_email_config()
        self.webhook_url = get_webhook_url()
    
    def send_email_notification(self, message: str, severity: str = "medium",
                               details: Optional[Dict] = None,
                               subject: Optional[str] = None) -> bool:
        """
        Email 알림 전송
        
        Args:
            message: 알림 메시지
            severity: 심각도 (low, medium, high)
            details: 추가 상세 정보
            subject: 이메일 제목 (선택사항)
            
        Returns:
            전송 성공 여부
        """
        email_config = self.channels.get('email', {})
        
        if not email_config.get('enabled', False):
            return False
        
        # 환경 변수 우선, 없으면 설정 파일 사용
        smtp_server = self.email_env_config.get('smtp_server') or email_config.get('smtp_server', '')
        smtp_port = self.email_env_config.get('smtp_port') or email_config.get('smtp_port', 587)
        smtp_user = self.email_env_config.get('smtp_user') or email_config.get('smtp_user', '')
        smtp_password = self.email_env_config.get('smtp_password') or email_config.get('smtp_password', '')
        from_email = self.email_env_config.get('from_email') or email_config.get('from_email', smtp_user)
        to_emails = self.email_env_config.get('to_emails') or email_config.get('to_emails', [])
        
        if not smtp_server or not smtp_user or not smtp_password:
            logger.warning("Email 설정이 완전하지 않습니다. (smtp_server, smtp_user, smtp_password 필요)")
            return False
        
        if not to_emails:
            logger.warning("수신자 이메일 주소가 설정되지 않았습니다.")
            return False
        
        # 이메일 제목 설정
        if not subject:
            severity_kr = {'low': '낮음', 'medium': '보통', 'high': '높음'}.get(severity, '알 수 없음')
            subject = f"[로그 이상 탐지] {severity_kr} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        try:
            # 이메일 메시지 생성
            msg = MIMEMultipart('alternative')
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails) if isinstance(to_emails, list) else to_emails
            msg['Subject'] = subject
            msg['Date'] = formatdate(localtime=True)
            
            # HTML 본문 생성
            html_body = self._create_email_html(message, severity, details)
            text_body = self._create_email_text(message, severity, details)
            
            # 본문 추가
            msg.attach(MIMEText(text_body, 'plain', 'utf-8'))
            msg.attach(MIMEText(html_body, 'html', 'utf-8'))
            
            # SMTP 서버 연결 및 전송
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()  # TLS 암호화
                server.login(smtp_user, smtp_password)
                
                # 수신자 리스트 처리
                recipients = to_emails if isinstance(to_emails, list) else [to_emails]
                server.sendmail(from_email, recipients, msg.as_string())
            
            logger.info(f"Email 알림 전송 성공: {recipients}")
            return True
            
        except smtplib.SMTPAuthenticationError as e:
            logger.error(f"Email 인증 실패: {e}")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"Email 전송 실패: {e}")
            return False
        except Exception as e:
            logger.error(f"Email 알림 전송 중 오류: {e}", exc_info=True)
            return False
    
    def _create_email_html(self, message: str, severity: str, details: Optional[Dict] = None) -> str:
        """
        HTML 이메일 본문 생성
        
        Args:
            message: 알림 메시지
            severity: 심각도
            details: 추가 상세 정보
            
        Returns:
            HTML 문자열
        """
        # 심각도에 따른 색상 및 아이콘
        severity_config = {
            'low': {'color': '#28a745', 'icon': 'ℹ️', 'label': '낮음'},
            'medium': {'color': '#ffc107', 'icon': '⚠️', 'label': '보통'},
            'high': {'color': '#dc3545', 'icon': '🚨', 'label': '높음'}
        }
        config = severity_config.get(severity, {'color': '#6c757d', 'icon': '📢', 'label': '알 수 없음'})
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: {config['color']}; color: white; padding: 20px; border-radius: 5px 5px 0 0; }}
                .content {{ background-color: #f8f9fa; padding: 20px; border-radius: 0 0 5px 5px; }}
                .severity-badge {{ display: inline-block; padding: 5px 10px; background-color: {config['color']}; color: white; border-radius: 3px; font-weight: bold; }}
                .details {{ margin-top: 20px; }}
                .detail-item {{ margin: 10px 0; padding: 10px; background-color: white; border-left: 3px solid {config['color']}; }}
                .footer {{ margin-top: 20px; padding-top: 20px; border-top: 1px solid #ddd; font-size: 12px; color: #666; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{config['icon']} 로그 이상 탐지 알림</h2>
                </div>
                <div class="content">
                    <p><strong>심각도:</strong> <span class="severity-badge">{config['label']}</span></p>
                    <p><strong>시간:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    <p><strong>메시지:</strong></p>
                    <p style="background-color: white; padding: 15px; border-radius: 5px;">{message}</p>
        """
        
        # 상세 정보 추가
        if details:
            html += '<div class="details"><h3>상세 정보</h3>'
            for key, value in details.items():
                html += f'<div class="detail-item"><strong>{key}:</strong> {value}</div>'
            html += '</div>'
        
        html += """
                    <div class="footer">
                        <p>이 메일은 Log Pattern Analyzer & Anomaly Predictor에서 자동으로 전송되었습니다.</p>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _create_email_text(self, message: str, severity: str, details: Optional[Dict] = None) -> str:
        """
        텍스트 이메일 본문 생성
        
        Args:
            message: 알림 메시지
            severity: 심각도
            details: 추가 상세 정보
            
        Returns:
            텍스트 문자열
        """
        severity_kr = {'low': '낮음', 'medium': '보통', 'high': '높음'}.get(severity, '알 수 없음')
        
        text = f"""
로그 이상 탐지 알림
==================

심각도: {severity_kr}
시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

메시지:
{message}
"""
        
        if details:
            text += "\n상세 정보:\n"
            for key, value in details.items():
                text += f"  - {key}: {value}\n"
        
        text += "\n이 메일은 Log Pattern Analyzer & Anomaly Predictor에서 자동으로 전송되었습니다."
        
        return text
    
    def send_webhook_notification(self, message: str, severity: str = "medium",
                                  details: Optional[Dict] = None) -> bool:
        """
        일반 웹훅 알림 전송
        
        Args:
            message: 알림 메시지
            severity: 심각도
            details: 추가 상세 정보
            
        Returns:
            전송 성공 여부
        """
        webhook_config = self.channels.get('webhook', {})
        
        if not webhook_config.get('enabled', False):
            return False
        
        # 환경 변수 우선, 없으면 설정 파일 사용
        webhook_url = self.webhook_url or webhook_config.get('url', '')
        if not webhook_url:
            logger.warning("웹훅 URL이 설정되지 않았습니다.")
            return False
        
        payload = {
            "message": message,
            "severity": severity,
            "timestamp": datetime.now().isoformat(),
            "details": details or {}
        }
        
        try:
            response = requests.post(
                webhook_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code in [200, 201, 204]:
                logger.info("웹훅 알림 전송 성공")
                return True
            else:
                logger.error(f"웹훅 알림 전송 실패: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"웹훅 알림 전송 중 오류: {e}", exc_info=True)
            return False
    
    def send_notification(self, alert_type: str, message: str,
                         severity: str = "medium", details: Optional[Dict] = None) -> Dict:
        """
        알림 전송 (모든 채널)
        
        Args:
            alert_type: 알림 타입 (anomaly, error_rate, response_time 등)
            message: 알림 메시지
            severity: 심각도
            details: 추가 상세 정보
            
        Returns:
            전송 결과 딕셔너리
        """
        results = {
            'email': False,
            'webhook': False
        }
        
        # Email 알림 전송
        if self.channels.get('email', {}).get('enabled', False):
            results['email'] = self.send_email_notification(message, severity, details)
        
        # 웹훅 알림 전송
        if self.channels.get('webhook', {}).get('enabled', False):
            results['webhook'] = self.send_webhook_notification(message, severity, details)
        
        return results
