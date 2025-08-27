from pydantic import EmailStr
from typing import Sequence
import smtplib
from email.mime.text import MIMEText
from pydantic import EmailStr
from .config import get_settings

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail
except Exception:
    SendGridAPIClient = None
    Mail = None

settings = get_settings()

def send_email(to: Sequence[EmailStr], subject: str, html: str) -> None:
    if settings.SENDGRID_API_KEY and SendGridAPIClient is not None:
        message = Mail(from_email=settings.EMAIL_SENDER, to_emails=list(to), subject=subject, html_content=html)
        sg = SendGridAPIClient(settings.SENDGRID_API_KEY)
        sg.send(message)
        return
    # Fallback: SMTP
    if not all([settings.SMTP_HOST, settings.SMTP_USERNAME, settings.SMTP_PASSWORD]):
        raise RuntimeError("Email not configured (SENDGRID_API_KEY or SMTP_* required)")
    msg = MIMEText(html, "html")
    msg["Subject"] = subject
    msg["From"] = settings.EMAIL_SENDER
    msg["To"] = ",".join(to)
    with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
        server.starttls()
        server.login(settings.SMTP_USERNAME, settings.SMTP_PASSWORD)
        server.sendmail(settings.EMAIL_SENDER, list(to), msg.as_string())
