from .memory import append_message
from ..core.email import send_email
from ..models.schemas import BookingDetails

def save_booking(details: BookingDetails) -> None:
    append_message(details.email, "system", f"BOOKED {details.datetime_iso} : {details.notes or ''}")

def send_confirmation(details: BookingDetails) -> None:
    subject = "Interview Booking Confirmed"
    html = f"""
    <h3>Hi {details.name},</h3>
    <p>Your interview is confirmed for <b>{details.datetime_iso}</b>.</p>
    <p>Notes: {details.notes or '-' }.</p>
    <p>— RAG Bot</p>
    """
    send_email([details.email], subject, html)
