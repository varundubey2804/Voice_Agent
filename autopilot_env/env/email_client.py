import imaplib
import email
from email.header import decode_header
from typing import List, Dict, Any
import uuid

def clean_text(text: str) -> str:
    # Basic cleanup to remove excessive newlines/spaces
    if not text:
        return ""
    lines = text.split('\n')
    cleaned = [line.strip() for line in lines if line.strip()]
    return "\n".join(cleaned)[:500] # truncate to avoid massive bodies

def fetch_recent_emails(username: str, password: str, server: str = "imap.gmail.com", count: int = 5) -> List[Dict[str, Any]]:
    # Connect to the server
    try:
        mail = imaplib.IMAP4_SSL(server)
        mail.login(username, password)
    except Exception as e:
        raise Exception(f"Failed to connect or login: {e}")

    try:
        mail.select("inbox")
        status, messages = mail.search(None, "ALL")

        email_ids = messages[0].split()
        if not email_ids:
            return []

        # Get the latest `count` emails
        latest_email_ids = email_ids[-count:]

        parsed_emails = []
        for e_id in reversed(latest_email_ids): # Newest first
            res, msg_data = mail.fetch(e_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])

                    # Decode subject
                    subject, encoding = decode_header(msg["Subject"])[0]
                    if isinstance(subject, bytes):
                        try:
                            subject = subject.decode(encoding if encoding else "utf-8")
                        except:
                            subject = subject.decode("utf-8", errors="ignore")

                    # Get sender
                    sender = msg.get("From")

                    # Get body
                    body = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            content_type = part.get_content_type()
                            content_disposition = str(part.get("Content-Disposition"))

                            if content_type == "text/plain" and "attachment" not in content_disposition:
                                try:
                                    body = part.get_payload(decode=True).decode()
                                    break # just get the first text part
                                except:
                                    pass
                    else:
                        try:
                            body = msg.get_payload(decode=True).decode()
                        except:
                            body = msg.get_payload()

                    parsed_emails.append({
                        "id": f"real_{uuid.uuid4().hex[:6]}",
                        "sender": sender or "Unknown",
                        "subject": subject or "No Subject",
                        "body": clean_text(body),
                        "is_read": False,
                        "is_archived": False,
                        "thread_id": f"thr_{uuid.uuid4().hex[:4]}"
                    })

        return parsed_emails

    finally:
        try:
            mail.close()
            mail.logout()
        except:
            pass
