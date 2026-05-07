"""Email service for handling subscriptions and sending summaries."""

import email
import imaplib
import logging
import os
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.utils import parseaddr
from typing import List

try:
    import markdown
except ImportError:
    markdown = None

from ..models import EmailConfig

logger = logging.getLogger(__name__)


class EmailManager:
    """Manages email subscriptions and sending summaries."""

    def __init__(self, config: EmailConfig, console=None):
        self.config = config
        self.pwd = os.getenv(self.config.password_env)
        if console is None:
            try:
                from rich.console import Console
                self.console = Console()
            except ImportError:
                class DummyConsole:
                    def print(self, *args, **kwargs):
                        print(*args, **kwargs)
                self.console = DummyConsole()
        else:
            self.console = console

        if not self.pwd and self.config.enabled:
            logger.warning(
                f"Environment variable {self.config.password_env} not set. Email features may fail."
            )
            self.console.print(f"[yellow]Warning: Environment variable {self.config.password_env} not set. Email features may fail.[/yellow]")

    def check_subscriptions(self, storage_manager):
        """Checks inbox for subscription requests and updates subscriber list."""
        if not self.config.enabled:
            return

        mail = None
        mailbox_selected = False
        try:
            mail = imaplib.IMAP4_SSL(self.config.imap_server, self.config.imap_port)
            mail.login(self.config.email_address, self.pwd)
            self._identify_imap_client(mail)
            self._select_inbox(mail)
            mailbox_selected = True

            keyword = self.config.subscribe_keyword
            # search_crit = f'(SUBJECT "{keyword}")'
            search_crit = f'(UNSEEN)'

            # status, messages = mail.search(None, search_crit)
            status, messages = mail.search(None, search_crit)

            if status == "OK" and messages[0]:
                email_ids = messages[0].split()
                subscribers = storage_manager.load_subscribers()

                for e_id in email_ids:
                    _, msg_data = mail.fetch(e_id, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])

                            subject = str(msg.get("Subject") or "").strip()
                            if subject.upper() != keyword.upper():
                                continue

                            sender = msg.get("From")

                            if sender:
                                _, email_addr = parseaddr(sender)
                                if email_addr and "@" in email_addr:
                                    if "noreply" in email_addr.lower() or "no-reply" in email_addr.lower():
                                        continue

                                    if email_addr not in subscribers:
                                        storage_manager.add_subscriber(email_addr)
                                        subscribers = storage_manager.load_subscribers()
                                        self._send_reply(
                                            email_addr,
                                            "Subscribed to Horizon",
                                            "You have been successfully subscribed to Horizon daily summaries.",
                                        )
                                        logger.info(f"Added subscriber: {email_addr}")
                                    else:
                                        logger.info(f"Already subscribed: {email_addr}")

            unsub_keyword = self.config.unsubscribe_keyword
            search_crit_unsub = f'(UNSEEN SUBJECT "{unsub_keyword}")'

            status, messages = mail.search(None, search_crit_unsub)

            if status == "OK" and messages[0]:
                email_ids = messages[0].split()
                subscribers = storage_manager.load_subscribers()

                for e_id in email_ids:
                    _, msg_data = mail.fetch(e_id, "(RFC822)")
                    for response_part in msg_data:
                        if isinstance(response_part, tuple):
                            msg = email.message_from_bytes(response_part[1])

                            subject = str(msg.get("Subject") or "").strip()
                            if subject.upper() != unsub_keyword.upper():
                                continue

                            sender = msg.get("From")

                            if sender:
                                _, email_addr = parseaddr(sender)
                                if email_addr and "@" in email_addr:
                                    if "noreply" in email_addr.lower() or "no-reply" in email_addr.lower():
                                        continue

                                    if email_addr in subscribers:
                                        storage_manager.remove_subscriber(email_addr)
                                        subscribers = storage_manager.load_subscribers()
                                        self._send_reply(
                                            email_addr,
                                            "Unsubscribed from Horizon",
                                            "You have been successfully unsubscribed from Horizon daily summaries.",
                                        )
                                        logger.info(f"Removed subscriber: {email_addr}")
                                    else:
                                        logger.info(f"Not subscribed: {email_addr}")

        except Exception as e:
            logger.error(f"Error checking subscriptions: {e}")
        finally:
            if mail:
                if mailbox_selected:
                    try:
                        mail.close()
                    except imaplib.IMAP4.error as e:
                        logger.debug(f"IMAP close failed: {e}")
                try:
                    mail.logout()
                except imaplib.IMAP4.error as e:
                    logger.debug(f"IMAP logout failed: {e}")

    def send_daily_summary(
        self, summary_md: str, subject: str, subscribers: List[str]
    ):
        """Sends the daily summary to all subscribers."""
        if not self.config.enabled or not subscribers:
            return

        html_content = (
            markdown.markdown(summary_md)
            if markdown
            else f"<pre>{summary_md}</pre>"
        )

        html_body = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                code {{ background-color: #f4f4f4; padding: 2px 5px; border-radius: 3px; font-family: monospace; }}
                pre {{ background-color: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
                blockquote {{ border-left: 4px solid #ddd; padding-left: 15px; color: #777; }}
                .footer {{ margin-top: 40px; font-size: 12px; color: #888; text-align: center; border-top: 1px solid #eee; padding-top: 20px; }}
            </style>
        </head>
        <body>
            {html_content}
            <div class="footer">
                <p>Sent by {self.config.sender_name}</p>
                <p>To unsubscribe, please reply with "{self.config.unsubscribe_keyword}"</p>
            </div>
        </body>
        </html>
        """

        try:
            with smtplib.SMTP_SSL(
                self.config.smtp_server, self.config.smtp_port
            ) as server:
                server.login(self.config.email_address, self.pwd)

                for subscriber in subscribers:
                    msg = MIMEMultipart("alternative")
                    msg["Subject"] = subject
                    msg["From"] = f"{self.config.sender_name} <{self.config.email_address}>"
                    msg["To"] = subscriber

                    text_part = MIMEText(summary_md, "plain")
                    html_part = MIMEText(html_body, "html")

                    msg.attach(text_part)
                    msg.attach(html_part)

                    try:
                        server.send_message(msg)
                        logger.info(f"Sent summary to {subscriber}")
                    except Exception as e:
                        logger.error(f"Failed to send to {subscriber}: {e}")

        except Exception as e:
            logger.error(f"SMTP Error: {e}")

    def _send_reply(self, to_email: str, subject: str, body: str):
        """Helper to send a simple reply."""
        try:
            with smtplib.SMTP_SSL(
                self.config.smtp_server, self.config.smtp_port
            ) as server:
                server.login(self.config.email_address, self.pwd)

                msg = MIMEText(body)
                msg["Subject"] = subject
                msg["From"] = f"{self.config.sender_name} <{self.config.email_address}>"
                msg["To"] = to_email

                server.send_message(msg)
        except Exception as e:
            logger.error(f"Failed to send reply to {to_email}: {e}")

    def _identify_imap_client(self, mail: imaplib.IMAP4_SSL):
        """Send IMAP ID for providers such as NetEase that reject unknown clients."""
        previous_id_states = imaplib.Commands.get("ID")
        imaplib.Commands["ID"] = ("AUTH", "SELECTED")

        args = (
            "name",
            "Horizon",
            "contact",
            self.config.email_address,
            "version",
            "1.0.0",
            "vendor",
            "Horizon",
        )

        try:
            status, data = mail._simple_command("ID", '("' + '" "'.join(args) + '")')
            if status != "OK":
                logger.info(f"IMAP ID command returned {status}: {data}")
        except imaplib.IMAP4.error as e:
            logger.error(f"IMAP ID command is not supported by this server: {e}")
        finally:
            if previous_id_states is None:
                imaplib.Commands.pop("ID", None)
            else:
                imaplib.Commands["ID"] = previous_id_states

    def _select_inbox(self, mail: imaplib.IMAP4_SSL):
        """Select INBOX with fallbacks for IMAP servers with strict parsing."""
        errors = []

        for mailbox in ("INBOX", "Inbox", '"INBOX"', '"Inbox"'):
            if self._try_select_mailbox(mail, mailbox, errors):
                return

        status, mailboxes = mail.list()
        if status == "OK":
            for mailbox_data in mailboxes:
                mailbox = self._parse_mailbox_name(mailbox_data)
                if mailbox and mailbox.lower() == "inbox":
                    if self._try_select_mailbox(mail, mailbox, errors):
                        return
                    if self._try_select_mailbox(mail, f'"{mailbox}"', errors):
                        return
        else:
            errors.append(f"LIST returned {status}: {mailboxes}")

        raise imaplib.IMAP4.error(
            "Unable to select INBOX. Tried: " + "; ".join(errors)
        )

    def _try_select_mailbox(
        self, mail: imaplib.IMAP4_SSL, mailbox: str, errors: list[str]
    ):
        try:
            status, data = mail.select(mailbox)
        except imaplib.IMAP4.error as e:
            errors.append(f"{mailbox}: {e}")
            return False

        if status == "OK":
            return True

        errors.append(f"{mailbox}: {status} {data}")
        return False

    def _parse_mailbox_name(self, mailbox_data):
        if isinstance(mailbox_data, bytes):
            text = mailbox_data.decode("utf-8", errors="replace")
        else:
            text = str(mailbox_data)

        match = re.search(r' (?:"((?:[^"\\]|\\.)*)"|([^ ]+))$', text)
        if not match:
            return None

        mailbox = match.group(1) if match.group(1) is not None else match.group(2)
        return mailbox.replace(r"\"", '"').replace(r"\\", "\\")
