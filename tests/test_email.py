"""Unit tests for email subscription IMAP helpers."""

import imaplib

from src.models import EmailConfig
from src.services.email import EmailManager


def _manager():
    config = EmailConfig(
        enabled=True,
        imap_server="imap.example.com",
        smtp_server="smtp.example.com",
        email_address="horizon@example.com",
    )
    return EmailManager(config)


class FakeMail:
    def __init__(self):
        self.id_args = None
        self.selected_mailboxes = []

    def _simple_command(self, command, args):
        self.id_args = (command, args)
        return "OK", [b'("name" "Horizon")']

    def select(self, mailbox):
        self.selected_mailboxes.append(mailbox)
        if mailbox == "Inbox":
            return "OK", [b"0"]
        return "NO", [b"Mailbox does not exist"]

    def list(self):
        return "OK", [b'(\\HasNoChildren) "/" "Inbox"']


def test_identify_imap_client_registers_id_command_temporarily():
    manager = _manager()
    mail = FakeMail()
    original_id_command = imaplib.Commands.get("ID")

    manager._identify_imap_client(mail)

    assert mail.id_args is not None
    assert mail.id_args[0] == "ID"
    assert '"name" "Horizon"' in mail.id_args[1]
    assert imaplib.Commands.get("ID") == original_id_command


def test_select_inbox_falls_back_to_provider_mailbox_name():
    manager = _manager()
    mail = FakeMail()

    manager._select_inbox(mail)

    assert mail.selected_mailboxes == ["INBOX", "Inbox"]


def test_select_inbox_reports_tried_mailboxes_when_all_fail():
    manager = _manager()
    mail = FakeMail()
    mail.select = lambda mailbox: ("NO", [b"nope"])

    try:
        manager._select_inbox(mail)
    except imaplib.IMAP4.error as exc:
        message = str(exc)
    else:
        raise AssertionError("Expected mailbox selection to fail")

    assert "Unable to select INBOX" in message
    assert "INBOX" in message
    assert "Inbox" in message
