"""Transactional email — address verification and password reset.

Resend is the configured provider, kept behind a single ``send()`` so it can be
swapped without touching the endpoints. Sending is **best-effort**: a provider
outage must never turn a signup into a 500, so failures are logged and reported
back as False for the caller to surface a "resend" affordance.

With no RESEND_API_KEY set (local development) nothing is sent — the link is
logged instead, so the whole verify/reset flow is still walkable offline.
"""

from __future__ import annotations

import logging

from app.config import settings

log = logging.getLogger(__name__)

RESEND_API_KEY = settings.resend_api_key
EMAIL_FROM = settings.email_from


def send(to: str, subject: str, html: str, link: str | None = None) -> bool:
    """True if handed to the provider. Never raises."""
    if not RESEND_API_KEY:
        log.warning("EMAIL NOT SENT (no RESEND_API_KEY). To=%s subject=%r link=%s",
                    to, subject, link or "-")
        return False
    try:
        import resend

        resend.api_key = RESEND_API_KEY
        resend.Emails.send({"from": EMAIL_FROM, "to": [to],
                            "subject": subject, "html": html})
        return True
    except Exception:  # noqa: BLE001 - email must never break the request
        log.exception("Failed to send %r to %s", subject, to)
        return False


# --------------------------------------------------------------------------- #
# Templates — plain, inlined CSS in the app's dark/teal palette. Email clients
# strip <style> blocks, so every rule has to live on the element.
# --------------------------------------------------------------------------- #
_WRAP = (
    "<div style=\"background:#0f1720;color:#e6edf3;font-family:system-ui,"
    "-apple-system,Segoe UI,Roboto,sans-serif;padding:32px\">"
    "<div style=\"max-width:520px;margin:0 auto;background:#16212e;"
    "border:1px solid #26333f;border-radius:14px;padding:28px\">"
    "<h1 style=\"font-size:18px;margin:0 0 14px\">Personal Health Strategist</h1>"
    "{body}"
    "<p style=\"color:#8aa0b2;font-size:12px;margin:22px 0 0\">"
    "If you didn't request this, you can ignore this email.</p>"
    "</div></div>"
)

_BUTTON = (
    "<a href=\"{link}\" style=\"display:inline-block;background:#2dd4bf;color:#04201c;"
    "text-decoration:none;font-weight:600;padding:11px 20px;border-radius:10px;"
    "margin:8px 0 4px\">{label}</a>"
)


def _render(body: str) -> str:
    return _WRAP.format(body=body)


def send_verification(to: str, link: str) -> bool:
    body = (
        "<p style=\"margin:0 0 14px;line-height:1.55\">Confirm this address to "
        "finish setting up your account.</p>"
        + _BUTTON.format(link=link, label="Verify my email")
        + "<p style=\"color:#8aa0b2;font-size:12px;margin:14px 0 0\">"
          "This link expires in 3 days.</p>"
    )
    return send(to, "Verify your email", _render(body), link=link)


def send_password_reset(to: str, link: str) -> bool:
    body = (
        "<p style=\"margin:0 0 14px;line-height:1.55\">Use the button below to "
        "choose a new password.</p>"
        + _BUTTON.format(link=link, label="Reset my password")
        + "<p style=\"color:#8aa0b2;font-size:12px;margin:14px 0 0\">"
          "This link expires in 1 hour and can only be used once.</p>"
    )
    return send(to, "Reset your password", _render(body), link=link)
