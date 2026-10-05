"""Automation: when a chat needs a human, save a ticket and optionally ping a webhook."""
import json
import logging
import time

import httpx

import config

log = logging.getLogger("handoff")


def create_ticket(ticket: dict) -> None:
    ticket = {**ticket, "created_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    config.STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.TICKETS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(ticket, ensure_ascii=False) + "\n")

    if config.WEBHOOK_URL:  # n8n / Zapier / Make / Slack-compatible endpoint
        try:
            httpx.post(config.WEBHOOK_URL, json=ticket, timeout=5)
        except Exception as exc:  # never break the chat because of a webhook
            log.warning("webhook failed: %s", exc)
