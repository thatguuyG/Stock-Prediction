"""NewsAPI headline ingestion."""
from __future__ import annotations

import datetime as dt
from typing import Iterable

import requests
from sqlalchemy.orm import Session

from packages.shared.config import get_settings
from packages.shared.db import upsert_ignore
from packages.shared.logging import get_logger
from packages.shared.models import NewsItem

log = get_logger(__name__)

NEWSAPI_URL = "https://newsapi.org/v2/everything"


class NewsAPIError(RuntimeError):
    pass


def fetch_headlines(
    symbol: str,
    query: str | None = None,
    page_size: int = 50,
    api_key: str | None = None,
    session: requests.Session | None = None,
) -> list[dict]:
    """Fetch recent headlines mentioning the symbol. Empty list if no API key configured."""
    key = api_key if api_key is not None else get_settings().newsapi_key
    if not key:
        log.warning("NEWSAPI_KEY not set; skipping news fetch for %s", symbol)
        return []
    http = session or requests.Session()
    params = {
        "q": query or symbol,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "apiKey": key,
    }
    resp = http.get(NEWSAPI_URL, params=params, timeout=15)
    if resp.status_code == 429:
        log.warning("NewsAPI rate-limited for %s", symbol)
        return []
    if resp.status_code != 200:
        raise NewsAPIError(f"NewsAPI {resp.status_code}: {resp.text[:200]}")
    payload = resp.json()
    if payload.get("status") != "ok":
        raise NewsAPIError(f"NewsAPI status={payload.get('status')} message={payload.get('message')}")
    return payload.get("articles", [])


def _parse_published_at(raw: str | None) -> dt.datetime | None:
    if not raw:
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def upsert_news(session: Session, symbol: str, articles: Iterable[dict]) -> int:
    """Insert news items idempotently (dedup by url). Returns count newly inserted."""
    rows: list[dict] = []
    seen_urls: set[str] = set()
    for art in articles:
        url = art.get("url")
        headline = art.get("title")
        if not url or not headline:
            continue
        if url in seen_urls:
            continue
        seen_urls.add(url)
        rows.append(
            {
                "symbol": symbol,
                "source": (art.get("source") or {}).get("name"),
                "headline": headline[:1024],
                "url": url[:2048],
                "published_at": _parse_published_at(art.get("publishedAt")),
            }
        )
    return upsert_ignore(session, NewsItem.__table__, rows, ["url"])


def ingest_symbol(session: Session, symbol: str) -> int:
    articles = fetch_headlines(symbol)
    inserted = upsert_news(session, symbol, articles)
    log.info("Ingested %d news items for %s", inserted, symbol)
    return inserted
