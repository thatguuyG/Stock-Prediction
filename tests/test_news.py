"""News ingestion tests — NewsAPI mocked via `responses`."""
from __future__ import annotations

import responses
from sqlalchemy import select

from packages.shared.models import NewsItem, Ticker
from services.ingestion import news


def _article(title: str, url: str, source: str = "TestWire", published: str = "2024-05-01T12:00:00Z") -> dict:
    return {
        "title": title,
        "url": url,
        "source": {"name": source},
        "publishedAt": published,
    }


@responses.activate
def test_fetch_headlines_returns_articles():
    responses.add(
        responses.GET,
        news.NEWSAPI_URL,
        json={
            "status": "ok",
            "articles": [_article("AAPL up", "https://a.example/1")],
        },
        status=200,
    )
    out = news.fetch_headlines("AAPL", api_key="dummy")
    assert len(out) == 1
    assert out[0]["title"] == "AAPL up"


@responses.activate
def test_fetch_headlines_handles_rate_limit():
    responses.add(responses.GET, news.NEWSAPI_URL, json={"status": "error"}, status=429)
    assert news.fetch_headlines("AAPL", api_key="dummy") == []


def test_fetch_headlines_skips_when_no_api_key(monkeypatch):
    from packages.shared import config as cfg
    monkeypatch.setattr(cfg, "_settings", None)
    monkeypatch.setenv("NEWSAPI_KEY", "")
    assert news.fetch_headlines("AAPL") == []


def test_upsert_news_dedupes_by_url(session):
    session.add(Ticker(symbol="AAPL", active=True))
    session.flush()

    articles = [
        _article("Headline 1", "https://a.example/1"),
        _article("Headline 1 dup", "https://a.example/1"),  # same url
        _article("Headline 2", "https://a.example/2"),
    ]
    n1 = news.upsert_news(session, "AAPL", articles)
    session.flush()
    n2 = news.upsert_news(session, "AAPL", articles)
    session.flush()

    assert n1 == 2
    assert n2 == 0

    rows = session.execute(select(NewsItem).where(NewsItem.symbol == "AAPL")).all()
    assert len(rows) == 2
