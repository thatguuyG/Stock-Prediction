"""Sentiment tests — VADER directionality + scoring of stored news items."""
from __future__ import annotations

from sqlalchemy import select

from packages.shared.models import NewsItem, Sentiment
from services.ingestion import sentiment


def test_score_headline_directionality():
    pos = sentiment.score_headline("Apple posts great wonderful amazing results")
    neg = sentiment.score_headline("Apple posts terrible awful disastrous results")
    assert pos["compound"] > 0
    assert neg["compound"] < 0


def test_score_unscored_writes_one_row_per_item(session):
    session.add_all(
        [
            NewsItem(symbol=None, headline="Great wonderful amazing news!", url="https://e.example/a"),
            NewsItem(symbol=None, headline="Terrible awful disastrous news.", url="https://e.example/b"),
        ]
    )
    session.flush()

    n = sentiment.score_unscored(session)
    session.flush()
    assert n == 2

    rows = session.execute(select(Sentiment)).scalars().all()
    assert len(rows) == 2
    assert all(r.model == sentiment.MODEL_TAG for r in rows)
    assert all(-1.0 <= r.compound <= 1.0 for r in rows)


def test_score_unscored_is_idempotent(session):
    session.add(NewsItem(symbol=None, headline="Okay news.", url="https://e.example/c"))
    session.flush()

    first = sentiment.score_unscored(session)
    second = sentiment.score_unscored(session)
    assert first == 1
    assert second == 0
