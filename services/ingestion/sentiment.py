"""VADER sentiment scoring for stored news headlines."""
from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from packages.shared.logging import get_logger
from packages.shared.models import NewsItem, Sentiment

log = get_logger(__name__)

MODEL_TAG = "vader-1.0"


_analyzer: SentimentIntensityAnalyzer | None = None


def _get_analyzer() -> SentimentIntensityAnalyzer:
    global _analyzer  # pylint: disable=global-statement
    if _analyzer is None:
        _analyzer = SentimentIntensityAnalyzer()
    return _analyzer


def score_headline(text: str) -> dict[str, float]:
    """Return VADER scores: keys pos, neu, neg, compound."""
    return _get_analyzer().polarity_scores(text)


def score_unscored(session: Session, batch_size: int = 200) -> int:
    """Score any NewsItem that does not yet have a Sentiment row. Returns count scored."""
    stmt = (
        select(NewsItem)
        .outerjoin(Sentiment, Sentiment.news_item_id == NewsItem.id)
        .where(Sentiment.id.is_(None))
        .limit(batch_size)
    )
    total = 0
    while True:
        items = session.execute(stmt).scalars().all()
        if not items:
            break
        for item in items:
            scores = score_headline(item.headline)
            session.add(
                Sentiment(
                    news_item_id=item.id,
                    model=MODEL_TAG,
                    compound=scores["compound"],
                    pos=scores["pos"],
                    neu=scores["neu"],
                    neg=scores["neg"],
                )
            )
            total += 1
        session.flush()
        if len(items) < batch_size:
            break
    log.info("Scored %d news items with %s", total, MODEL_TAG)
    return total
