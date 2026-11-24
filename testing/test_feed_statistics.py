"""Tests for feed statistics reconciliation against the article database."""

import json
from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rssidian.models import Article, Base, Feed
from rssidian.stats import compute_feed_statistics, reconcile_feed_statistics


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _feed(session, title="F", url="https://example.com/feed"):
    feed = Feed(title=title, url=url)
    session.add(feed)
    session.commit()
    return feed


def _article(session, feed, guid, tier=None, score=None):
    article = Article(
        feed_id=feed.id,
        title=guid,
        url=f"https://example.com/{guid}",
        guid=guid,
        published_at=datetime(2026, 11, 1),
        quality_tier=tier,
        quality_score=score,
    )
    session.add(article)
    session.commit()
    return article


def test_compute_statistics_counts_actual_articles():
    session = _session()
    feed = _feed(session)
    _article(session, feed, "a", tier="S", score=90)
    _article(session, feed, "b", tier="A", score=70)
    _article(session, feed, "c")  # unanalyzed

    stats = compute_feed_statistics(session, feed.id)

    assert stats["article_count"] == 3
    assert stats["quality_tier_counts"] == {"S": 1, "A": 1}
    assert stats["avg_quality_score"] == 80.0


def test_reconcile_fixes_drifted_counts():
    session = _session()
    feed = _feed(session)
    for guid in ("a", "b", "c"):
        _article(session, feed, guid, tier="B", score=50)

    # Simulate drift: denormalized counter says 10.
    feed.article_count = 10
    feed.quality_tier_counts = json.dumps({"B": 10})
    feed.avg_quality_score = 50.0
    session.commit()

    result = reconcile_feed_statistics(session)

    session.refresh(feed)
    assert feed.article_count == 3
    assert json.loads(feed.quality_tier_counts) == {"B": 3}
    assert feed.avg_quality_score == 50.0
    assert result["feeds_changed"] == 1


def test_reconcile_is_noop_when_already_consistent():
    session = _session()
    feed = _feed(session)
    _article(session, feed, "a", tier="A", score=70)
    reconcile_feed_statistics(session)

    result = reconcile_feed_statistics(session)
    assert result["feeds_changed"] == 0


def test_reconcile_limited_to_selected_feeds():
    session = _session()
    feed1 = _feed(session, "one", "https://one/feed")
    feed2 = _feed(session, "two", "https://two/feed")
    _article(session, feed1, "a", tier="S", score=90)
    _article(session, feed2, "b", tier="A", score=70)

    result = reconcile_feed_statistics(session, feed_ids=[feed1.id])

    assert result["feeds_checked"] == 1
    session.refresh(feed2)
    # feed2 untouched (still its default 0), feed1 corrected.
    assert feed1.article_count == 1
    assert feed2.article_count == 0


def test_zero_articles_resets_stats():
    session = _session()
    feed = _feed(session)
    feed.article_count = 5
    feed.avg_quality_score = 42.0
    feed.quality_tier_counts = json.dumps({"S": 5})
    session.commit()

    reconcile_feed_statistics(session)

    session.refresh(feed)
    assert feed.article_count == 0
    assert feed.avg_quality_score is None
    assert feed.quality_tier_counts is None
