"""MCP feed statistics must reflect the real article database."""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rssidian.config import Config
from rssidian.mcp import ModelContextProtocol
from rssidian.models import Article, Base, Feed


class _Cfg(Config):
    def __init__(self):
        pass


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add(session, title, url, tiers):
    feed = Feed(title=title, url=url, article_count=999)  # deliberately wrong
    session.add(feed)
    session.commit()
    for i, (tier, score) in enumerate(tiers):
        session.add(
            Article(
                feed_id=feed.id,
                title=f"{title}-{i}",
                url=f"{url}/{i}",
                guid=f"{title}-{i}",
                published_at=datetime(2026, 11, 1),
                quality_tier=tier,
                quality_score=score,
            )
        )
    session.commit()
    return feed


def test_feed_stats_single_reflects_database():
    session = _session()
    _add(session, "F", "https://f/feed", [("S", 90), ("A", 70)])

    mcp = ModelContextProtocol(_Cfg(), session)
    stats = mcp.get_feed_stats("F")

    assert stats["article_count"] == 2
    assert stats["quality_tier_counts"] == {"S": 1, "A": 1}
    assert stats["avg_quality_score"] == 80.0


def test_feed_stats_all_reflects_database():
    session = _session()
    _add(session, "A", "https://a/feed", [("S", 90)])
    _add(session, "B", "https://b/feed", [("B", 50), ("B", 50), ("B", 50)])

    mcp = ModelContextProtocol(_Cfg(), session)
    stats = mcp.get_feed_stats()

    assert stats["total_articles"] == 4
    per_feed = {f["title"]: f["article_count"] for f in stats["feeds"]}
    assert per_feed == {"A": 1, "B": 3}


def test_subscriptions_report_database_counts():
    session = _session()
    _add(session, "A", "https://a/feed", [("S", 90), ("S", 90)])

    mcp = ModelContextProtocol(_Cfg(), session)
    subs = mcp.get_subscriptions()

    assert subs[0]["article_count"] == 2
