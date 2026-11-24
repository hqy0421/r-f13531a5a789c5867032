"""Feed statistics reconciliation.

``Feed.article_count``, ``Feed.avg_quality_score`` and
``Feed.quality_tier_counts`` are denormalized for fast display (Constitution
Article VIII.3).  They must always agree with the articles actually stored in
the database.

Historically the counters were bumped incrementally from a single code path
(``process_articles``).  Articles that were analyzed during ingestion, imported
by other paths, or counted before a crash therefore drifted from reality.

:func:`reconcile_feed_statistics` recomputes the three fields directly from the
``articles`` table so the stored inventory is always a faithful reflection of
what is actually there.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from .models import Article, Feed

logger = logging.getLogger(__name__)

QUALITY_TIER_ORDER = ("S", "A", "B", "C", "D")


def compute_feed_statistics(db_session: Session, feed_id: int) -> Dict[str, Any]:
    """Compute inventory statistics for one feed from the database.

    Returns:
        ``{"article_count": int, "avg_quality_score": float | None,
        "quality_tier_counts": {"S": n, ...}}``
    """
    article_count = (
        db_session.query(func.count(Article.id))
        .filter(Article.feed_id == feed_id)
        .scalar()
        or 0
    )

    tier_rows = (
        db_session.query(Article.quality_tier, func.count(Article.id))
        .filter(Article.feed_id == feed_id, Article.quality_tier.isnot(None))
        .group_by(Article.quality_tier)
        .all()
    )
    tier_counts: Counter = Counter()
    for tier, count in tier_rows:
        if tier in QUALITY_TIER_ORDER:
            tier_counts[tier] = int(count or 0)

    score_row = (
        db_session.query(func.avg(Article.quality_score))
        .filter(Article.feed_id == feed_id, Article.quality_score.isnot(None))
        .scalar()
    )
    avg_quality_score = float(score_row) if score_row is not None else None

    return {
        "article_count": int(article_count),
        "avg_quality_score": avg_quality_score,
        "quality_tier_counts": {
            tier: tier_counts[tier] for tier in QUALITY_TIER_ORDER if tier_counts[tier]
        },
    }


def _bulk_statistics(
    db_session: Session, feed_ids: Optional[Iterable[int]]
) -> Dict[int, Dict[str, Any]]:
    """Compute statistics for many feeds with three aggregate queries."""
    counts_query = db_session.query(Article.feed_id, func.count(Article.id))
    tier_query = db_session.query(
        Article.feed_id, Article.quality_tier, func.count(Article.id)
    ).filter(Article.quality_tier.isnot(None))
    score_query = db_session.query(
        Article.feed_id, func.avg(Article.quality_score)
    ).filter(Article.quality_score.isnot(None))

    ids: Optional[List[int]] = None
    if feed_ids is not None:
        ids = sorted({int(feed_id) for feed_id in feed_ids})
        if not ids:
            return {}
        counts_query = counts_query.filter(Article.feed_id.in_(ids))
        tier_query = tier_query.filter(Article.feed_id.in_(ids))
        score_query = score_query.filter(Article.feed_id.in_(ids))

    result: Dict[int, Dict[str, Any]] = {}

    def entry(feed_id: int) -> Dict[str, Any]:
        return result.setdefault(
            feed_id,
            {"article_count": 0, "avg_quality_score": None, "quality_tier_counts": {}},
        )

    for feed_id, count in counts_query.group_by(Article.feed_id).all():
        entry(feed_id)["article_count"] = int(count or 0)

    for feed_id, tier, count in tier_query.group_by(
        Article.feed_id, Article.quality_tier
    ).all():
        if tier in QUALITY_TIER_ORDER:
            entry(feed_id)["quality_tier_counts"][tier] = int(count or 0)

    for feed_id, avg_score in score_query.group_by(Article.feed_id).all():
        entry(feed_id)["avg_quality_score"] = (
            float(avg_score) if avg_score is not None else None
        )

    # Order tiers consistently for stable serialization.
    for stats in result.values():
        stats["quality_tier_counts"] = {
            tier: stats["quality_tier_counts"][tier]
            for tier in QUALITY_TIER_ORDER
            if stats["quality_tier_counts"].get(tier)
        }

    return result


def reconcile_feed_statistics(
    db_session: Session,
    feed_ids: Optional[Iterable[int]] = None,
    commit: bool = True,
) -> Dict[str, Any]:
    """Rewrite stored feed statistics so they match the articles table.

    Args:
        db_session: Active session.
        feed_ids: Feeds to reconcile. Defaults to every feed.
        commit: Whether to commit the session after reconciliation.

    Returns:
        Summary ``{"feeds_checked", "feeds_changed", "changes": [...]}``.
    """
    if feed_ids is None:
        feeds = db_session.query(Feed).all()
    else:
        ids = sorted({int(feed_id) for feed_id in feed_ids})
        if not ids:
            return {"feeds_checked": 0, "feeds_changed": 0, "changes": []}
        feeds = db_session.query(Feed).filter(Feed.id.in_(ids)).all()

    computed = _bulk_statistics(db_session, [feed.id for feed in feeds])

    changes: List[Dict[str, Any]] = []

    for feed in feeds:
        stats = computed.get(
            feed.id,
            {"article_count": 0, "avg_quality_score": None, "quality_tier_counts": {}},
        )

        new_tier_json = (
            json.dumps(stats["quality_tier_counts"]) if stats["quality_tier_counts"] else None
        )

        before = (
            feed.article_count or 0,
            feed.avg_quality_score,
            feed.quality_tier_counts,
        )
        after = (
            stats["article_count"],
            stats["avg_quality_score"],
            new_tier_json,
        )

        if before == after:
            continue

        feed.article_count = stats["article_count"]
        feed.avg_quality_score = stats["avg_quality_score"]
        feed.quality_tier_counts = new_tier_json

        changes.append(
            {
                "feed_id": feed.id,
                "title": feed.title,
                "article_count": {"before": before[0], "after": after[0]},
                "avg_quality_score": {"before": before[1], "after": after[1]},
            }
        )
        logger.info(
            "Reconciled feed '%s' (id=%s): article_count %s -> %s",
            feed.title,
            feed.id,
            before[0],
            after[0],
        )

    if commit and changes:
        db_session.commit()

    return {
        "feeds_checked": len(feeds),
        "feeds_changed": len(changes),
        "changes": changes,
    }
