"""Integration tests for coverage-aware ingest and resume.

These drive ``process_all_feeds`` with a fake fetcher so no network is used.
"""

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from rssidian.config import Config
from rssidian.core import RSSProcessor
from rssidian.models import Article, Base, Feed
from rssidian.progress import IngestManifest
from rssidian.rss import FeedFetchResult


class FakeConfig(Config):
    def __init__(self):
        pass

    @property
    def default_lookback(self):
        return 7

    @property
    def max_articles_per_feed(self):
        return 25

    @property
    def analyze_during_ingestion(self):
        return False

    @property
    def openrouter_api_key(self):
        return None


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _add_feeds(session, n):
    feeds = [
        Feed(title=f"Feed{i}", url=f"https://site{i}.example/feed") for i in range(n)
    ]
    session.add_all(feeds)
    session.commit()
    return feeds


def _entry(feed_index, n, now):
    return {
        "id": f"g{feed_index}-{n}",
        "link": f"https://site{feed_index}.example/a{n}",
        "title": f"F{feed_index}A{n}",
        "summary": "word " * 500,
        "published_parsed": now.timetuple(),
    }


def _make_fetcher(failing, now):
    def fetch(url, max_retries=3, retry_delay=2):
        index = int(url.split("site")[1].split(".")[0])
        if index in failing:
            return FeedFetchResult(status="failed", error=f"HTTP 500 {url}")
        return FeedFetchResult(
            status="success",
            entries=[_entry(index, n, now) for n in range(3)],
            feed_title=f"Feed{index}",
        )

    return fetch


def test_ingest_records_progress_and_resume_only_touches_failures(monkeypatch, tmp_path):
    now = datetime.utcnow()
    session = _session()
    _add_feeds(session, 10)
    processor = RSSProcessor(FakeConfig(), session)
    path = str(tmp_path / "manifest.json")

    failing = {6, 7, 8, 9}
    monkeypatch.setattr("rssidian.rss.fetch_feed_entries", _make_fetcher(failing, now))

    manifest = IngestManifest.empty(path=path)
    result = processor.ingest_feeds(manifest=manifest)
    manifest.save()

    assert result["new_articles"] == 18
    assert result["manifest_summary"]["success"] == 6
    assert result["manifest_summary"]["failed"] == 4
    assert session.query(Article).count() == 18

    # Only the failed sources remain.
    reloaded = IngestManifest.load(path)
    pending = [s["url"] for s in reloaded.pending_sources()]
    assert len(pending) == 4
    assert {int(u.split("site")[1].split(".")[0]) for u in pending} == {6, 7, 8, 9}

    # Server recovers; resume should process exactly the four failed feeds.
    failing.clear()
    monkeypatch.setattr("rssidian.rss.fetch_feed_entries", _make_fetcher(failing, now))

    retry_sources = reloaded.retry_sources(
        *[datetime.fromisoformat(x) for x in
          (reloaded.data["window"]["start"], reloaded.data["window"]["end"])]
    )
    assert len(retry_sources) == 4

    result2 = processor.ingest_feeds(
        manifest=reloaded,
        resume=True,
        only_sources=retry_sources,
        window_start=datetime.fromisoformat(reloaded.data["window"]["start"]),
        window_end=datetime.fromisoformat(reloaded.data["window"]["end"]),
    )
    reloaded.save()

    assert result2["feeds_processed"] == 4
    assert result2["manifest_summary"]["success"] == 4
    assert result2["manifest_summary"]["failed"] == 0
    assert session.query(Article).count() == 30

    # Nothing left to resume.
    final = IngestManifest.load(path)
    assert final.retry_sources(
        datetime.fromisoformat(final.data["window"]["start"]),
        datetime.fromisoformat(final.data["window"]["end"]),
    ) == []


def test_resume_does_not_duplicate_articles(monkeypatch, tmp_path):
    now = datetime.utcnow()
    session = _session()
    _add_feeds(session, 2)
    processor = RSSProcessor(FakeConfig(), session)

    monkeypatch.setattr(
        "rssidian.rss.fetch_feed_entries", _make_fetcher(set(), now)
    )
    manifest = IngestManifest.empty(path=str(tmp_path / "m.json"))
    processor.ingest_feeds(manifest=manifest)
    assert session.query(Article).count() == 6

    # A plain re-ingest re-reads the (slightly shifted) window every time, but
    # GUID/URL dedup must prevent duplicate articles.
    result2 = processor.ingest_feeds(manifest=manifest)
    assert result2["new_articles"] == 0
    assert session.query(Article).count() == 6


def test_resume_with_confirmed_sources_skips_them(monkeypatch):
    now = datetime.utcnow()
    session = _session()
    _add_feeds(session, 2)
    processor = RSSProcessor(FakeConfig(), session)
    monkeypatch.setattr("rssidian.rss.fetch_feed_entries", _make_fetcher(set(), now))

    manifest = IngestManifest.empty()
    first = processor.ingest_feeds(manifest=manifest)
    window_start = manifest.data["window"]["start"]
    window_end = manifest.data["window"]["end"]

    # Resuming against the exact recorded window must skip confirmed sources.
    result2 = processor.ingest_feeds(
        manifest=manifest,
        resume=True,
        window_start=datetime.fromisoformat(window_start),
        window_end=datetime.fromisoformat(window_end),
    )
    assert first["new_articles"] == 6
    assert result2["feeds_skipped"] == 2
    assert result2["new_articles"] == 0
    assert session.query(Article).count() == 6


def test_only_sources_empty_list_processes_nothing(monkeypatch):
    now = datetime.utcnow()
    session = _session()
    _add_feeds(session, 3)
    processor = RSSProcessor(FakeConfig(), session)
    monkeypatch.setattr("rssidian.rss.fetch_feed_entries", _make_fetcher(set(), now))

    result = processor.ingest_feeds(manifest=IngestManifest.empty(), only_sources=[])
    assert result["feeds_processed"] == 0
    assert session.query(Article).count() == 0
