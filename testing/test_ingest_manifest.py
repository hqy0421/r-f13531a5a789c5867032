"""Tests for the re-consumable ingest progress manifest and retry planning."""

from datetime import datetime

from rssidian.progress import (
    STATUS_FAILED,
    STATUS_SUCCESS,
    STATUS_UNCONFIRMED,
    IngestManifest,
    intervals_from_json,
    intervals_to_json,
    merge_intervals,
    subtract_intervals,
)
from rssidian.rss import FeedIngestResult


class FakeFeed:
    def __init__(self, feed_id, title, url):
        self.id = feed_id
        self.title = title
        self.url = url


T0 = datetime(2026, 11, 1)
T1 = datetime(2026, 11, 2)
T2 = datetime(2026, 11, 3)
T3 = datetime(2026, 11, 4)


# ---------------------------------------------------------------------------
# Interval math
# ---------------------------------------------------------------------------

def test_merge_intervals_unions_overlaps_and_touching():
    merged = merge_intervals([(T0, T2), (T1, T3)])
    assert merged == [(T0, T3)]


def test_merge_intervals_keeps_gaps():
    merged = merge_intervals([(T0, T1), (T2, T3)])
    assert merged == [(T0, T1), (T2, T3)]


def test_subtract_intervals_removes_covered_middle():
    remaining = subtract_intervals([(T0, T3)], [(T1, T2)])
    assert remaining == [(T0, T1), (T2, T3)]


def test_subtract_intervals_all_covered_returns_empty():
    assert subtract_intervals([(T0, T2)], [(T0, T3)]) == []


def test_subtract_intervals_no_coverage_returns_base():
    assert subtract_intervals([(T0, T2)], []) == [(T0, T2)]


def test_interval_json_roundtrip():
    encoded = intervals_to_json([(T0, T1)])
    assert intervals_from_json(encoded) == [(T0, T1)]


def test_intervals_from_json_drops_bad_entries():
    raw = [["2026-11-01T00:00:00", "2026-11-02T00:00:00"], ["nope"], ["x", "y"]]
    assert intervals_from_json(raw) == [(T0, T1)]


# ---------------------------------------------------------------------------
# Manifest planning
# ---------------------------------------------------------------------------

def _result(feed, status, coverage, articles_seen=0, articles_new=0, error=None):
    return FeedIngestResult(
        feed_id=feed.id,
        feed_title=feed.title,
        feed_url=feed.url,
        status=status,
        window_start=T0,
        window_end=T3,
        coverage=list(coverage),
        articles_seen=articles_seen,
        articles_new=articles_new,
        error=error,
        started_at=T0,
        finished_at=T3,
    )


def test_fresh_manifest_plans_full_window_for_every_source():
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feeds = [FakeFeed(1, "A", "https://a/feed"), FakeFeed(2, "B", "https://b/feed")]

    plans = manifest.plan(feeds, T0, T3)

    assert len(plans) == 2
    assert all(plan.should_process for plan in plans)
    assert all(plan.intervals == [(T0, T3)] for plan in plans)


def test_confirmed_source_is_skipped_and_no_ranges_remain():
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feed = FakeFeed(1, "A", "https://a/feed")
    plan = manifest.plan([feed], T0, T3)[0]
    manifest.mark_started(plan)
    manifest.record(_result(feed, STATUS_SUCCESS, [(T0, T3)]))

    # A later run in the same window should skip the source.
    manifest.begin(T0, T3, run_id="next")
    plan2 = manifest.plan([feed], T0, T3)[0]

    assert plan2.action == "skip"
    assert plan2.intervals == []


def test_failed_source_keeps_full_range_pending():
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feed = FakeFeed(1, "A", "https://a/feed")
    plan = manifest.plan([feed], T0, T3)[0]
    manifest.mark_started(plan)
    manifest.record(_result(feed, STATUS_FAILED, [], error="boom"))

    record = manifest.source_for(feed)
    assert record["status"] == STATUS_FAILED
    assert record["last_error"] == "boom"
    assert intervals_from_json(record["pending"]) == [(T0, T3)]


def test_partial_success_retries_only_uncovered_tail():
    """A success that only confirmed the newest half should retry the older half."""
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feed = FakeFeed(1, "A", "https://a/feed")
    plan = manifest.plan([feed], T0, T3)[0]
    manifest.mark_started(plan)
    # Confirmed T1..T3 only.
    manifest.record(_result(feed, STATUS_SUCCESS, [(T1, T3)]))

    record = manifest.source_for(feed)
    assert intervals_from_json(record["pending"]) == [(T0, T1)]
    assert record["status"] == STATUS_UNCONFIRMED

    # Retry reads only T0..T1.
    manifest.begin(T0, T3, run_id="retry")
    retry_plan = manifest.plan([feed], T0, T3)[0]
    assert retry_plan.should_process
    assert retry_plan.intervals == [(T0, T1)]


def test_overlapping_retry_results_merge_into_contiguous_coverage():
    """Two partial successes with overlapping ranges merge by time overlap."""
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feed = FakeFeed(1, "A", "https://a/feed")

    plan = manifest.plan([feed], T0, T3)[0]
    manifest.mark_started(plan)
    manifest.record(_result(feed, STATUS_SUCCESS, [(T0, T2)]))

    manifest.begin(T0, T3, run_id="retry")
    plan2 = manifest.plan([feed], T0, T3)[0]
    assert plan2.intervals == [(T2, T3)]

    manifest.mark_started(plan2)
    manifest.record(_result(feed, STATUS_SUCCESS, [(T1, T3)]))

    record = manifest.source_for(feed)
    assert intervals_from_json(record["coverage"]) == [(T0, T3)]
    assert intervals_from_json(record["pending"]) == []
    assert record["status"] == STATUS_SUCCESS


def test_results_merge_by_source_url_not_feed_id():
    """Same source keeps one record even when the feed row id changes."""
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    first = FakeFeed(1, "A", "https://a/feed")
    plan = manifest.plan([first], T0, T3)[0]
    manifest.mark_started(plan)
    manifest.record(_result(first, STATUS_SUCCESS, [(T0, T2)]))

    # Feed re-added later gets a new id but the same URL.
    second = FakeFeed(99, "A renamed", "https://a/feed")
    record = manifest.source_for(second)

    assert len(manifest.data["sources"]) == 1
    assert record["feed_id"] == 99
    assert record["title"] == "A renamed"
    assert intervals_from_json(record["coverage"]) == [(T0, T2)]


def test_summary_counts_success_failed_and_skipped():
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feeds = [
        FakeFeed(1, "ok", "https://a/feed"),
        FakeFeed(2, "bad", "https://b/feed"),
    ]
    plans = manifest.plan(feeds, T0, T3)
    for plan in plans:
        manifest.mark_started(plan)
    manifest.record(_result(feeds[0], STATUS_SUCCESS, [(T0, T3)], articles_seen=5, articles_new=2))
    manifest.record(_result(feeds[1], STATUS_FAILED, [], error="nope"))

    summary = manifest.summary_dict()
    assert summary["success"] == 1
    assert summary["failed"] == 1
    assert summary["skipped"] == 0
    assert summary["articles_seen"] == 5
    assert summary["articles_new"] == 2


def test_save_and_load_roundtrip_preserves_coverage(tmp_path):
    path = tmp_path / "manifest.json"
    manifest = IngestManifest.empty(path=str(path))
    manifest.begin(T0, T3)
    feed = FakeFeed(1, "A", "https://a/feed")
    plan = manifest.plan([feed], T0, T3)[0]
    manifest.mark_started(plan)
    manifest.record(_result(feed, STATUS_SUCCESS, [(T0, T2)]))
    manifest.save()

    reloaded = IngestManifest.load(str(path))
    record = reloaded._find_source("https://a/feed")
    assert record is not None
    assert intervals_from_json(record["coverage"]) == [(T0, T2)]
    assert reloaded.data["schema_version"] == 1


def test_load_missing_file_returns_empty_manifest(tmp_path):
    manifest = IngestManifest.load(str(tmp_path / "does-not-exist.json"))
    assert manifest.data["sources"] == []


def test_load_corrupt_file_returns_empty_manifest(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not valid json")
    manifest = IngestManifest.load(str(path))
    assert manifest.data["sources"] == []


def test_interrupted_run_leaves_source_pending():
    """mark_started must persist before any network work so crashes retry."""
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feed = FakeFeed(1, "A", "https://a/feed")
    plan = manifest.plan([feed], T0, T3)[0]

    manifest.mark_started(plan)

    record = manifest.source_for(feed)
    assert record["status"] == STATUS_UNCONFIRMED
    assert intervals_from_json(record["pending"]) == [(T0, T3)]
    # And it shows up as retryable work.
    manifest.begin(T0, T3, run_id="retry")
    assert manifest.plan([feed], T0, T3)[0].should_process


def test_pending_sources_lists_only_unconfirmed():
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feeds = [
        FakeFeed(1, "ok", "https://a/feed"),
        FakeFeed(2, "bad", "https://b/feed"),
    ]
    plans = manifest.plan(feeds, T0, T3)
    for plan in plans:
        manifest.mark_started(plan)
    manifest.record(_result(feeds[0], STATUS_SUCCESS, [(T0, T3)]))
    manifest.record(_result(feeds[1], STATUS_FAILED, []))

    pending = manifest.pending_sources()
    assert [source["url"] for source in pending] == ["https://b/feed"]


def test_retry_sources_excludes_fully_confirmed():
    manifest = IngestManifest.empty()
    manifest.begin(T0, T3)
    feeds = [
        FakeFeed(1, "ok", "https://a/feed"),
        FakeFeed(2, "bad", "https://b/feed"),
    ]
    plans = manifest.plan(feeds, T0, T3)
    for plan in plans:
        manifest.mark_started(plan)
    manifest.record(_result(feeds[0], STATUS_SUCCESS, [(T0, T3)]))
    manifest.record(_result(feeds[1], STATUS_SUCCESS, [(T0, T3)]))

    assert manifest.retry_sources(T0, T3) == []


def test_retry_sources_includes_wider_lookback():
    """Asking for a wider window retries even sources that fully covered the old one."""
    manifest = IngestManifest.empty()
    manifest.begin(T1, T3)
    feed = FakeFeed(1, "ok", "https://a/feed")
    plan = manifest.plan([feed], T1, T3)[0]
    manifest.mark_started(plan)
    manifest.record(_result(feed, STATUS_SUCCESS, [(T1, T3)]))

    # Same window: nothing to retry.
    assert manifest.retry_sources(T1, T3) == []
    # Wider window: uncovered T0..T1, so the source is retried.
    assert manifest.retry_sources(T0, T3) == ["https://a/feed"]
