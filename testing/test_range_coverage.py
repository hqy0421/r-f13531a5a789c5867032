"""Tests for range-aware feed processing and coverage computation."""

from datetime import datetime, timedelta

from rssidian.rss import _filter_entries_by_ranges, _success_coverage

T0 = datetime(2026, 11, 1)
T1 = datetime(2026, 11, 2)
T2 = datetime(2026, 11, 3)
T3 = datetime(2026, 11, 4)


def _entry(published):
    entry = {"id": str(published), "title": "t", "link": "https://x"}
    if published is not None:
        entry["published_parsed"] = published.timetuple()
    return entry


def test_filter_keeps_only_entries_in_ranges():
    entries = [_entry(T0), _entry(T2), _entry(T3)]
    kept = _filter_entries_by_ranges(entries, [(T1, T2 + timedelta(hours=1))])
    assert [e["id"] for e in kept] == [str(T2)]


def test_filter_keeps_undated_entries():
    """Entries without a date cannot be placed in time, so they are kept."""
    entries = [_entry(None), _entry(T2)]
    kept = _filter_entries_by_ranges(entries, [(T0, T1)])
    assert [e["id"] for e in kept] == [str(None)]


def test_filter_with_no_ranges_keeps_everything():
    entries = [_entry(T0), _entry(T3)]
    assert _filter_entries_by_ranges(entries, None) == entries
    assert _filter_entries_by_ranges(entries, []) == entries


def test_success_coverage_without_truncation_confirms_whole_range():
    coverage = _success_coverage([(T0, T3)], [], truncated=False)
    assert coverage == [(T0, T3)]


def test_success_coverage_truncated_clips_to_oldest_kept_entry():
    """When the feed was capped, only the range from the oldest kept entry is proven."""
    kept = [({"id": "a"}, T2), ({"id": "b"}, T1)]
    coverage = _success_coverage([(T0, T3)], kept, truncated=True)
    assert coverage == [(T1, T3)]


def test_success_coverage_truncated_with_no_dates_confirms_whole_range():
    coverage = _success_coverage([(T0, T3)], [({"id": "a"}, None)], truncated=True)
    assert coverage == [(T0, T3)]


def test_success_coverage_truncated_oldest_after_range_end_confirms_nothing():
    kept = [({"id": "a"}, T3 + timedelta(days=5))]
    coverage = _success_coverage([(T0, T3)], kept, truncated=True)
    assert coverage == []
