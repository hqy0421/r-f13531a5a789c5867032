"""Re-consumable ingest progress manifests.

An ingest run is a batch over many feed sources.  A run can be interrupted or
partially fail, so ``ingest`` persists a manifest describing, per source, which
time ranges were *confirmed* (successfully fetched and stored) and which ranges
are still failed or unconfirmed.

The manifest is a plain JSON document with a stable schema so that it can be
consumed again by a later ``rssidian ingest --resume`` (or by any external
tool).  Retries use the recorded coverage to read only the uncovered time
ranges, and merge their results back by source identity (feed URL) and time
overlap.

Schema (``schema_version`` 1)::

    {
      "schema_version": 1,
      "created_at": "2026-11-25T00:00:00",
      "updated_at": "2026-11-25T00:05:00",
      "run_id": "20261125T000000000000Z",
      "lookback_days": 7,
      "window": {"start": "...", "end": "..."},
      "sources": [
        {
          "feed_id": 1,
          "title": "Example",
          "url": "https://example.com/feed.xml",
          "status": "success",          # success | failed | unconfirmed
          "coverage": [["start", "end"], ...],
          "pending": [["start", "end"], ...],
          "attempts": 2,
          "last_attempt_at": "...",
          "last_error": null,
          "last_articles_seen": 12,
          "last_articles_new": 3,
          "total_articles_seen": 30,
          "total_articles_new": 8,
          "last_action": "process",     # process | skip
          "last_run_id": "...",
          "runs": [
            {"run_id": "...", "started_at": "...", "finished_at": "...",
             "window": ["start", "end"], "status": "success",
             "articles_seen": 12, "articles_new": 3, "error": null}
          ]
        }
      ],
      "summary": {
        "sources": 10, "success": 6, "failed": 3, "unconfirmed": 1,
        "skipped": 0, "articles_seen": 0, "articles_new": 0
      }
    }
"""

from __future__ import annotations

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

Interval = Tuple[datetime, datetime]

ACTION_PROCESS = "process"
ACTION_SKIP = "skip"

STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_UNCONFIRMED = "unconfirmed"


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def utcnow() -> datetime:
    """Return the current UTC time as a naive datetime (database convention)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def to_naive(value: datetime) -> datetime:
    """Normalize a datetime to naive UTC."""
    if value.tzinfo is not None:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def iso(value: Optional[datetime]) -> Optional[str]:
    """Serialize a datetime for the manifest."""
    if value is None:
        return None
    return to_naive(value).isoformat()


def parse_datetime(value: Any) -> Optional[datetime]:
    """Parse a datetime from a manifest value (ISO string or datetime)."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return to_naive(value)
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return to_naive(datetime.fromisoformat(text))
    except ValueError:
        return None


def intervals_from_json(raw: Optional[Iterable[Any]]) -> List[Interval]:
    """Convert manifest JSON intervals to datetime pairs, dropping bad values."""
    result: List[Interval] = []
    for item in raw or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            start = parse_datetime(item[0])
            end = parse_datetime(item[1])
            if start is not None and end is not None and start < end:
                result.append((start, end))
    return result


def intervals_to_json(intervals: Iterable[Interval]) -> List[List[str]]:
    """Serialize intervals for the manifest."""
    return [[iso(start), iso(end)] for start, end in merge_intervals(intervals)]


# ---------------------------------------------------------------------------
# Interval math
# ---------------------------------------------------------------------------

def merge_intervals(intervals: Iterable[Interval]) -> List[Interval]:
    """Merge overlapping or touching intervals, sorted by start."""
    cleaned: List[Interval] = []
    for start, end in intervals:
        start, end = to_naive(start), to_naive(end)
        if start < end:
            cleaned.append((start, end))
    cleaned.sort(key=lambda pair: pair[0])

    merged: List[Interval] = []
    for start, end in cleaned:
        if merged and start <= merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def subtract_intervals(base: Iterable[Interval], covers: Iterable[Interval]) -> List[Interval]:
    """Return the parts of ``base`` not covered by ``covers``."""
    covers_merged = merge_intervals(covers)
    result: List[Interval] = []

    for start, end in merge_intervals(base):
        cursor = start
        for cover_start, cover_end in covers_merged:
            if cover_end <= cursor:
                continue
            if cover_start >= end:
                break
            if cover_start > cursor:
                result.append((cursor, min(cover_start, end)))
            cursor = max(cursor, cover_end)
            if cursor >= end:
                break
        if cursor < end:
            result.append((cursor, end))

    return result


def interval_contains(intervals: Sequence[Interval], value: datetime) -> bool:
    """Return True when ``value`` falls inside any of ``intervals``."""
    if not intervals:
        return True
    value = to_naive(value)
    return any(start <= value <= end for start, end in intervals)


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------

@dataclass
class SourcePlan:
    """Work to perform for a single feed source in this run."""

    feed: Any
    url: str
    window_start: datetime
    window_end: datetime
    intervals: List[Interval] = field(default_factory=list)
    action: str = ACTION_PROCESS
    reason: Optional[str] = None

    @property
    def should_process(self) -> bool:
        return self.action == ACTION_PROCESS

    @property
    def feed_id(self) -> Optional[int]:
        return getattr(self.feed, "id", None)

    @property
    def title(self) -> str:
        return getattr(self.feed, "title", "") or self.url

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feed_id": self.feed_id,
            "title": self.title,
            "url": self.url,
            "action": self.action,
            "reason": self.reason,
            "window": [iso(self.window_start), iso(self.window_end)],
            "intervals": intervals_to_json(self.intervals),
        }


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

_KNOWN_TOP_LEVEL = (
    "schema_version",
    "created_at",
    "updated_at",
    "run_id",
    "lookback_days",
    "window",
    "sources",
    "summary",
    "run_count",
    "last_run",
)

_MAX_RUN_HISTORY = 100


class IngestManifest:
    """Persistent per-source ingest progress.

    Use :meth:`load` to read an existing manifest, :meth:`begin` to start a
    run, :meth:`plan` to compute per-source uncovered ranges, and
    :meth:`record` to merge a source result back in.  Save after every source
    so an interrupted run leaves usable progress behind.
    """

    def __init__(self, data: Optional[Dict[str, Any]] = None, path: Optional[str] = None):
        if data is None:
            data = {}
        if not isinstance(data, dict):
            logger.warning("Ignoring malformed ingest manifest (expected an object)")
            data = {}

        self.path = path

        # Preserve unknown keys so newer consumers do not lose data.
        self.data: Dict[str, Any] = dict(data)
        self.data.setdefault("schema_version", SCHEMA_VERSION)
        self.data.setdefault("created_at", iso(utcnow()))
        self.data.setdefault("updated_at", self.data["created_at"])
        self.data.setdefault("run_id", None)
        self.data.setdefault("lookback_days", None)
        self.data.setdefault("window", {})
        sources = self.data.get("sources")
        if not isinstance(sources, list):
            self.data["sources"] = []
        self.data.setdefault("summary", {})
        self.data.setdefault("run_count", 0)

    # -- construction -------------------------------------------------------

    @classmethod
    def empty(
        cls, lookback_days: Optional[int] = None, path: Optional[str] = None
    ) -> "IngestManifest":
        manifest = cls({}, path=path)
        manifest.data["lookback_days"] = lookback_days
        return manifest

    @classmethod
    def load(cls, path: str) -> "IngestManifest":
        """Load a manifest from disk.

        A missing file yields an empty manifest.  A corrupt file is logged and
        treated as empty so ingestion can still proceed.
        """
        if not os.path.exists(path):
            return cls.empty(path=path)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read ingest manifest %s: %s", path, exc)
            return cls.empty(path=path)

        manifest = cls(data, path=path)
        version = manifest.data.get("schema_version")
        if version != SCHEMA_VERSION:
            logger.warning(
                "Ingest manifest %s uses schema version %s (this version reads %s)",
                path,
                version,
                SCHEMA_VERSION,
            )
        return manifest

    # -- persistence --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        self.data["updated_at"] = iso(utcnow())
        self.data["summary"] = self.summary_dict()
        return self.data

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False, default=str)

    def save(self, path: Optional[str] = None) -> None:
        """Atomically write the manifest to ``path`` (defaults to its origin)."""
        path = path or self.path
        if not path:
            raise ValueError("IngestManifest.save() needs a path")
        self.path = path
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        tmp_path = f"{path}.tmp"
        payload = self.to_json()
        with open(tmp_path, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.write("\n")
        os.replace(tmp_path, path)

    # -- source records -----------------------------------------------------

    def _sources(self) -> List[Dict[str, Any]]:
        return self.data["sources"]

    def _find_source(self, url: str) -> Optional[Dict[str, Any]]:
        for source in self._sources():
            if source.get("url") == url:
                return source
        return None

    def source_for(self, feed: Any) -> Dict[str, Any]:
        """Return the record for ``feed`` (by URL), creating it if needed."""
        url = getattr(feed, "url", None) or str(feed)
        record = self._find_source(url)
        if record is None:
            record = {
                "feed_id": getattr(feed, "id", None),
                "title": getattr(feed, "title", ""),
                "url": url,
                "status": STATUS_UNCONFIRMED,
                "coverage": [],
                "pending": [],
                "attempts": 0,
                "last_attempt_at": None,
                "last_error": None,
                "last_articles_seen": 0,
                "last_articles_new": 0,
                "total_articles_seen": 0,
                "total_articles_new": 0,
                "last_action": None,
                "last_run_id": None,
                "runs": [],
            }
            self._sources().append(record)
        else:
            # Keep identity fields current (feed may have been re-added).
            record["feed_id"] = getattr(feed, "id", record.get("feed_id"))
            record["title"] = getattr(feed, "title", record.get("title", ""))
        return record

    def coverage_for(self, url: str) -> List[Interval]:
        record = self._find_source(url)
        if not record:
            return []
        return intervals_from_json(record.get("coverage"))

    # -- run lifecycle ------------------------------------------------------

    @staticmethod
    def new_run_id() -> str:
        return utcnow().strftime("%Y%m%dT%H%M%S%fZ")

    def begin(
        self,
        window_start: datetime,
        window_end: datetime,
        lookback_days: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> str:
        """Record the window and identity of the run about to start."""
        window_start, window_end = to_naive(window_start), to_naive(window_end)
        run_id = run_id or self.new_run_id()
        self.data["run_id"] = run_id
        self.data["lookback_days"] = lookback_days
        self.data["window"] = {"start": iso(window_start), "end": iso(window_end)}
        self.data["run_count"] = int(self.data.get("run_count") or 0) + 1
        self.data["last_run"] = {
            "run_id": run_id,
            "started_at": iso(utcnow()),
            "window": {"start": iso(window_start), "end": iso(window_end)},
            "lookback_days": lookback_days,
        }
        return run_id

    def plan(
        self,
        feeds: Sequence[Any],
        window_start: datetime,
        window_end: datetime,
    ) -> List[SourcePlan]:
        """Compute, per feed, the uncovered ranges within ``[start, end]``."""
        window_start, window_end = to_naive(window_start), to_naive(window_end)
        plans: List[SourcePlan] = []

        for feed in feeds:
            url = getattr(feed, "url", None)
            if not url:
                continue
            record = self.source_for(feed)
            uncovered = subtract_intervals(
                [(window_start, window_end)],
                intervals_from_json(record.get("coverage")),
            )
            plan = SourcePlan(
                feed=feed,
                url=url,
                window_start=window_start,
                window_end=window_end,
                intervals=uncovered,
            )
            if not uncovered:
                plan.action = ACTION_SKIP
                plan.reason = "already confirmed"

            record["last_action"] = plan.action
            record["last_run_id"] = self.data.get("run_id")
            record["pending"] = intervals_to_json(uncovered)
            plans.append(plan)

        return plans

    def mark_started(self, plan: SourcePlan) -> None:
        """Mark a source as in-flight so an interrupted run leaves it pending."""
        record = self.source_for(plan.feed)
        record["status"] = STATUS_UNCONFIRMED
        record["attempts"] = int(record.get("attempts") or 0) + 1
        record["last_attempt_at"] = iso(utcnow())
        record["last_error"] = None
        record["last_articles_seen"] = 0
        record["last_articles_new"] = 0
        record["pending"] = intervals_to_json(plan.intervals)
        record["last_action"] = ACTION_PROCESS
        record["last_run_id"] = self.data.get("run_id")

    def record(self, result: Any) -> Dict[str, Any]:
        """Merge a source result into the manifest.

        ``result`` is duck-typed (usually ``rssidian.rss.FeedIngestResult``) and
        must expose ``feed_url``, ``feed_title``, ``status``, ``window_start``,
        ``window_end``, ``coverage``, ``articles_seen``, ``articles_new`` and
        ``error``.  Coverage is unioned with prior successful coverage, which is
        how retries merge results by source and overlapping time range.
        """
        record = self._find_source(getattr(result, "feed_url", ""))
        if record is None:
            record = self.source_for(
                _FeedProxy(
                    getattr(result, "feed_id", None),
                    getattr(result, "feed_title", ""),
                    getattr(result, "feed_url", ""),
                )
            )

        status = getattr(result, "status", STATUS_FAILED)
        run_id = self.data.get("run_id")
        started = getattr(result, "started_at", None)
        finished = getattr(result, "finished_at", None) or utcnow()
        window_start = getattr(result, "window_start", None)
        window_end = getattr(result, "window_end", None)
        articles_seen = int(getattr(result, "articles_seen", 0) or 0)
        articles_new = int(getattr(result, "articles_new", 0) or 0)
        error = getattr(result, "error", None)
        coverage = [
            (to_naive(start), to_naive(end))
            for start, end in (getattr(result, "coverage", None) or [])
            if start is not None and end is not None and to_naive(start) < to_naive(end)
        ]

        if status == STATUS_SUCCESS:
            merged = merge_intervals(
                intervals_from_json(record.get("coverage")) + coverage
            )
            record["coverage"] = intervals_to_json(merged)

        record["feed_id"] = getattr(result, "feed_id", record.get("feed_id"))
        record["title"] = getattr(result, "feed_title", record.get("title"))
        record["attempts"] = int(record.get("attempts") or 0)
        record["last_attempt_at"] = iso(finished)
        record["last_error"] = error
        record["last_articles_seen"] = articles_seen
        record["last_articles_new"] = articles_new
        record["total_articles_seen"] = (
            int(record.get("total_articles_seen") or 0) + articles_seen
        )
        record["total_articles_new"] = (
            int(record.get("total_articles_new") or 0) + articles_new
        )
        record["last_action"] = ACTION_PROCESS
        record["last_run_id"] = run_id

        # Recompute remaining work for this run's window.
        last_window_start = parse_datetime(
            (self.data.get("window") or {}).get("start")
        ) or window_start
        last_window_end = parse_datetime(
            (self.data.get("window") or {}).get("end")
        ) or window_end
        if last_window_start and last_window_end:
            pending = subtract_intervals(
                [(last_window_start, last_window_end)],
                intervals_from_json(record.get("coverage")),
            )
        else:
            pending = []
        record["pending"] = intervals_to_json(pending)

        if status == STATUS_SUCCESS:
            # Success is only "confirmed" when nothing is left for the window.
            record["status"] = STATUS_SUCCESS if not pending else STATUS_UNCONFIRMED
        else:
            record["status"] = STATUS_FAILED

        runs = record.get("runs")
        if not isinstance(runs, list):
            runs = []
        runs.append(
            {
                "run_id": run_id,
                "started_at": iso(started),
                "finished_at": iso(finished),
                "window": [iso(window_start), iso(window_end)],
                "intervals": intervals_to_json(coverage),
                "status": getattr(result, "status", STATUS_FAILED),
                "articles_seen": articles_seen,
                "articles_new": articles_new,
                "truncated": bool(getattr(result, "truncated", False)),
                "error": error,
            }
        )
        record["runs"] = runs[-_MAX_RUN_HISTORY:]

        return record

    # -- reporting ----------------------------------------------------------

    def summary_dict(self) -> Dict[str, Any]:
        """Summarize the sources touched by the current run."""
        run_id = self.data.get("run_id")
        run_sources = [
            source
            for source in self._sources()
            if source.get("last_run_id") == run_id
        ]
        counts: Counter = Counter()
        skipped = 0
        articles_seen = 0
        articles_new = 0

        for source in run_sources:
            if source.get("last_action") == ACTION_SKIP:
                skipped += 1
                continue
            counts[source.get("status") or STATUS_UNCONFIRMED] += 1
            articles_seen += int(source.get("last_articles_seen") or 0)
            articles_new += int(source.get("last_articles_new") or 0)

        return {
            "sources": len(run_sources),
            "success": counts.get(STATUS_SUCCESS, 0),
            "failed": counts.get(STATUS_FAILED, 0),
            "unconfirmed": counts.get(STATUS_UNCONFIRMED, 0),
            "skipped": skipped,
            "articles_seen": articles_seen,
            "articles_new": articles_new,
        }

    def failed_sources(self) -> List[Dict[str, Any]]:
        """Sources from the current run that still need a retry."""
        run_id = self.data.get("run_id")
        return [
            source
            for source in self._sources()
            if source.get("last_run_id") == run_id
            and source.get("status") in (STATUS_FAILED, STATUS_UNCONFIRMED)
        ]

    def pending_sources(self) -> List[Dict[str, Any]]:
        """All sources with unconfirmed coverage remaining."""
        return [source for source in self._sources() if source.get("pending")]

    def retry_sources(
        self,
        window_start: datetime,
        window_end: datetime,
    ) -> List[str]:
        """URLs worth retrying for ``[window_start, window_end]``.

        A source is included when it is failed/unconfirmed, has leftover
        pending ranges, or has not yet confirmed the whole requested window
        (for example when a later resume asks for a wider lookback than the
        original run).
        """
        target = [(to_naive(window_start), to_naive(window_end))]
        urls: List[str] = []
        for source in self._sources():
            url = source.get("url")
            if not url:
                continue
            uncovered = subtract_intervals(
                target, intervals_from_json(source.get("coverage"))
            )
            status = source.get("status")
            if uncovered or source.get("pending") or status in (
                STATUS_FAILED,
                STATUS_UNCONFIRMED,
            ):
                urls.append(url)
        return urls


class _FeedProxy:
    """Minimal feed-like object used when a result has no matching record."""

    def __init__(self, feed_id: Optional[int], title: str, url: str):
        self.id = feed_id
        self.title = title
        self.url = url
