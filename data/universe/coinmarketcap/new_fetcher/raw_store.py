"""Raw CMC snapshot storage. Write once, never edit.

History is immutable: a date fetched today returns the same bytes next year, so
a date is fetched EXACTLY ONCE, ever. Everything downstream is re-derivable from
these files, which means a parsing bug costs a re-derive, not a re-download.

MANIFEST.csv sits beside the year folders, one row per fetched date. `raw/` is
gitignored and the manifest is committed, so git carries a record of what was
fetched and when without carrying ~172 MB of payload.

No content hash: a self-computed one proves "unchanged since I wrote it", never
"this is what CMC sent", which is not worth the concept. The Binance side does
verify, but only because the vendor publishes .CHECKSUM files.

ORDERING MATTERS. The .gz is written BEFORE the manifest row is appended. A
crash between the two leaves a file with no manifest row, which is harmless —
exists() is the checkpoint, so a re-run reconciles it. The reverse order would
leave a manifest row pointing at a file that is not there, which is a lie, and
lies in an audit trail are worse than gaps.
"""

from __future__ import annotations

import csv
import gzip
import threading
from pathlib import Path

from config import DATA_DIR
from data.universe.coinmarketcap.new_fetcher.client import RawSnapshot

RAW_CMC_DIR = DATA_DIR / "raw" / "cmc"
MANIFEST_PATH = RAW_CMC_DIR / "MANIFEST.csv"
MANIFEST_FIELDS = ["date", "url", "fetched_at", "n_rows", "n_bytes"]

# The .gz files need no locking — one path per date, no two workers ever touch
# the same one. MANIFEST.csv is the only shared mutable state, so only the
# append is serialised, and it is serialised HERE rather than in the caller so
# that write() is safe from any thread without the caller knowing to wrap it.
_manifest_lock = threading.Lock()


def path_for(date: str) -> Path:
    """data/raw/cmc/2020/2020-02-02.json.gz

    One file per DATE. The old SnapshotCache._date_to_month_filename
    (data/universe/coingecko/cache.py:117) collapsed every date in a month to
    YYYY_MM.json — daily snapshots would overwrite each other ~30x. That class is
    monthly-by-construction and is not reused here.
    """
    return RAW_CMC_DIR / date[:4] / f"{date}.json.gz"


def exists(date: str) -> bool:
    """This IS the checkpoint (§4.3). Deliberately keyed off the FILE, not the
    manifest: the file is the thing that matters, and a manifest row without a
    file must never be able to make a re-run skip a date."""
    return path_for(date).exists()


def write(snapshot: RawSnapshot) -> Path:
    """Atomic gzip write, then manifest append. Refuses to overwrite.

    Thread-safe: callable directly from worker threads.
    """
    path = path_for(snapshot.date)
    if path.exists():
        raise FileExistsError(f"raw snapshot already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)

    # 2020-02-02.json.gz -> 2020-02-02.json.tmp (verified: .suffixes is
    # ['.json', '.gz'] — the hyphens in the date are not parsed as suffixes)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=9) as fh:
        fh.write(snapshot.body)
    tmp.rename(path)          # atomic on the same filesystem

    _append_manifest({
        "date": snapshot.date,
        "url": snapshot.url,
        "fetched_at": snapshot.fetched_at,
        "n_rows": snapshot.n_rows,
        "n_bytes": len(snapshot.body),
    })
    return path


def read(date: str) -> bytes:
    with gzip.open(path_for(date), "rb") as fh:
        return fh.read()


def manifest() -> list[dict[str, str]]:
    """Read the manifest back. Empty list if nothing has been fetched yet.

    Append-only, so a date fetched twice (a raw file was deleted and the
    backfill repaired it) appears twice — that is truthful history, not a bug,
    but it means len(manifest()) counts FETCHES, not dates. For a date count
    use {r["date"] for r in manifest()}, or just count the files.
    """
    if not MANIFEST_PATH.exists():
        return []
    with open(MANIFEST_PATH, newline="") as fh:
        return list(csv.DictReader(fh))


def reconcile() -> tuple[set[str], set[str]]:
    """(files_without_a_manifest_row, manifest_rows_without_a_file).

    The first set is benign — a crash between the two writes, fixed by a re-run.
    The second should always be empty; if it is not, something deleted a raw
    file behind the pipeline's back.
    """
    on_disk = {p.name[: -len(".json.gz")] for p in RAW_CMC_DIR.rglob("*.json.gz")}
    in_manifest = {r["date"] for r in manifest()}
    return on_disk - in_manifest, in_manifest - on_disk


def _append_manifest(row: dict) -> None:
    with _manifest_lock:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        is_new = not MANIFEST_PATH.exists()
        with open(MANIFEST_PATH, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow(row)
