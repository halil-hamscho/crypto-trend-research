"""The per-coin run's inputs: the dataset manifest fixed at the freeze, and the checked loader.

Design: docs/october/per_coin_run_driver_design_2026-10-24.md §4.1-4.2.

  dataset manifest   the SHA-256 of every source file any stage-A run may read, written into the run
                     manifest at the freeze
  checked loader     every `pandas.read_csv` of a file during a run goes through it: it reads the bytes,
                     hashes exactly those bytes, compares them with the dataset manifest, checks the file
                     is in the current scope's permitted subset, and only then parses them. A file missing
                     from the manifest, a different hash, or a read outside the scope raises before any
                     computation uses the data, so a price, funding rate or CMC row changed after the
                     freeze fails even if every run reads the same altered file
  derived hashes     membership, the K5 reference index and forecast, and each run's decision tables,
                     hashed so the identity checks (§4.3) compare them field by field

The configs are YAML, read at import time; their hashes are checked by the driver's guard, not here.
"""
import hashlib
import io
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from data.universe.coinmarketcap.new_fetcher import contract_map, normalize, panel

ROOT = Path(__file__).resolve().parent.parent
PER_COIN_CONFIG = "docs/october/per_coin_support/frozen_config.yaml"
SEPTEMBER_CONFIG = "docs/september/prereg_support/frozen_config.yaml"


class DataManifestError(ValueError):
    """A file read during the run is absent from the dataset manifest or differs from its frozen hash."""


class PermittedSubsetError(ValueError):
    """A run read a file outside the subset its manifest entry permits."""


def relative(path) -> str:
    """The manifest key of a file: its path relative to the repository root."""
    p = Path(path).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract_files(contract: str) -> list:
    """The three derived files a contract can be read from: daily klines, funding, hourly marks."""
    return [panel.DERIVED_DIR / "ohlcv" / f"{contract}_1d.csv", panel.DERIVED_DIR / "funding" / f"{contract}.csv",
            panel.MARK_DIR / f"{contract}.csv"]


def stage_a_inventory(map_table=None) -> list:
    """Every source file a stage-A run may read (§4.1): the CMC universe table and the sealed contract
    map, plus klines, funding and marks for EVERY contract in that map, members or not, since a
    non-member's volume can change membership. Files that do not exist (a contract with no archived
    marks) are left out: reading one would fail anyway."""
    table = Path(map_table or contract_map.MAP_TABLE)
    contracts = sorted(set(pd.read_csv(table).contract))
    files = [Path(normalize.UNIVERSE_TABLE), table]
    for c in contracts:
        files += [f for f in contract_files(c) if f.exists()]
    return files


def dataset_manifest(files) -> dict:
    """{relative path: sha256} for the given files, read once each."""
    return {relative(f): sha256_bytes(Path(f).read_bytes()) for f in sorted({Path(f) for f in files})}


def permitted_files(contracts, include_membership: bool = False) -> list:
    """The relative paths a book over `contracts` may read; membership adds the CMC table and map."""
    out = [relative(f) for c in contracts for f in contract_files(c)]
    if include_membership:
        out += [relative(normalize.UNIVERSE_TABLE), relative(contract_map.MAP_TABLE)]
    return sorted(set(out))


class CheckedLoader:
    """Patches `pandas.read_csv` while active. Reads of a path are checked; reads of a buffer pass
    through (they are of bytes already checked, e.g. a zip member). `reads` records, per scope, the
    relative paths read and their hashes. Bytes are cached per path after the first check, so a file
    changed in the middle of a run cannot be served to a later scope."""

    def __init__(self, dataset: dict):
        self.dataset = dict(dataset)
        self.reads = {}
        self._cache = {}
        self._scope, self._permitted = None, None
        self._original = None

    @contextmanager
    def active(self):
        if self._original is not None:
            raise RuntimeError("the checked loader is already active")
        self._original = pd.read_csv
        loader = self

        def checked_read_csv(source, *args, **kwargs):
            if isinstance(source, (str, Path)):
                return loader._original(io.BytesIO(loader._bytes(source)), *args, **kwargs)
            return loader._original(source, *args, **kwargs)

        pd.read_csv = checked_read_csv
        try:
            yield self
        finally:
            pd.read_csv = self._original
            self._original = None

    @contextmanager
    def scope(self, name: str, permitted=None):
        """Reads inside are attributed to `name`; with `permitted`, any other file raises."""
        previous = (self._scope, self._permitted)
        self._scope, self._permitted = name, None if permitted is None else set(permitted)
        try:
            yield self
        finally:
            self._scope, self._permitted = previous

    def _bytes(self, source) -> bytes:
        key = relative(source)
        if key not in self.dataset:
            raise DataManifestError(f"{key} is not in the dataset manifest")
        if self._permitted is not None and key not in self._permitted:
            raise PermittedSubsetError(f"{self._scope}: {key} is outside this run's permitted files")
        if key not in self._cache:
            data = Path(source).read_bytes()
            digest = sha256_bytes(data)
            if digest != self.dataset[key]:
                raise DataManifestError(f"{key} differs from the dataset manifest (sha256 {digest[:12]}, "
                                        f"frozen {self.dataset[key][:12]})")
            self._cache[key] = data
        self.reads.setdefault(self._scope, {})[key] = self.dataset[key]
        return self._cache[key]


# ------------------------------------------------------------------ derived-input hashes
def frame_hash(obj) -> str:
    """sha256 of a frame or series, index included, values as their exact text: equal hashes mean
    equal tables."""
    h = hashlib.sha256()
    if isinstance(obj, pd.Series):
        obj = obj.to_frame()
    frame = obj.reset_index().astype(str)
    h.update("|".join(map(str, frame.columns)).encode())
    h.update(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes())
    return h.hexdigest()


def membership_hash(members_by_date: dict) -> str:
    """Members by reset date: the contract list (and weights) of every reconstitution."""
    rows = [(str(d), r.contract, float(getattr(r, "weight", float("nan"))))
            for d in sorted(members_by_date) for r in members_by_date[d].itertuples()]
    return frame_hash(pd.DataFrame(rows, columns=["reset", "contract", "weight"]))


def tables_hashes(tables, inputs) -> dict:
    """Every derived input of one book, hashed separately so a comparison can name what differs."""
    return {"forecast": frame_hash(tables.forecast), "sigma": frame_hash(tables.sigma),
            "price": frame_hash(tables.price), "mark_price": frame_hash(tables.mark_price),
            "rho_bar": frame_hash(tables.rho_bar), "eligibility": frame_hash(tables.eligible.astype(int)),
            "fill_prices": frame_hash(inputs.fill_prices), "decision_prices": frame_hash(inputs.decision_prices),
            "tradable": frame_hash(inputs.tradable.astype(int)),
            "settlements": frame_hash(pd.concat({c: inputs.settlements[c].astype(str)
                                                 for c in sorted(inputs.settlements)}))}
