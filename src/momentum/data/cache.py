"""Local Parquet cache so every run uses identical data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


class Cache:
    """Store named DataFrames as Parquet files with a JSON metadata sidecar."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def _paths(self, name: str) -> tuple[Path, Path]:
        return self.root / f"{name}.parquet", self.root / f"{name}.json"

    def exists(self, name: str) -> bool:
        data, meta = self._paths(name)
        return data.exists() and meta.exists()

    def save(self, name: str, df: pd.DataFrame, meta: dict[str, Any]) -> None:
        data, meta_path = self._paths(name)
        self.root.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data)
        meta_path.write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    def load(self, name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
        data, meta_path = self._paths(name)
        meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        return pd.read_parquet(data), meta
