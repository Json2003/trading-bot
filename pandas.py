"""A tiny subset of pandas used for unit tests.
This stub implements only the features required by the tests.
"""
from __future__ import annotations

import csv
from typing import Any, Dict, Iterable, List, Optional


class Series(list):
    """Minimal list-like series supporting dtype attribute."""

    @property
    def dtype(self):
        for v in self:
            if isinstance(v, float):
                return float
            if isinstance(v, int):
                return int
        return str


class _Loc:
    def __init__(self, df: "DataFrame"):
        self._df = df

    def __getitem__(self, key):
        row, col = key
        return self._df._rows[row][col]


class DataFrame:
    def __init__(self, data: Optional[Any] = None, columns: Optional[List[str]] = None):
        self._rows: List[Dict[str, Any]] = []
        self.columns: List[str] = []
        if isinstance(data, dict):
            cols = list(data.keys())
            length = len(next(iter(data.values()))) if data else 0
            for i in range(length):
                row = {c: data[c][i] for c in cols}
                self._rows.append(row)
            self.columns = cols
        elif isinstance(data, list):
            for row in data:
                self._rows.append(dict(row))
            if self._rows:
                self.columns = list(self._rows[0].keys())
        elif data is None:
            pass
        else:
            raise TypeError("Unsupported data type for DataFrame")
        if columns:
            self.columns = columns

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, key: str) -> Series:
        return Series([row.get(key) for row in self._rows])

    @property
    def loc(self) -> _Loc:
        return _Loc(self)

    # utility methods
    def to_csv(self, path: Any, index: bool = False, header: bool = True, mode: str = "w"):
        with open(path, mode, newline="") as f:
            fieldnames = self.columns or (list(self._rows[0].keys()) if self._rows else [])
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if header:
                writer.writeheader()
            for row in self._rows:
                writer.writerow(row)

    def to_dict(self) -> List[Dict[str, Any]]:
        return [dict(r) for r in self._rows]


# module level helpers

def DataFrame_from_records(records: Iterable[Dict[str, Any]]) -> DataFrame:
    return DataFrame(list(records))


def read_csv(path: Any, parse_dates: Optional[List[str]] = None, index_col: Optional[str] = None) -> DataFrame:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows: List[Dict[str, Any]] = []
        for row in reader:
            new_row: Dict[str, Any] = {}
            for k, v in row.items():
                if v is None:
                    new_row[k] = v
                    continue
                try:
                    new_row[k] = int(v)
                except ValueError:
                    try:
                        new_row[k] = float(v)
                    except ValueError:
                        new_row[k] = v
            rows.append(new_row)
    return DataFrame(rows)

DataFrame.from_records = staticmethod(DataFrame_from_records)

