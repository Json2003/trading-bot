"""A lightweight Pandas-inspired stub for offline tests.

The full project uses Pandas extensively, however the execution environment
that powers the kata intentionally lacks binary dependencies such as NumPy and
Pandas.  To keep the examples runnable we provide a very small subset of the
APIs that the unit tests rely on.  The implementation intentionally mirrors the
behaviour of Pandas where practical, yet the focus is correctness rather than
performance.

Only the pieces exercised by the tests are implemented: ``Series`` and
``DataFrame`` containers, ``date_range``/``to_datetime`` helpers, ``concat`` and
``read_csv``.  The containers expose the handful of arithmetic, rolling and
exponential moving average operations that the trading utilities require.  The
module is pure Python and builds upon the local ``numpy`` stub shipped alongside
this repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import csv
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np

Number = Union[int, float]


def _is_scalar(obj: Any) -> bool:
    return not isinstance(obj, (list, tuple, Series, DataFrame, np.ndarray))


class Index(list):
    """Very small index object with ``get_loc`` helper."""

    def get_loc(self, value):
        for i, item in enumerate(self):
            if item == value:
                return i
        raise KeyError(value)


class _ILoc:
    def __init__(self, data, index: Index):
        self._data = data
        self._index = index

    def __getitem__(self, item):
        if isinstance(item, slice):
            rng = range(*item.indices(len(self._index)))
            return [self._data[i] for i in rng]
        if item < 0:
            item = len(self._index) + item
        return self._data[item]


class Series:
    """Minimal Series implementation."""

    __array_priority__ = 1000

    def __init__(self, data: Any = None, index: Optional[Sequence[Any]] = None, dtype=None, name: Optional[str] = None):
        if isinstance(data, Series):
            data = data._values[:]
            index = data.index if index is None else index
        elif isinstance(data, dict):
            index = list(data.keys())
            data = list(data.values())
        elif isinstance(data, Iterable) and not isinstance(data, (list, tuple, str, bytes)):
            data = list(data)
        elif data is None:
            data = []
        elif not isinstance(data, (list, tuple)):
            data = [data]

        if index is None:
            idx = list(range(len(data)))
        else:
            idx = list(index)
            if len(data) == 1 and len(idx) > 1:
                data = list(data) * len(idx)
        self.index = Index(idx)
        self._values: List[Any] = list(data)
        if dtype is not None:
            self._values = [dtype(v) for v in self._values]
        self.name = name
        self.iloc = _ILoc(self._values, self.index)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._values)

    def __iter__(self) -> Iterator[Any]:
        return iter(self._values)

    def __getitem__(self, item):
        if isinstance(item, slice):
            idx = self.index[item]
            return Series(self._values[item], index=idx)
        if item in self.index:
            position = self.index.get_loc(item) if hasattr(self.index, "get_loc") else self.index.index(item)
            return self._values[position]
        if isinstance(item, Series):
            positions = [i for i, flag in enumerate(item._values) if flag]
            return Series([self._values[i] for i in positions], index=[self.index[i] for i in positions])
        if isinstance(item, list):
            return Series([self._values[i] for i in item], index=[self.index[i] for i in item])
        return self._values[item]

    def __setitem__(self, key, value):
        if isinstance(key, Series):
            positions = [i for i, flag in enumerate(key._values) if flag]
            for pos in positions:
                self._values[pos] = value
            return
        if hasattr(key, "_values") and hasattr(key, "index"):
            try:
                flags = list(key._values)
            except TypeError:
                flags = None
            else:
                if flags is not None and len(flags) == len(self._values):
                    updated = False
                    for pos, flag in enumerate(flags):
                        if bool(flag):
                            self._values[pos] = value
                            updated = True
                    if updated:
                        return
        if isinstance(key, list):
            for k, v in zip(key, value if isinstance(value, list) else [value] * len(key)):
                self._values[k] = v
        else:
            self._values[key] = value

    def copy(self) -> "Series":
        return Series(self._values[:], index=self.index[:], name=self.name)

    @property
    def values(self) -> np.ndarray:
        try:
            return np.asarray(self._values)
        except Exception:
            return np.asarray(self._values)

    def to_list(self) -> List[Any]:
        return list(self._values)

    @property
    def dtype(self):
        for v in self._values:
            if v is None:
                continue
            if isinstance(v, bool):
                return bool
            if isinstance(v, int):
                return int
            if isinstance(v, float):
                return float
        return type(None)

    # ------------------------------------------------------------------
    # Arithmetic / comparison helpers
    # ------------------------------------------------------------------
    def _binary(self, other, op: Callable[[Any, Any], Any]):
        if isinstance(other, Series):
            other_values = other._align_to(self.index)
        elif isinstance(other, list):
            other_values = other
        else:
            other_values = [other] * len(self)
        return Series([op(a, b) for a, b in zip(self._values, other_values)], index=self.index[:])

    def _align_to(self, index: Sequence[Any]) -> List[Any]:
        lookup = {idx: val for idx, val in zip(self.index, self._values)}
        return [lookup.get(i, math.nan) for i in index]

    def __add__(self, other):
        return self._binary(other, lambda a, b: _arith(a, b, lambda x, y: x + y))

    def __sub__(self, other):
        return self._binary(other, lambda a, b: _arith(a, b, lambda x, y: x - y))

    def __mul__(self, other):
        return self._binary(other, lambda a, b: _arith(a, b, lambda x, y: x * y))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        def _div(a, b):
            if _isna(a) or _isna(b):
                return math.nan
            if b == 0:
                return math.inf
            return a / b

        return self._binary(other, _div)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return Series([other - v if not _isna(v) else math.nan for v in self._values], index=self.index[:])

    def __rtruediv__(self, other):
        def _rdiv(v):
            if _isna(v):
                return math.nan
            if v == 0:
                if _isna(other) or other == 0:
                    return math.nan
                return math.copysign(math.inf, other)
            return other / v

        return Series([_rdiv(v) for v in self._values], index=self.index[:])

    def __neg__(self):
        return Series([-v if not _isna(v) else v for v in self._values], index=self.index[:])

    def __gt__(self, other):
        return self._binary(other, lambda a, b: False if _isna(a) or _isna(b) else a > b)

    def __lt__(self, other):
        return self._binary(other, lambda a, b: False if _isna(a) or _isna(b) else a < b)

    def __ge__(self, other):
        return self._binary(other, lambda a, b: False if _isna(a) or _isna(b) else a >= b)

    def __le__(self, other):
        return self._binary(other, lambda a, b: False if _isna(a) or _isna(b) else a <= b)

    def __eq__(self, other):
        return self._binary(other, lambda a, b: a == b)

    def abs(self) -> "Series":
        return Series([abs(v) if v is not None else v for v in self._values], index=self.index[:])

    def __and__(self, other):
        return self._binary(other, lambda a, b: (not _isna(a) and bool(a)) and (not _isna(b) and bool(b)))

    def __or__(self, other):
        return self._binary(other, lambda a, b: (not _isna(a) and bool(a)) or (not _isna(b) and bool(b)))

    def __invert__(self):
        return Series([not bool(v) if not _isna(v) else False for v in self._values], index=self.index[:])

    def __rand__(self, other):
        return self._binary(other, lambda a, b: (not _isna(b) and bool(b)) and bool(other))

    def __ror__(self, other):
        return self._binary(other, lambda a, b: (not _isna(b) and bool(b)) or bool(other))

    def gt(self, other):
        return self > other

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    def mean(self) -> float:
        vals = [float(v) for v in self._values if not _isna(v)]
        return sum(vals) / len(vals) if vals else 0.0

    def std(self, ddof: int = 0) -> float:
        vals = [float(v) for v in self._values if not _isna(v)]
        if not vals:
            return 0.0
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals)
        denom = max(1, len(vals) - ddof)
        return math.sqrt(var / denom)

    def sum(self) -> float:
        vals = [float(v) for v in self._values if not _isna(v)]
        return sum(vals)

    def max(self):
        vals = [v for v in self._values if not _isna(v)]
        return max(vals) if vals else math.nan

    def min(self):
        vals = [v for v in self._values if not _isna(v)]
        return min(vals) if vals else math.nan

    def median(self) -> float:
        vals = sorted([float(v) for v in self._values if not _isna(v)])
        if not vals:
            return 0.0
        mid = len(vals) // 2
        if len(vals) % 2 == 1:
            return vals[mid]
        return (vals[mid - 1] + vals[mid]) / 2

    # ------------------------------------------------------------------
    # Pandas-like helpers used by the tests
    # ------------------------------------------------------------------
    def shift(self, periods: int = 1) -> "Series":
        values = [math.nan] * len(self)
        for i in range(len(self)):
            j = i - periods
            if 0 <= j < len(self):
                values[i] = self._values[j]
        return Series(values, index=self.index[:])

    def diff(self, periods: int = 1) -> "Series":
        values = [math.nan] * len(self)
        for i in range(len(self)):
            j = i - periods
            if 0 <= j < len(self) and not _isna(self._values[i]) and not _isna(self._values[j]):
                values[i] = self._values[i] - self._values[j]
        return Series(values, index=self.index[:])

    def pct_change(self, periods: int = 1) -> "Series":
        values = [math.nan] * len(self)
        for i in range(len(self)):
            j = i - periods
            if 0 <= j < len(self):
                prev = self._values[j]
                cur = self._values[i]
                if not _isna(prev) and prev != 0 and not _isna(cur):
                    values[i] = (cur - prev) / prev
                else:
                    values[i] = 0.0
        return Series(values, index=self.index[:])

    def clip(self, lower=None, upper=None):
        def _clip(v):
            if lower is not None and v < lower:
                v = lower
            if upper is not None and v > upper:
                v = upper
            return v

        return Series([_clip(v) if not _isna(v) else v for v in self._values], index=self.index[:])

    def fillna(self, value=None, method: Optional[str] = None):
        values = self._values[:]
        if method == "ffill":
            last = None
            for i, v in enumerate(values):
                if _isna(v):
                    values[i] = last if last is not None else v
                else:
                    last = v
        elif method == "bfill":
            last = None
            for i in reversed(range(len(values))):
                v = values[i]
                if _isna(v):
                    values[i] = last if last is not None else v
                else:
                    last = v
        elif value is not None:
            values = [value if _isna(v) else v for v in values]
        return Series(values, index=self.index[:])

    def replace(self, to_replace, value):
        if not isinstance(to_replace, (list, tuple)):
            to_replace = [to_replace]
        values = [value if v in to_replace else v for v in self._values]
        return Series(values, index=self.index[:])

    def astype(self, dtype) -> "Series":
        return Series([dtype(v) for v in self._values], index=self.index[:])

    def rolling(self, window: int, min_periods: Optional[int] = None):
        return _Rolling(self, window, min_periods or window)

    def ewm(self, span: int, adjust: bool = False, min_periods: int = 0):
        return _EWM(self, span, min_periods)

    def to_dict(self):
        return {idx: val for idx, val in zip(self.index, self._values)}

    def any(self) -> bool:
        return any(bool(v) for v in self._values if not _isna(v))

    def all(self) -> bool:
        return all(bool(v) for v in self._values if not _isna(v))

    def idxmax(self):
        if not self._values:
            return None
        max_val = None
        max_idx = None
        for idx, val in zip(self.index, self._values):
            if _isna(val):
                continue
            if max_val is None or val > max_val:
                max_val = val
                max_idx = idx
        return max_idx

    def apply(self, func: Callable[[Any], Any]):
        return Series([func(v) for v in self._values], index=self.index[:])

    def get(self, key, default=None):
        try:
            pos = self.index.get_loc(key)
            return self._values[pos]
        except KeyError:
            return default

    def rank(self, pct: bool = False) -> "Series":
        pairs = [(v, idx) for v, idx in zip(self._values, self.index) if not _isna(v)]
        sorted_vals = sorted(pairs, key=lambda x: x[0])
        ranks: Dict[Any, float] = {}
        for order, (_, idx) in enumerate(sorted_vals, start=1):
            ranks[idx] = order
        out = []
        for idx in self.index:
            val = ranks.get(idx, math.nan)
            if pct and not math.isnan(val):
                val = val / len(sorted_vals)
            out.append(val)
        return Series(out, index=self.index[:])

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"Series({self._values})"


class _Rolling:
    def __init__(self, series: Series, window: int, min_periods: int):
        self.series = series
        self.window = window
        self.min_periods = min_periods

    def _apply(self, func: Callable[[List[Any]], Any]) -> Series:
        values = []
        data = self.series._values
        for i in range(len(data)):
            start = max(0, i - self.window + 1)
            window = [v for v in data[start:i + 1] if not _isna(v)]
            if len(window) >= self.min_periods:
                values.append(func(window))
            else:
                values.append(math.nan)
        return Series(values, index=self.series.index[:])

    def mean(self):
        return self._apply(lambda w: sum(w) / len(w) if w else math.nan)

    def std(self):
        def _std(w):
            if not w:
                return math.nan
            mean = sum(w) / len(w)
            var = sum((v - mean) ** 2 for v in w) / len(w)
            return math.sqrt(var)

        return self._apply(_std)

    def sum(self):
        return self._apply(lambda w: sum(w))

    def max(self):
        return self._apply(lambda w: max(w) if w else math.nan)

    def min(self):
        return self._apply(lambda w: min(w) if w else math.nan)

    def rank(self, pct: bool = False):
        def _rank(window):
            if not window:
                return math.nan
            last = window[-1]
            sorted_window = sorted(window)
            pos = sorted_window.index(last)
            if pct:
                return (pos + 1) / len(sorted_window)
            return pos + 1

        return self._apply(_rank)


class _EWM:
    def __init__(self, series: Series, span: int, min_periods: int):
        self.series = series
        self.span = span
        self.min_periods = min_periods

    def mean(self):
        alpha = 2 / (self.span + 1)
        values = []
        ema = None
        count = 0
        for v in self.series._values:
            if _isna(v):
                values.append(math.nan)
                continue
            count += 1
            if ema is None:
                ema = v
            else:
                ema = alpha * v + (1 - alpha) * ema
            values.append(ema if count >= self.min_periods else math.nan)
        return Series(values, index=self.series.index[:])


class DataFrame:
    """Minimal DataFrame implementation."""

    def __init__(self, data: Any = None, index: Optional[Sequence[Any]] = None, columns: Optional[Sequence[str]] = None):
        if isinstance(data, DataFrame) or (hasattr(data, "_data") and hasattr(data, "columns") and hasattr(data, "index")):
            base = data.copy()
            self._data = {col: base._data[col].copy() for col in base.columns}
            self.columns = list(base.columns)
            self.index = Index(list(base.index))
            self.iloc = _DataFrameILoc(self)
            self.loc = _DataFrameLoc(self)
            return
        elif isinstance(data, list) and data and isinstance(data[0], dict):
            if columns:
                columns = list(columns)
            else:
                ordered_keys = {}
                for row in data:
                    for key in row.keys():
                        if key not in ordered_keys:
                            ordered_keys[key] = None
                columns = list(ordered_keys.keys())
            index = index or range(len(data))
            self._data = {
                col: Series([row.get(col, math.nan) for row in data], index=index)
                for col in columns
            }
        elif isinstance(data, dict):
            columns = list(data.keys()) if columns is None else list(columns)
            if index is None:
                index = None
                for value in data.values():
                    if isinstance(value, Series):
                        index = value.index
                        break
                if index is None:
                    sample = next(iter(data.values())) if data else []
                    if hasattr(sample, "__len__") and not isinstance(sample, (str, bytes)):
                        index = range(len(sample))
                    else:
                        index = []
            index = Index(list(index))
            self._data = {}
            for col in columns:
                col_data = data.get(col, [])
                if isinstance(col_data, Series):
                    self._data[col] = Series(col_data._align_to(index), index=index)
                else:
                    if _is_scalar(col_data):
                        values = [col_data for _ in range(len(index))]
                    else:
                        values = list(col_data)
                    self._data[col] = Series(values, index=index)
        else:
            self._data = {}
            columns = list(columns or [])
            index = Index(list(index or []))

        self.columns = list(columns or self._data.keys())
        if index is None:
            any_series = next(iter(self._data.values()), Series())
            self.index = Index(any_series.index[:])
        else:
            self.index = Index(list(index))
        self.iloc = _DataFrameILoc(self)
        self.loc = _DataFrameLoc(self)

    def copy(self) -> "DataFrame":
        return DataFrame({col: series.copy() for col, series in self._data.items()}, index=self.index[:], columns=self.columns[:])

    def __getitem__(self, key):
        if isinstance(key, list):
            return DataFrame({col: self._data[col] for col in key}, index=self.index[:], columns=key)
        if isinstance(key, Series):
            positions = [i for i, flag in enumerate(key._values) if flag]
            data = {}
            for col in self.columns:
                values = [self._data[col]._values[i] for i in positions]
                idx = [self.index[i] for i in positions]
                data[col] = Series(values, index=idx)
            return DataFrame(data, index=[self.index[i] for i in positions], columns=self.columns[:])
        return self._data[key]

    def __setitem__(self, key, value):
        if isinstance(value, Series):
            self._data[key] = value
        else:
            self._data[key] = Series(value, index=self.index[:])
        if key not in self.columns:
            self.columns.append(key)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def to_dict(self, orient: str = "dict"):
        if orient == "records":
            rows = []
            for i in range(len(self.index)):
                row = {col: self._data[col]._values[i] for col in self.columns}
                rows.append(row)
            return rows
        return {col: series._values[:] for col, series in self._data.items()}

    def to_csv(self, path: Union[str, Path], index: bool = True, header: bool = True, mode: str = "w"):
        with open(path, mode, newline="") as f:
            writer = csv.writer(f)
            header_row = []
            if index:
                header_row.append("index")
            header_row.extend(self.columns)
            if header:
                writer.writerow(header_row)
            for idx_pos, idx_val in enumerate(self.index):
                row = []
                if index:
                    row.append(idx_val)
                row.extend(self._data[col]._values[idx_pos] for col in self.columns)
                writer.writerow(row)

    def merge(self, other: "DataFrame", on: str, suffixes: Tuple[str, str] = ("_x", "_y")) -> "DataFrame":
        left_rows = self.to_dict("records")
        right_lookup = {}
        for row in other.to_dict("records"):
            right_lookup.setdefault(row[on], []).append(row)
        merged_rows = []
        for row in left_rows:
            matches = right_lookup.get(row[on], [{}])
            for match in matches:
                merged = {}
                for col, val in row.items():
                    merged[col + suffixes[0] if col != on else col] = val
                for col, val in match.items():
                    if col == on:
                        continue
                    merged[col + suffixes[1]] = val
                merged_rows.append(merged)
        return DataFrame(merged_rows)

    def set_index(self, column: str) -> "DataFrame":
        new_index = self._data[column]._values[:]
        data = {col: Series(series._values[:], index=new_index) for col, series in self._data.items() if col != column}
        return DataFrame(data, index=new_index)

    def reset_index(self, drop: bool = False) -> "DataFrame":
        rows = []
        for pos, idx in enumerate(self.index):
            row = {col: self._data[col]._values[pos] for col in self.columns}
            if not drop:
                row["index"] = idx
            rows.append(row)
        cols = self.columns[:] + ([] if drop else ["index"])
        return DataFrame(rows, columns=cols)

    def reindex(self, new_index: Sequence[Any]) -> "DataFrame":
        rows = {idx: pos for pos, idx in enumerate(self.index)}
        data: Dict[str, List[Any]] = {}
        for col, series in self._data.items():
            col_values = []
            for idx in new_index:
                pos = rows.get(idx)
                col_values.append(series._values[pos] if pos is not None else math.nan)
            data[col] = col_values
        return DataFrame(data, index=new_index)

    def ffill(self) -> "DataFrame":
        data = {}
        for col, series in self._data.items():
            data[col] = series.fillna(method="ffill")
        return DataFrame(data, index=self.index[:])

    def fillna(self, value=None, method: Optional[str] = None) -> "DataFrame":
        data: Dict[str, Series] = {}

        if method is not None:
            method = method.lower()
            if method not in {"ffill", "bfill"}:
                raise ValueError("Only forward and backward fill are supported in this stub")
            for col, series in self._data.items():
                data[col] = series.fillna(method=method)
            return DataFrame(data, index=self.index[:], columns=self.columns[:])

        if isinstance(value, DataFrame):
            fill_values: Dict[str, Any] = {col: value._data.get(col) for col in value.columns}
        elif isinstance(value, Series):
            fill_values = {value.name: value}
        elif isinstance(value, dict):
            fill_values = value
        else:
            fill_values = {}

        for col in self.columns:
            series = self._data[col]
            fill_value = None
            if fill_values:
                if col in fill_values and not isinstance(fill_values[col], Series):
                    fill_value = fill_values[col]
                elif col in fill_values and isinstance(fill_values[col], Series):
                    aligned = fill_values[col]._align_to(self.index)
                    data[col] = Series(
                        [aligned[i] if _isna(series._values[i]) else series._values[i] for i in range(len(series))],
                        index=self.index[:],
                    )
                    continue
                elif None in fill_values and not isinstance(fill_values[None], Series):
                    fill_value = fill_values[None]

            if isinstance(fill_value, Series):
                aligned = fill_value._align_to(self.index)
                data[col] = Series(
                    [aligned[i] if _isna(series._values[i]) else series._values[i] for i in range(len(series))],
                    index=self.index[:],
                )
            elif fill_value is not None:
                data[col] = series.fillna(value=fill_value)
            elif value is not None and fill_value is None:
                data[col] = series.fillna(value=value)
            else:
                data[col] = series.copy()

        return DataFrame(data, index=self.index[:], columns=self.columns[:])

    def __len__(self):
        return len(self.index)

    def max(self, axis: Optional[int] = None):
        if axis == 1:
            values = []
            for i in range(len(self.index)):
                row_vals = [self._data[col]._values[i] for col in self.columns if not _isna(self._data[col]._values[i])]
                values.append(max(row_vals) if row_vals else math.nan)
            return Series(values, index=self.index[:])
        raise NotImplementedError("max with axis other than 1 is not implemented")

    def min(self, axis: Optional[int] = None):
        if axis == 1:
            values = []
            for i in range(len(self.index)):
                row_vals = [self._data[col]._values[i] for col in self.columns if not _isna(self._data[col]._values[i])]
                values.append(min(row_vals) if row_vals else math.nan)
            return Series(values, index=self.index[:])
        raise NotImplementedError("min with axis other than 1 is not implemented")

    def __repr__(self):  # pragma: no cover - debugging helper
        return f"DataFrame(columns={self.columns}, rows={len(self)})"

    def __contains__(self, key):
        return key in self.columns

    def __iter__(self):
        return iter(self.columns)


class _DataFrameILoc:
    def __init__(self, frame: DataFrame):
        self._frame = frame

    def __getitem__(self, item):
        if isinstance(item, slice):
            rng = range(*item.indices(len(self._frame.index)))
            rows = []
            for i in rng:
                row = {col: self._frame._data[col]._values[i] for col in self._frame.columns}
                rows.append(row)
            return DataFrame(rows)
        if item < 0:
            item = len(self._frame.index) + item
        values = [self._frame._data[col]._values[item] for col in self._frame.columns]
        series = Series(values, index=self._frame.columns[:])
        if item < len(self._frame.index):
            series.name = self._frame.index[item]
        return series


class _DataFrameLoc:
    def __init__(self, frame: DataFrame):
        self._frame = frame

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row_key, col_key = key
            if row_key in self._frame.index:
                pos = self._frame.index.index(row_key)
                series = self._frame._data[col_key]
                return series._values[pos]
            raise KeyError(row_key)
        raise TypeError("loc requires (row, column) tuple")


def Series_from_records(records: Iterable[Dict[str, Any]]) -> Series:
    return Series(list(records))


def concat(items: Sequence[Union[Series, DataFrame]], axis: int = 0) -> Union[Series, DataFrame]:
    if axis == 0:
        raise NotImplementedError("row-wise concat not required in tests")
    if not items:
        return DataFrame()
    index = items[0].index
    data = {}
    for idx, item in enumerate(items):
        if hasattr(item, "_values") and hasattr(item, "index"):
            values = item._align_to(index) if hasattr(item, "_align_to") else list(item)
            data[f"col_{idx}"] = Series(values, index=index)
        else:
            for col in item.columns:
                series = item[col]
                values = series._align_to(index) if hasattr(series, "_align_to") else list(series)
                data[col] = Series(values, index=index)
    return DataFrame(data, index=index)


def isna(value) -> bool:
    return _isna(value)


def _isna(value) -> bool:
    if isinstance(value, float):
        return math.isnan(value)
    return value is None


def _zero_if_na(value):
    return 0 if _isna(value) else value


def _arith(a, b, op: Callable[[Any, Any], Any]):
    if _isna(a) or _isna(b):
        return math.nan
    return op(a, b)


def DataFrame_from_records(records: Iterable[Dict[str, Any]]) -> DataFrame:
    return DataFrame(list(records))


def read_csv(path: Union[str, Path]) -> DataFrame:
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    for row in rows:
        for key, value in row.items():
            if value is None:
                continue
            try:
                if "." in value:
                    row[key] = float(value)
                else:
                    row[key] = int(value)
            except Exception:
                pass
    return DataFrame(rows)


def date_range(start: Union[str, datetime], periods: Optional[int] = None, freq: str = "D", end: Optional[Union[str, datetime]] = None, tz: Optional[str] = None):
    if isinstance(start, str):
        try:
            start_dt = datetime.fromisoformat(start)
        except ValueError:
            if len(start) == 4 and start.isdigit():
                start_dt = datetime(int(start), 1, 1)
            else:
                start_dt = datetime.strptime(start, "%Y-%m-%d")
    else:
        start_dt = start
    if end is not None:
        end_dt = datetime.fromisoformat(end) if isinstance(end, str) else end
    elif periods is not None:
        delta = _freq_to_delta(freq)
        end_dt = start_dt + delta * (periods - 1)
    else:
        raise ValueError("must supply either periods or end")
    delta = _freq_to_delta(freq)
    out = []
    cur = start_dt
    while cur <= end_dt:
        out.append(cur.replace(tzinfo=timezone.utc) if tz else cur)
        cur = cur + delta
        if periods and len(out) >= periods:
            break
    return out


def _freq_to_delta(freq: str) -> timedelta:
    freq = freq.upper()
    if freq.endswith("MIN") or freq.endswith("T"):
        minutes = int(freq[:-3] or 1) if freq.endswith("MIN") else 1
        return timedelta(minutes=minutes)
    if freq.endswith("H"):
        return timedelta(hours=int(freq[:-1] or 1))
    if freq.endswith("D"):
        return timedelta(days=int(freq[:-1] or 1))
    return timedelta(0)


def to_datetime(values: Iterable[Any]) -> Series:
    parsed = []
    for val in values:
        if isinstance(val, datetime):
            parsed.append(val)
        else:
            parsed.append(datetime.fromisoformat(str(val)))
    return Series(parsed)


class _DateTimeAccessor:
    def __init__(self, series: Series):
        self._series = series

    def tz_convert(self, tz_name: str):
        tzinfo = timezone.utc if tz_name.upper() == "UTC" else timezone.utc
        values = []
        for val in self._series._values:
            if isinstance(val, datetime):
                values.append(val.astimezone(tzinfo))
            else:
                values.append(val)
        return Series(values, index=self._series.index[:])


Series.dt = property(lambda self: _DateTimeAccessor(self))  # type: ignore[attr-defined]


def qcut(series: Series, q: int, labels=False) -> Series:
    values = [v for v in series._values if not _isna(v)]
    if not values:
        return Series([math.nan] * len(series), index=series.index[:])
    sorted_vals = sorted(values)
    bins = []
    for i in range(1, q):
        pos = int(i * len(sorted_vals) / q)
        bins.append(sorted_vals[pos])

    def _label(v):
        if _isna(v):
            return math.nan
        for idx, threshold in enumerate(bins):
            if v <= threshold:
                return idx if labels is False else labels[idx]
        return (q - 1) if labels is False else labels[-1]

    return Series([_label(v) for v in series._values], index=series.index[:])


def Series_constructor(*args, **kwargs):
    return Series(*args, **kwargs)


def DataFrame_constructor(*args, **kwargs):
    return DataFrame(*args, **kwargs)


__all__ = [
    "Series",
    "DataFrame",
    "Index",
    "concat",
    "isna",
    "read_csv",
    "date_range",
    "to_datetime",
    "qcut",
]

