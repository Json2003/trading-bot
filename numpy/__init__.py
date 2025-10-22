"""Very small NumPy-compatible stub used for tests.

The real project depends on :mod:`numpy` and :mod:`pandas`, however the kata
environment purposely ships without binary wheels.  The unit tests exercise a
fairly small subset of NumPy features (array creation, basic arithmetic,
rolling statistics, random helpers).  Implementing the entire NumPy API would
be unnecessary; instead this module implements only the pieces that are
required by the tests.  The goal is API-compatibility rather than raw
performance, therefore the implementation relies solely on pure Python.

The public surface mimics the real package sufficiently for the production
code to operate:

* ``array``/``asarray`` constructors backed by a light-weight ``ndarray``
  class that implements the arithmetic dunder methods used throughout the
  codebase.
* ``zeros``/``ones``/``zeros_like`` helpers.
* Math helpers such as ``sqrt``, ``exp``, ``tanh``, ``clip`` and ``sign`` that
  transparently operate on either scalars or ``ndarray`` instances.
* Reductions (``mean``, ``std``, ``sum``), window operations
  (``maximum.accumulate``), broadcasting aware comparisons and ``where``.
* Random number generation through ``random.default_rng`` plus module level
  fallbacks for ``normal`` and ``lognormal`` used by the scripts.

This is intentionally tiny but well documented so future contributors know
where to extend it when a missing attribute shows up in a unit test failure.
"""

from __future__ import annotations

import math
import random as _stdlib_random
from typing import Iterable, Iterator, List, Sequence, Tuple, Union, overload
import builtins as _bi

__all__ = [
    "__version__",
    "array",
    "asarray",
    "ndarray",
    "zeros",
    "zeros_like",
    "ones",
    "roll",
    "mean",
    "std",
    "sum",
    "sqrt",
    "exp",
    "tanh",
    "clip",
    "sign",
    "where",
    "maximum",
    "minimum",
    "concatenate",
    "abs",
    "random",
]

# ``pandas`` queries ``numpy.__version__`` during import to adjust behaviour for
# specific releases.  The light-weight stub historically omitted the attribute
# which caused ``AttributeError`` when pandas (and therefore our test suite)
# attempted to import.  Providing a modern sentinel version string keeps the
# import contract intact while remaining explicit that this is a compatibility
# shim.
__version__ = "1.26.0"

# Common dtype aliases expected by some libraries/tests
int_ = int
int64 = int
uint = int

Number = Union[int, float]


def _ensure_list(value):
    if isinstance(value, ndarray):
        return value._data
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _ensure_float(value: Number) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    return float(value)


class ndarray:
    """Extremely small ndarray implementation backed by nested lists."""

    __array_priority__ = 1000

    def __init__(self, data):
        if isinstance(data, ndarray):
            data = data._data
        self._data = self._deep_copy(data)

    # ------------------------------------------------------------------
    # Container protocol helpers
    # ------------------------------------------------------------------
    def __iter__(self) -> Iterator:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, item):
        if isinstance(self._data, list):
            return self._wrap(self._data[item])
        return self._data[item]

    def __setitem__(self, key, value):
        if isinstance(self._data, list):
            if isinstance(value, ndarray):
                value = value._data
            self._data[key] = value
        else:
            raise TypeError("ndarray does not support item assignment for scalars")

    # ------------------------------------------------------------------
    # Representation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _deep_copy(value):
        if isinstance(value, ndarray):
            return ndarray._deep_copy(value._data)
        if isinstance(value, list):
            return [ndarray._deep_copy(v) for v in value]
        if isinstance(value, tuple):
            return [ndarray._deep_copy(v) for v in value]
        return value

    @staticmethod
    def _wrap(value):
        if isinstance(value, list):
            return ndarray(value)
        return value

    # ------------------------------------------------------------------
    # Basic stats and conversions
    # ------------------------------------------------------------------
    @property
    def shape(self) -> Tuple[int, ...]:
        if not isinstance(self._data, list):
            return ()
        if not self._data:
            return (0,)
        first = self._data[0]
        if isinstance(first, list):
            inner_shape = ndarray(first).shape
            return (len(self._data),) + inner_shape
        return (len(self._data),)

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        if not isinstance(self._data, list):
            return 1
        if not self._data:
            return 0
        if isinstance(self._data[0], list):
            return sum(ndarray(v).size for v in self._data)
        return len(self._data)

    def copy(self) -> "ndarray":
        return ndarray(self._data)

    def astype(self, dtype) -> "ndarray":
        if callable(dtype):
            convert = dtype
        else:
            convert = dtype  # type: ignore[assignment]

        def _convert(value):
            if isinstance(value, list):
                return [_convert(v) for v in value]
            try:
                return convert(value)
            except Exception:
                return value

        return ndarray(_convert(self._data))

    def reshape(self, *shape: int) -> "ndarray":
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        flat = self.flatten()._data
        total = 1
        for s in shape:
            total *= s
        if total != len(flat):
            raise ValueError("cannot reshape array")
        if len(shape) == 1:
            return ndarray(flat[: shape[0]])
        out: List = []
        idx = 0
        for _ in range(shape[0]):
            row = []
            for _ in range(shape[1]):
                row.append(flat[idx])
                idx += 1
            out.append(row)
        return ndarray(out)

    def flatten(self) -> "ndarray":
        if not isinstance(self._data, list):
            return ndarray([self._data])
        if self.ndim <= 1:
            return ndarray(self._data[:])
        out: List = []
        for item in self._data:
            out.extend(ndarray(item).flatten()._data)
        return ndarray(out)

    def tolist(self):
        if not isinstance(self._data, list):
            return [self._data]
        return [ndarray(v).tolist() if isinstance(v, list) else v for v in self._data]

    # ------------------------------------------------------------------
    # Reduction helpers
    # ------------------------------------------------------------------
    def sum(self) -> float:
        return float(sum(self._iter_flat()))

    def mean(self) -> float:
        flat = list(self._iter_flat())
        return float(sum(flat) / len(flat)) if flat else 0.0

    def std(self, ddof: int = 0) -> float:
        flat = list(self._iter_flat())
        if not flat:
            return 0.0
        mean = sum(flat) / len(flat)
        var = sum((x - mean) ** 2 for x in flat)
        denom = _bi.max(1, len(flat) - ddof)
        return float(math.sqrt(var / denom))

    def _iter_flat(self) -> Iterator[float]:
        if isinstance(self._data, list):
            for v in self._data:
                if isinstance(v, list):
                    yield from ndarray(v)._iter_flat()
                else:
                    yield _ensure_float(v)
        else:
            yield _ensure_float(self._data)

    # ------------------------------------------------------------------
    # Unary / binary ops
    # ------------------------------------------------------------------
    def _binary(self, other, op):
        if isinstance(other, ndarray):
            other = other._data
        return ndarray(_broadcast_binary(self._data, other, op))

    def _unary(self, op):
        if isinstance(self._data, list):
            return ndarray(
                [ndarray(v)._unary(op)._data if isinstance(v, list) else op(v) for v in self._data]
            )
        return ndarray(op(self._data))

    def __add__(self, other):
        return self._binary(other, lambda a, b: a + b)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return self._binary(other, lambda a, b: a - b)

    def __rsub__(self, other):
        return ndarray(other).__sub__(self)

    def __mul__(self, other):
        return self._binary(other, lambda a, b: a * b)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        def _div(a, b):
            if isinstance(a, list):
                return [_div(x, b) for x in a]
            if isinstance(b, list):
                return [_div(a, x) for x in b]
            try:
                return a / b
            except ZeroDivisionError:
                return float("inf")

        return self._binary(other, _div)

    def __rtruediv__(self, other):
        return ndarray(other).__truediv__(self)

    def __neg__(self):
        return self._unary(lambda a: -a)

    def __abs__(self):
        return self._unary(abs)
def abs(x):  # type: ignore[override]
    if isinstance(x, ndarray):
        return x.__abs__()
    try:
        return __builtins__["abs"](x)  # type: ignore[index]
    except Exception:
        return x

    def __lt__(self, other):
        return self._binary(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._binary(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._binary(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._binary(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._binary(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._binary(other, lambda a, b: a != b)

    def __matmul__(self, other):
        if isinstance(other, ndarray):
            other = other._data
        return ndarray(_matmul(self._data, other))

    def __rmatmul__(self, other):
        return ndarray(other).__matmul__(self)

    @property
    def T(self):
        if self.ndim <= 1:
            return self.copy()
        rows = len(self._data)
        cols = len(self._data[0]) if rows else 0
        transposed = [[self._data[r][c] for r in range(rows)] for c in range(cols)]
        return ndarray(transposed)


def _broadcast_binary(a, b, op):
    if isinstance(a, list) and any(isinstance(x, list) for x in a):
        if isinstance(b, list) and any(isinstance(x, list) for x in b):
            return [_broadcast_binary(x, y, op) for x, y in zip(a, b)]
        return [_broadcast_binary(x, b, op) for x in a]
    if isinstance(a, list):
        if isinstance(b, list):
            return [op(x, y) for x, y in zip(a, b)]
        return [op(x, b) for x in a]
    if isinstance(b, list):
        return [op(a, y) for y in b]
    return op(a, b)


def _matmul(a, b):
    if not isinstance(a, list):
        a = [a]
    if isinstance(b, list) and any(isinstance(x, list) for x in b):
        # matrix @ matrix/column vector
        if isinstance(b[0], list):
            cols = len(b[0])
            rows = len(a)
            out = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    val = 0.0
                    for k in range(len(b)):
                        val += _ensure_float(a[r][k]) * _ensure_float(b[k][c])
                    row.append(val)
                out.append(row)
            return out
        # matrix @ vector
        out_vec = []
        for row in a:
            val = 0.0
            for i, coeff in enumerate(row):
                val += _ensure_float(coeff) * _ensure_float(b[i])
            out_vec.append(val)
        return out_vec
    # vector dot product
    if isinstance(b, list) and not any(isinstance(x, list) for x in b):
        total = 0.0
        for x, y in zip(a, b):
            total += _ensure_float(x) * _ensure_float(y)
        return total
    if isinstance(b, list) and any(isinstance(x, list) for x in b):
        return _matmul([a], b)
    raise TypeError("unsupported matmul operands")


def array(data, dtype=None):
    if hasattr(data, "_values"):
        data = getattr(data, "_values")
        if callable(data):
            data = data()
        if hasattr(data, "_values"):
            data = data._values
    arr = ndarray(data)
    return arr.astype(dtype) if dtype is not None else arr


def asarray(data, dtype=None):
    return array(data, dtype=dtype)


def zeros(shape, dtype=float):
    if isinstance(shape, tuple):
        if len(shape) == 2:
            return ndarray([[dtype() for _ in range(shape[1])] for _ in range(shape[0])])
    return ndarray([dtype() for _ in range(int(shape))])


def ones(shape, dtype=float):
    if isinstance(shape, tuple):
        if len(shape) == 2:
            return ndarray([[dtype(1) for _ in range(shape[1])] for _ in range(shape[0])])
    return ndarray([dtype(1) for _ in range(int(shape))])


def zeros_like(arr):
    return zeros(array(arr).shape, dtype=float)


def ones_like(arr):
    return ones(array(arr).shape, dtype=float)


def sqrt(x):
    if isinstance(x, ndarray):
        return x._unary(lambda a: math.sqrt(a))
    return math.sqrt(x)


def exp(x):
    if isinstance(x, ndarray):
        return x._unary(lambda a: math.exp(a))
    return math.exp(x)


def tanh(x):
    if isinstance(x, ndarray):
        return x._unary(lambda a: math.tanh(a))
    return math.tanh(x)


def clip(x, a_min, a_max):
    def _clip(v):
        if a_min is not None and v < a_min:
            v = a_min
        if a_max is not None and v > a_max:
            v = a_max
        return v

    if isinstance(x, ndarray):
        return x._unary(_clip)
    return _clip(x)


def sign(x):
    def _sign(v):
        if v > 0:
            return 1.0
        if v < 0:
            return -1.0
        return 0.0

    if isinstance(x, ndarray):
        return x._unary(_sign)
    return _sign(x)


def mean(x, axis=None):
    arr = array(x)
    if axis is None:
        return arr.mean()
    if axis == 0:
        cols = arr.shape[1] if arr.ndim > 1 else 1
        out = []
        for c in range(cols):
            col = [row[c] for row in arr._data]
            out.append(sum(col) / len(col) if col else 0.0)
        return ndarray(out)
    raise NotImplementedError("axis != 0 not implemented in stub")


def std(x, ddof=0):
    return array(x).std(ddof=ddof)


def sum(x):  # type: ignore[override]
    arr = array(x)
    return arr.sum()


def max(x):  # type: ignore[override]
    arr = array(x)
    values = list(arr._iter_flat())
    return float(max(values)) if values else float("nan")


def min(x):  # type: ignore[override]
    arr = array(x)
    values = list(arr._iter_flat())
    return float(min(values)) if values else float("nan")


nan = float("nan")
inf = float("inf")


def isnan(x):
    if isinstance(x, ndarray):
        return x._unary(lambda a: math.isnan(a))
    return math.isnan(x)


def where(condition, x, y):
    cond = array(condition)
    return ndarray(_where(cond._data, x, y))


def _where(cond, x, y):
    if isinstance(cond, list):
        return [_where(c, x, y) for c in cond]
    return x if cond else y


def diff(x):
    arr = array(x).flatten()._data
    return ndarray([arr[i + 1] - arr[i] for i in range(len(arr) - 1)])


def cumsum(x):
    arr = array(x).flatten()._data
    out = []
    total = 0.0
    for v in arr:
        total += _ensure_float(v)
        out.append(total)
    return ndarray(out)


def roll(x, shift, axis=None):
    """Roll array elements along a given axis.

    Minimal 1-D implementation used by tests. For multi-dimensional inputs,
    this flattens when axis is None, or applies along axis=0.
    """
    arr = array(x)
    if arr.ndim == 0:
        return arr.copy()
    data = arr.flatten()._data if axis is None or arr.ndim == 1 else arr._data
    n = len(data)
    if n == 0:
        return ndarray([])
    s = int(shift) % n
    if s == 0:
        return ndarray(data[:]) if (axis is None or arr.ndim == 1) else ndarray(data)
    rolled = data[-s:] + data[:-s]
    return ndarray(rolled)


def arange(start, stop=None, step=1):
    if stop is None:
        stop = start
        start = 0
    values = []
    current = start
    while (step > 0 and current < stop) or (step < 0 and current > stop):
        values.append(current)
        current += step
    return ndarray(values)


def linspace(start, stop, num):
    if num <= 1:
        return ndarray([start])
    step = (stop - start) / (num - 1)
    return ndarray([start + i * step for i in range(num)])


def vstack(arrays: Sequence[Iterable]) -> ndarray:
    stacked = [array(a).flatten()._data for a in arrays]
    return ndarray(stacked)


class _MaximumHelper:
    def accumulate(self, arr):
        values = array(arr).flatten()._data
        out = []
        current = -float("inf")
        for v in values:
            if v > current:
                current = v
            out.append(current)
        return ndarray(out)


class _MaximumUfunc:
    def __call__(self, a, b):
        aa = array(a)
        bb = array(b)
        def _elem(x, y):
            return x if x >= y else y
        return ndarray(_broadcast_binary(aa._data, bb._data, _elem))

    def accumulate(self, arr):
        values = array(arr).flatten()._data
        out = []
        current = -float("inf")
        for v in values:
            if v > current:
                current = v
            out.append(current)
        return ndarray(out)

maximum = _MaximumUfunc()

def minimum(a, b):
    aa = array(a)
    bb = array(b)
    def _elem(x, y):
        return x if x <= y else y
    return ndarray(_broadcast_binary(aa._data, bb._data, _elem))


def nanmin(x):
    values = [v for v in array(x)._iter_flat() if not math.isnan(v)]
    return min(values) if values else float("nan")


def nanpercentile(x, q):
    values = sorted(v for v in array(x)._iter_flat() if not math.isnan(v))
    if not values:
        return float("nan")
    pos = int(round((q / 100.0) * (len(values) - 1)))
    pos = _bi.max(0, _bi.min(len(values) - 1, pos))
    return float(values[pos])


def quantile(x, q):
    values = sorted(array(x)._iter_flat())
    if not values:
        return float("nan")
    pos = q * (len(values) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return float(values[low])
    frac = pos - low
    return float(values[low] * (1 - frac) + values[high] * frac)


# Minimal datetime64 constructor for compatibility
def datetime64(value, unit=None):
    # Return Python datetime or pass-through string/number if parsing fails.
    try:
        from datetime import datetime
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value)
            except Exception:
                return value
        return value
    except Exception:
        return value


def timedelta64(value, unit=None):
    try:
        from datetime import timedelta
        if isinstance(value, (int, float)):
            # Interpret numeric with unit (only basic units supported)
            unit = (unit or 's').lower()
            if unit in ('s', 'sec', 'second', 'seconds'):
                return timedelta(seconds=float(value))
            if unit in ('ms', 'millisecond', 'milliseconds'):
                return timedelta(milliseconds=float(value))
            if unit in ('us', 'microsecond', 'microseconds'):
                return timedelta(microseconds=float(value))
            if unit in ('m', 'min', 'minute', 'minutes'):
                return timedelta(minutes=float(value))
            if unit in ('h', 'hour', 'hours'):
                return timedelta(hours=float(value))
            if unit in ('d', 'day', 'days'):
                return timedelta(days=float(value))
            return timedelta(seconds=float(value))
        return value
    except Exception:
        return value


def array_equal(a, b):
    return array(a).tolist() == array(b).tolist()


# ----------------------------------------------------------------------
# Random helpers
# ----------------------------------------------------------------------


class _RandomModule:
    def __init__(self):
        self._rng = _stdlib_random.Random()

    def seed(self, value: int | None = None) -> None:
        self._rng.seed(value)

    class _Generator:
        def __init__(self, seed: int | None):
            self._rng = _stdlib_random.Random(seed)

        def _normal(self, loc: float, scale: float):
            return self._rng.gauss(loc, scale)

        def normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
            if size is None:
                return self._normal(loc, scale)
            if isinstance(size, tuple):
                if len(size) == 2:
                    return ndarray(
                        [[self._normal(loc, scale) for _ in range(size[1])] for _ in range(size[0])]
                    )
            return ndarray([self._normal(loc, scale) for _ in range(int(size))])

        def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size=None):
            def _one():
                return math.exp(self._normal(mean, sigma))

            if size is None:
                return _one()
            return ndarray([_one() for _ in range(int(size))])

        def random(self, size=None):
            if size is None:
                return self._rng.random()
            return ndarray([self._rng.random() for _ in range(int(size))])

        def uniform(self, low: float = 0.0, high: float = 1.0, size=None):
            if size is None:
                return self._rng.uniform(low, high)
            return ndarray([self._rng.uniform(low, high) for _ in range(int(size))])

        def integers(self, low: int, high: int | None = None, size=None):
            if high is None:
                low, high = 0, low
            if size is None:
                return self._rng.randrange(low, high)
            return ndarray([self._rng.randrange(low, high) for _ in range(int(size))])

        def choice(self, seq: Sequence, size=None):
            if size is None:
                return self._rng.choice(list(seq))
            return ndarray([self._rng.choice(list(seq)) for _ in range(int(size))])

    def default_rng(self, seed: int | None = None):
        return self._Generator(seed)

    def normal(self, loc: float = 0.0, scale: float = 1.0, size=None):
        return self.default_rng().normal(loc=loc, scale=scale, size=size)

    def lognormal(self, mean: float = 0.0, sigma: float = 1.0, size=None):
        return self.default_rng().lognormal(mean=mean, sigma=sigma, size=size)

    def random(self, size=None):
        return self.default_rng().random(size=size)

    def uniform(self, low: float = 0.0, high: float = 1.0, size=None):
        return self.default_rng().uniform(low=low, high=high, size=size)

    def default_random(self):
        return self._rng.random()


random = _RandomModule()


def max_(x):  # pragma: no cover - alias for completeness
    return max(x)


__all__ = [
    "array",
    "asarray",
    "zeros",
    "ones",
    "zeros_like",
    "ones_like",
    "sqrt",
    "exp",
    "tanh",
    "clip",
    "sign",
    "mean",
    "std",
    "sum",
    "max",
    "min",
    "diff",
    "cumsum",
    "arange",
    "linspace",
    "vstack",
    "where",
    "nan",
    "inf",
    "isnan",
    "nanmin",
    "nanpercentile",
    "quantile",
    "maximum",
    "array_equal",
    "random",
    "ndarray",
]
