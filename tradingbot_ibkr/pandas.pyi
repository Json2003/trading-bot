# typing shim for the package-local pandas fallback
from typing import Any, Iterable, Dict

class DataFrame: ...

def read_csv(path: object, *args: object, **kwargs: object) -> Any: ...

def to_datetime(arg: object, *args: object, **kwargs: object) -> Any: ...

# minimal helper to satisfy mypy about DataFrame.from_records
class _DF:
    @staticmethod
    def from_records(records: Iterable[Dict[str, Any]]) -> Any: ...

DataFrame = _DF  # type: ignore
