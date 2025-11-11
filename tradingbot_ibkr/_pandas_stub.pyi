from typing import Any, Iterable, Dict

class DataFrame:
    @staticmethod
    def from_records(records: Iterable[Dict[str, Any]]) -> Any: ...


def read_csv(path: object, *args: object, **kwargs: object) -> Any: ...

def to_datetime(arg: object, *args: object, **kwargs: object) -> Any: ...
