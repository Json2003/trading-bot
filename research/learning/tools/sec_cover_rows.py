"""Extract displayed security-class facts from one local SEC inline-XBRL filing."""

import argparse
import json
from pathlib import Path
from xml.etree import ElementTree

NON_NUMERIC = "{http://www.xbrl.org/2013/inlineXBRL}nonNumeric"
FIELDS = {
    "dei:TradingSymbol": "symbol",
    "dei:Security12bTitle": "title",
    "dei:SecurityExchangeName": "exchange",
}


def read_cover_rows(path):
    """Return explicit triples sharing a context in the audited filing format."""
    contexts = {}
    for fact in ElementTree.parse(path).iter(NON_NUMERIC):
        field = FIELDS.get(fact.get("name"))
        if field is None:
            continue
        context = fact.get("contextRef")
        if not context or fact.get("continuedAt"):
            raise ValueError("Missing context or continued fact requires manual review")
        value = " ".join("".join(fact.itertext()).split())
        fields = contexts.setdefault(context, {})
        if not value or field in fields:
            raise ValueError(f"Empty or duplicate {field} in context {context}")
        fields[field] = value

    if not contexts:
        raise ValueError("No supported cover facts found")
    rows = []
    for context, fields in contexts.items():
        if fields.keys() != set(FIELDS.values()):
            raise ValueError(f"Incomplete security-class facts in context {context}")
        rows.append({"context": context, **fields})
    return sorted(rows, key=lambda row: (row["symbol"], row["context"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("filing", type=Path)
    args = parser.parse_args()
    print(json.dumps(read_cover_rows(args.filing), indent=2))
