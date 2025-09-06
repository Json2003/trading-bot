import zipfile
from pathlib import Path
from typing import Optional
import tempfile

import pandas as pd


def load_zipped_csv(zip_path: str | Path, csv_filename: Optional[str] = None) -> pd.DataFrame:
    """Load the first CSV from a ZIP archive into a DataFrame.

    Works with the lightweight pandas stub used in tests by writing the CSV to a
    temporary file when file-like objects are unsupported.
    """
    with zipfile.ZipFile(zip_path, "r") as z:
        if csv_filename is None:
            csv_filename = z.namelist()[0]
        with z.open(csv_filename) as csv_file:
            try:
                df = pd.read_csv(csv_file)
            except TypeError:
                data = csv_file.read()
                with tempfile.NamedTemporaryFile("wb", delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    df = pd.read_csv(tmp_path)
                finally:
                    Path(tmp_path).unlink(missing_ok=True)
    return df


if __name__ == "__main__":  # pragma: no cover - manual invocation helper
    import argparse

    parser = argparse.ArgumentParser(description="Load a zipped CSV into pandas")
    parser.add_argument("zip_path", help="Path to ZIP file containing a CSV")
    parser.add_argument("csv_filename", nargs="?", help="CSV filename inside the ZIP")
    args = parser.parse_args()

    dataframe = load_zipped_csv(args.zip_path, args.csv_filename)
    print(dataframe.head())
