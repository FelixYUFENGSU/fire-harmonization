"""Loading helpers for the FIRMS active-fire CSVs (MODIS and VIIRS).

The CSVs were exported from Google Earth Engine, so they have a few quirks:
  * there is no latitude/longitude column - the position is inside a ``.geo``
    GeoJSON string such as ``{"type":"Point","coordinates":[lon,lat]}``;
  * ``system:index`` and ``.geo`` are Earth Engine bookkeeping columns;
  * ``acq_time`` is an integer HHMM (UTC) that loses its leading zeros
    (``124`` means 01:24);
  * VIIRS 2020 mixes day/night codes ``D``/``N`` with ``1``/``0``.
"""

from pathlib import Path

import pandas as pd

SENSORS = ("modis", "viirs")

# Earth Engine bookkeeping columns we do not need after extracting coordinates.
EE_ARTIFACT_COLUMNS = ["system:index", ".geo"]

# Some files code day/night as numbers. Checked against acquisition times:
# code 1 is acquired ~19:30-22:20 UTC (midday in California) -> day,
# code 0 is acquired ~08:00-11:30 UTC (small hours in California) -> night.
DAYNIGHT_MAP = {"D": "D", "N": "N", "1": "D", "0": "N"}


def find_files(sensor: str, data_dir) -> list[Path]:
    """Return all ``<sensor>_YYYY_*.csv`` files under data_dir (searches subfolders)."""
    sensor = sensor.lower()
    if sensor not in SENSORS:
        raise ValueError(f"sensor must be one of {SENSORS}, got {sensor!r}")
    return sorted(Path(data_dir).rglob(f"{sensor}_*.csv"))


def _parse_dates(acq_date: pd.Series, acq_time: pd.Series | None) -> pd.Series:
    """Turn acq_date (+ acq_time) into a datetime column (UTC, timezone-naive)."""
    if pd.api.types.is_numeric_dtype(acq_date):
        # Epoch milliseconds (an Earth Engine date export).
        return pd.to_datetime(acq_date, unit="ms")
    # Strings like "2015-06-10". Some exports hold epoch ms as text, so try that too.
    as_number = pd.to_numeric(acq_date, errors="coerce")
    if as_number.notna().all():
        return pd.to_datetime(as_number, unit="ms")
    dates = pd.to_datetime(acq_date, errors="coerce")
    if acq_time is not None:
        # "124" -> "0124" -> 1 hour 24 minutes after midnight UTC.
        hhmm = pd.to_numeric(acq_time, errors="coerce").fillna(0).astype(int)
        dates = dates + pd.to_timedelta(hhmm // 100, unit="h") + pd.to_timedelta(hhmm % 100, unit="m")
    return dates


def _read_one(path: Path) -> pd.DataFrame:
    """Read a single CSV and clean it up."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Pull latitude/longitude out of the GeoJSON text if they are not columns.
    if ".geo" in df.columns and "latitude" not in df.columns:
        xy = df[".geo"].str.extract(r"\[\s*(-?[\d.eE+-]+)\s*,\s*(-?[\d.eE+-]+)\s*\]").astype(float)
        df["longitude"], df["latitude"] = xy[0], xy[1]

    df = df.drop(columns=[c for c in EE_ARTIFACT_COLUMNS if c in df.columns])

    # Some columns can be read as numbers or text depending on the file; normalise.
    for col in ("daynight", "confidence", "satellite"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    df["daynight"] = df["daynight"].map(DAYNIGHT_MAP).fillna(df["daynight"])

    df["datetime"] = _parse_dates(df["acq_date"], df.get("acq_time"))
    df["file_year"] = int(path.stem.split("_")[1])  # year written in the filename
    return df


def load_sensor(sensor: str, data_dir) -> pd.DataFrame:
    """Load every CSV for "modis" or "viirs" into one DataFrame.

    Extra columns added: ``datetime`` (UTC), ``year`` (from the date),
    ``file_year`` (from the filename), ``sensor``. Returns an empty DataFrame if
    no files are found.
    """
    files = find_files(sensor, data_dir)
    if not files:
        return pd.DataFrame()
    df = pd.concat([_read_one(f) for f in files], ignore_index=True)
    df["year"] = df["datetime"].dt.year
    df["sensor"] = sensor.lower()
    return df
