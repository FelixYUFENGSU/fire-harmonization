"""Data-quality checks for the MODIS and VIIRS California CSVs."""

from pathlib import Path

import pytest

from fire_harmonization.io import find_files, load_sensor

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
YEARS = range(2012, 2021)
CA_LAT, CA_LON = (32.0, 42.1), (-124.5, -114.0)

REQUIRED = {
    "modis": {"latitude", "longitude", "frp", "acq_date", "acq_time", "satellite", "confidence",
              "daynight", "scan", "track", "type", "brightness", "bright_t31"},
    "viirs": {"latitude", "longitude", "frp", "acq_date", "acq_time", "satellite", "confidence",
              "daynight", "scan", "track", "bright_ti4", "bright_ti5"},
}

if not any(DATA_DIR.rglob("*.csv")):
    pytest.skip("data/ has no CSV files", allow_module_level=True)


@pytest.fixture(scope="module", params=["modis", "viirs"])
def sensor_df(request):
    """Load each sensor once and share it between tests."""
    df = load_sensor(request.param, DATA_DIR)
    assert len(df) > 0, f"No {request.param} rows loaded from {DATA_DIR}"
    return request.param, df


def test_all_18_files_exist():
    missing = [f"{s}_{y}_CA.csv" for s in ("modis", "viirs") for y in YEARS
               if not any(p.name == f"{s}_{y}_CA.csv" for p in find_files(s, DATA_DIR))]
    assert not missing, f"Missing data files: {missing}"


def test_required_columns(sensor_df):
    sensor, df = sensor_df
    missing = REQUIRED[sensor] - set(df.columns)
    assert not missing, f"{sensor}: missing columns {sorted(missing)}"


def test_earth_engine_columns_dropped(sensor_df):
    sensor, df = sensor_df
    assert not {"system:index", ".geo"} & set(df.columns), f"{sensor}: Earth Engine columns remain"


def test_lat_lon_in_california(sensor_df):
    sensor, df = sensor_df
    assert df[["latitude", "longitude"]].notna().all().all(), f"{sensor}: missing coordinates"
    bad = df[~df.latitude.between(*CA_LAT) | ~df.longitude.between(*CA_LON)]
    assert bad.empty, (f"{sensor}: {len(bad)} points outside California's bounding box "
                       f"lat {CA_LAT}, lon {CA_LON}. Example:\n{bad[['latitude', 'longitude']].head()}")


def test_frp_non_negative(sensor_df):
    sensor, df = sensor_df
    assert df.frp.notna().all(), f"{sensor}: {df.frp.isna().sum()} missing FRP values"
    assert (df.frp >= 0).all(), f"{sensor}: {(df.frp < 0).sum()} negative FRP values (min {df.frp.min()})"


def test_dates_in_range_and_match_filename(sensor_df):
    sensor, df = sensor_df
    assert df.datetime.notna().all(), f"{sensor}: {df.datetime.isna().sum()} unparseable dates"
    out = df[~df.year.between(2012, 2020)]
    assert out.empty, f"{sensor}: {len(out)} rows outside 2012-2020 (years: {sorted(out.year.unique())})"
    mismatch = df[df.year != df.file_year]
    assert mismatch.empty, (f"{sensor}: {len(mismatch)} rows whose date year differs from the filename year. "
                            f"Example:\n{mismatch[['acq_date', 'file_year']].head()}")


def test_daynight_only_d_or_n(sensor_df):
    sensor, df = sensor_df
    bad = set(df.daynight.unique()) - {"D", "N"}
    assert not bad, f"{sensor}: unexpected daynight values {bad} (loader maps 0->N, 1->D)"


def test_modis_confidence_0_to_100():
    df = load_sensor("modis", DATA_DIR)
    conf = df.confidence.astype(float)
    assert conf.between(0, 100).all(), f"MODIS confidence outside 0-100: min {conf.min()}, max {conf.max()}"


def test_viirs_confidence_categories():
    df = load_sensor("viirs", DATA_DIR)
    unknown = set(df.confidence.unique()) - {"l", "n", "h", "low", "nominal", "high"}
    assert not unknown, f"VIIRS confidence has unrecognized values: {unknown}"
