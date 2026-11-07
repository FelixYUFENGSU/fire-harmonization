import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import folium

    from fire_harmonization.io import load_sensor

    return folium, load_sensor, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Understanding the MODIS and VIIRS fire data (California, 2012-2020)

    Every row in these files is one **active-fire detection**: a satellite pixel
    (about 1 km for MODIS, 375 m for VIIRS) that looked hot enough to contain fire.
    The key number is **FRP, Fire Radiative Power** (megawatts), roughly how much
    heat energy the fire was radiating at that moment.

    Our project goal: make FRP from MODIS and VIIRS comparable, so the fire
    record can continue after MODIS shuts down. This notebook just *looks* at
    the raw data.
    """)
    return


@app.cell
def _(load_sensor, mo):
    # The data lives in <repo>/data (the loader searches sub-folders too).
    data_dir = mo.notebook_dir().parent / "data"
    modis = load_sensor("modis", data_dir)
    viirs = load_sensor("viirs", data_dir)

    # MODIS flies on two satellites. Aqua is the one being retired in 2027,
    # so for fair comparisons we also keep an Aqua-only table.
    modis_aqua = modis[modis["satellite"] == "Aqua"]
    return modis, modis_aqua, viirs


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. How big is the data?

    Row counts per sensor and per year, then the columns each sensor has, their
    types (`dtype`) and how many values are missing.
    """)
    return


@app.cell
def _(modis, pd, viirs):
    counts_by_year = pd.DataFrame({
        "modis (Aqua+Terra)": modis.groupby("year").size(),
        "viirs": viirs.groupby("year").size(),
    })
    counts_by_year.loc["total"] = counts_by_year.sum()
    counts_by_year
    return


@app.cell
def _(mo, modis, pd, viirs):
    # One table per sensor: column name, type, number of missing values.
    _tables = []
    for _name, _df in [("MODIS", modis), ("VIIRS", viirs)]:
        _info = pd.DataFrame({
            "dtype": _df.dtypes.astype(str),
            "missing": _df.isna().sum(),
        })
        _tables.append(mo.vstack([mo.md(f"**{_name}** - {len(_df.columns)} columns"), _info]))
    mo.hstack(_tables, align="start", gap=3)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. How are categorical fields coded?

    - `satellite`: MODIS = Aqua / Terra, VIIRS = N (Suomi NPP).
    - `confidence`: MODIS is a number 0-100; VIIRS is a category (l = low, n = nominal, h = high).
    - `daynight`: D / N. (The raw VIIRS 2020 file uses 1/0 instead; the loader converts them to D/N.)
    - `type`: 0 vegetation fire, 1 volcano, 2 static land source (e.g. industrial flare), 3 offshore.
    """)
    return


@app.cell
def _(mo, modis, pd, viirs):
    _blocks = []
    for _col in ["satellite", "daynight", "type"]:
        _t = pd.DataFrame({
            "MODIS": modis[_col].value_counts(),
            "VIIRS": viirs[_col].value_counts(),
        }).fillna(0).astype(int)
        _blocks.append(mo.vstack([mo.md(f"**{_col}**"), _t]))

    # MODIS confidence is numeric: bin it. VIIRS confidence is already categories.
    _bins = pd.cut(modis["confidence"].astype(float), [-1, 29, 79, 100],
                   labels=["0-29 (low)", "30-79 (nominal)", "80-100 (high)"])
    _blocks.append(mo.vstack([mo.md("**MODIS confidence (binned)**"), _bins.value_counts().sort_index()]))
    _blocks.append(mo.vstack([mo.md("**VIIRS confidence**"), viirs["confidence"].value_counts()]))
    mo.hstack(_blocks, align="start", gap=2, wrap=True)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. FRP: how big is a typical detection?

    Summary statistics, then a histogram. FRP spans orders of magnitude, so
    the x-axis is logarithmic. VIIRS pixels are ~7x smaller in area, so each
    one sees less fire and reports a smaller FRP.
    """)
    return


@app.cell
def _(modis, pd, viirs):
    frp_stats = pd.DataFrame({
        "MODIS (all)": modis["frp"].describe(percentiles=[0.5, 0.9, 0.99]),
        "VIIRS": viirs["frp"].describe(percentiles=[0.5, 0.9, 0.99]),
    }).round(2)
    frp_stats
    return


@app.cell
def _(modis, np, plt, viirs):
    _fig, _ax = plt.subplots(figsize=(8, 4))
    # Logarithmically spaced bins; FRP of exactly 0 cannot be shown on a log axis.
    _edges = np.logspace(-1.5, 4, 60)
    _ax.hist(modis["frp"].clip(lower=0.03), bins=_edges, alpha=0.6, color="tab:red", label="MODIS")
    _ax.hist(viirs["frp"].clip(lower=0.03), bins=_edges, alpha=0.6, color="tab:orange", label="VIIRS")
    _ax.set_xscale("log")
    _ax.set_xlabel("FRP per detection (MW, log scale)")
    _ax.set_ylabel("Number of detections")
    _ax.legend()
    _ax.set_title("FRP per detection")
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The core gap: yearly detections and total FRP

    MODIS here is **Aqua only** (the satellite we need to replace). Total FRP
    is the sum of FRP over all detections in a year. The ratio VIIRS / MODIS
    is the gap we want to harmonize. If the sensors agreed, the ratio would be 1.
    """)
    return


@app.cell
def _(modis_aqua, pd, viirs):
    yearly = pd.DataFrame({
        "modis_n": modis_aqua.groupby("year").size(),
        "viirs_n": viirs.groupby("year").size(),
        "modis_frp": modis_aqua.groupby("year")["frp"].sum(),
        "viirs_frp": viirs.groupby("year")["frp"].sum(),
    })
    yearly["ratio_n"] = yearly["viirs_n"] / yearly["modis_n"]
    yearly["ratio_frp"] = yearly["viirs_frp"] / yearly["modis_frp"]
    yearly.round(2)
    return (yearly,)


@app.cell
def _(plt, yearly):
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))
    for _ax, _col, _title in [(_axes[0], "n", "Detections per year"), (_axes[1], "frp", "Total FRP per year (MW)")]:
        _ax.plot(yearly.index, yearly[f"modis_{_col}"], "o-", color="tab:red", label="MODIS (Aqua)")
        _ax.plot(yearly.index, yearly[f"viirs_{_col}"], "o-", color="tab:orange", label="VIIRS")
        _ax.set_title(_title)
        _ax.legend()
    _axes[2].plot(yearly.index, yearly["ratio_n"], "o-", label="detections")
    _axes[2].plot(yearly.index, yearly["ratio_frp"], "o-", label="total FRP")
    _axes[2].axhline(1, color="grey", ls="--")
    _axes[2].set_title("VIIRS / MODIS ratio")
    _axes[2].legend()
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Day vs night

    MODIS and VIIRS differ in how well they see fires by day vs night
    (daytime reflected sunlight, different overpass times), so we check the split.
    """)
    return


@app.cell
def _(modis_aqua, pd, viirs):
    daynight_table = pd.DataFrame({
        "MODIS Aqua detections": modis_aqua["daynight"].value_counts(),
        "VIIRS detections": viirs["daynight"].value_counts(),
    })
    daynight_share = (daynight_table / daynight_table.sum() * 100).round(1).add_suffix(" (%)")
    daynight_median_frp = pd.DataFrame({
        "MODIS Aqua median FRP": modis_aqua.groupby("daynight")["frp"].median(),
        "VIIRS median FRP": viirs.groupby("daynight")["frp"].median(),
    })
    pd.concat([daynight_table, daynight_share, daynight_median_frp], axis=1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Pixel footprint (scan x track)

    `scan` is the pixel width across the satellite's path and `track` the
    length along it, both in km. Pixels grow towards the edge of the swath
    (the "bow-tie" effect), so `scan` doubles as a measure of how far from
    nadir (straight below the satellite) the pixel was. Area = scan x track.
    """)
    return


@app.cell
def _(modis, pd, viirs):
    _rows = {}
    for _name, _df in [("MODIS", modis), ("VIIRS", viirs)]:
        _area = _df["scan"] * _df["track"]
        _rows[_name] = pd.concat([_df["scan"].describe(), _df["track"].describe(), _area.describe()],
                                 keys=["scan (km)", "track (km)", "area (km2)"]).round(3)
    footprint_stats = pd.DataFrame(_rows).unstack(level=0)
    footprint_stats
    return


@app.cell
def _(modis, plt, viirs):
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))
    for _ax, _df, _name, _color in [(_axes[0], modis, "MODIS", "tab:red"), (_axes[1], viirs, "VIIRS", "tab:orange")]:
        # Median pixel area for each (rounded) scan size = footprint vs scan position.
        _area = (_df["scan"] * _df["track"]).groupby(_df["scan"].round(2)).median()
        _n = _df["scan"].round(2).value_counts().reindex(_area.index)
        _ax.scatter(_area.index, _area.values, s=10 + 200 * _n / _n.max(), color=_color, alpha=0.6)
        _ax.set_xlabel("scan (km) - larger = further from nadir")
        _ax.set_ylabel("median pixel area (km2)")
        _ax.set_title(f"{_name}: footprint vs scan position (dot size = detections)")
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Seasonality

    Share of each sensor's detections falling in each calendar month, all years combined.
    California's fire season peaks in late summer and autumn.
    """)
    return


@app.cell
def _(modis_aqua, pd, plt, viirs):
    _m = modis_aqua["datetime"].dt.month.value_counts().sort_index()
    _v = viirs["datetime"].dt.month.value_counts().sort_index()
    monthly = pd.DataFrame({"MODIS Aqua": _m, "VIIRS": _v})
    _fig, _ax = plt.subplots(figsize=(8, 4))
    (monthly / monthly.sum() * 100).plot.bar(ax=_ax, color=["tab:red", "tab:orange"])
    _ax.set_xlabel("Month")
    _ax.set_ylabel("% of detections")
    _fig
    return (monthly,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Map of detections

    Pick a year. Up to 2000 random MODIS (Aqua, red) and 2000 VIIRS (orange)
    points are drawn so the map stays fast.
    """)
    return


@app.cell
def _(mo):
    year_dd = mo.ui.dropdown(options=[str(y) for y in range(2012, 2021)], value="2018", label="Year")
    year_dd
    return (year_dd,)


@app.cell
def _(folium, mo, modis_aqua, viirs, year_dd):
    _year = int(year_dd.value)
    _map = folium.Map(location=[37.2, -119.5], zoom_start=6, tiles="OpenStreetMap")
    for _df, _color in [(modis_aqua, "red"), (viirs, "orange")]:
        _pts = _df[_df["year"] == _year]
        _pts = _pts.sample(min(2000, len(_pts)), random_state=0)
        for _lat, _lon, _frp in zip(_pts["latitude"], _pts["longitude"], _pts["frp"]):
            folium.CircleMarker([_lat, _lon], radius=2, color=_color, fill=True, weight=0,
                                fill_opacity=0.7, tooltip=f"FRP {_frp:.1f} MW").add_to(_map)
    mo.Html(_map._repr_html_())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 9. Persistent hotspots

    Wildfires burn for days, not for years. A location that shows fire in
    *many different months* is almost certainly not a wildfire: oil fields,
    flares, landfills or industrial plants. We snap VIIRS points to a 0.02 degree grid
    (~2 km) and count the distinct year-months each cell was active.
    """)
    return


@app.cell
def _(np, viirs):
    _cells = viirs.assign(
        cell_lat=(np.floor(viirs["latitude"] / 0.02) * 0.02 + 0.01).round(3),
        cell_lon=(np.floor(viirs["longitude"] / 0.02) * 0.02 + 0.01).round(3),
        month_id=viirs["year"] * 12 + viirs["datetime"].dt.month,
    )
    hotspots = (
        _cells.groupby(["cell_lat", "cell_lon"])
        .agg(active_months=("month_id", "nunique"), detections=("frp", "size"), mean_frp=("frp", "mean"))
        .sort_values("active_months", ascending=False)
        .head(15)
        .round(2)
        .reset_index()
    )
    hotspots
    return (hotspots,)


@app.cell
def _(folium, hotspots, mo):
    _map = folium.Map(location=[hotspots["cell_lat"].mean(), hotspots["cell_lon"].mean()],
                      zoom_start=6, tiles="OpenStreetMap")
    for _r in hotspots.itertuples():
        folium.CircleMarker([_r.cell_lat, _r.cell_lon], radius=4 + _r.active_months / 10,
                            color="purple", fill=True, fill_opacity=0.6,
                            tooltip=f"{_r.active_months} active months, {_r.detections} detections").add_to(_map)
    mo.Html(_map._repr_html_())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 10. Key findings (computed from the data)
    """)
    return


@app.cell
def _(hotspots, mo, modis, modis_aqua, monthly, viirs, yearly):
    _peak_month = monthly["VIIRS"].idxmax()
    _top = hotspots.iloc[0]
    _static_share = (viirs["type"] == 2).mean() * 100
    mo.md(f"""
    - **Volume:** {len(modis):,} MODIS detections ({len(modis_aqua):,} from Aqua) and
      {len(viirs):,} VIIRS detections, 2012-2020.
    - **VIIRS sees much more:** per year, VIIRS has on average
      {yearly['ratio_n'].mean():.1f}x the Aqua detection count (range {yearly['ratio_n'].min():.1f}-{yearly['ratio_n'].max():.1f}x),
      because its pixels are smaller.
    - **But total FRP is closer:** the VIIRS/Aqua total-FRP ratio averages
      {yearly['ratio_frp'].mean():.2f} (range {yearly['ratio_frp'].min():.2f}-{yearly['ratio_frp'].max():.2f}).
      This ratio is what harmonization must correct.
    - **Per detection**, median FRP is {viirs['frp'].median():.1f} MW for VIIRS vs
      {modis_aqua['frp'].median():.1f} MW for Aqua.
    - **Worst year for fire:** {yearly['viirs_frp'].idxmax()} by VIIRS total FRP.
    - **Season:** VIIRS detections peak in month {_peak_month}.
    - **Not all fires are wildfires:** {_static_share:.1f}% of VIIRS detections are tagged as static land sources;
      the most persistent 2 km cell ({_top.cell_lat}, {_top.cell_lon}) was active in {int(_top.active_months)} different months.
      Consider filtering these before harmonizing.
    - **Data quirks:** Earth Engine files have no lat/lon columns (taken from `.geo`),
      and VIIRS 2020 codes day/night as 1/0 instead of D/N.
    """)
    return


if __name__ == "__main__":
    app.run()
