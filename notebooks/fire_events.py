import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib
    import folium
    import branca.colormap as bcm
    from sklearn.cluster import DBSCAN

    # Re-use the project's loader: it reads every CSV, extracts lat/lon from the
    # Earth Engine ".geo" column, lowercases column names and parses dates.
    from fire_harmonization.io import load_sensor

    return DBSCAN, bcm, folium, load_sensor, matplotlib, mo, np, pd, plt


@app.cell
def _(mo):
    mo.md(r"""
    # Which real fires do the detections near Redding belong to?

    Each dot on the map is one satellite **active-fire detection**. We color the
    detections by **date** and also group them automatically in **space and time**
    (DBSCAN) so we can compare the groups against fires we know about
    (Bagley, Bully, Carr, Hirz, Delta, Zogg).

    *Tip: change the bounding box in the next cell, then everything below re-runs.*
    """)
    return


@app.cell
def _():
    # ------------------------------------------------------------------
    # EDIT ME: region of interest (degrees). Roughly NW of Redding, CA:
    # Whiskeytown, Shasta Lake, Lakehead, Shasta-Trinity National Forest.
    # ------------------------------------------------------------------
    LAT_MIN, LAT_MAX = 40.40, 41.05
    LON_MIN, LON_MAX = -122.90, -122.10
    return LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


@app.cell
def _(pd):
    # Known fires used for labelling / validation. Add more rows here later.
    # Bagley's ignition point is uncertain (likely at/outside the east edge of the
    # box), so lat/lon are left empty (NaN): it is not drawn and not distance-matched.
    known_fires = pd.DataFrame(
        [
            ("Bagley", "2012-08-18", "2012-09-15", float("nan"), float("nan")),
            ("Bully", "2014-07-11", "2014-07-26", 40.431, -122.751),
            ("Carr", "2018-07-23", "2018-08-30", 40.654, -122.624),
            ("Hirz", "2018-08-09", "2018-09-12", 40.89, -122.25),
            ("Delta", "2018-09-05", "2018-10-07", 40.9425, -122.43),
            ("Zogg", "2020-09-27", "2020-10-13", 40.539, -122.567),
        ],
        columns=["fire", "start", "end", "lat", "lon"],
    )
    known_fires["start"] = pd.to_datetime(known_fires["start"])
    # "End" is the contained date, so include that whole day (end of the day).
    known_fires["end"] = pd.to_datetime(known_fires["end"]) + pd.Timedelta(hours=23, minutes=59)
    known_fires
    return (known_fires,)


@app.cell
def _(LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, load_sensor, mo, pd):
    # Data lives in <repo>/data (the loader searches sub-folders too).
    _data_dir = mo.notebook_dir().parent / "data"

    _parts = []
    for _name in ("modis", "viirs"):
        _df = load_sensor(_name, _data_dir)
        # Keep only what we need, and only inside the bounding box (all years),
        # which keeps the notebook fast.
        _df = _df[
            _df["latitude"].between(LAT_MIN, LAT_MAX)
            & _df["longitude"].between(LON_MIN, LON_MAX)
        ]
        if _name == "modis":
            # satellite is "Aqua" or "Terra" in the MODIS files.
            _df = _df.assign(sensor_label="MODIS " + _df["satellite"])
        else:
            _df = _df.assign(sensor_label="VIIRS")
        _parts.append(
            _df[["datetime", "year", "latitude", "longitude", "frp", "confidence",
                 "daynight", "sensor_label"]]
        )

    detections = pd.concat(_parts, ignore_index=True)
    mo.md(f"Loaded **{len(detections):,}** detections inside the bounding box (all years, all sensors).")
    return (detections,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Controls
    """)
    return


@app.cell
def _(mo):
    year = mo.ui.dropdown(options=[str(_y) for _y in range(2012, 2021)], value="2018", label="Year")
    sensors = mo.ui.multiselect(
        options=["MODIS Aqua", "MODIS Terra", "VIIRS"],
        value=["MODIS Aqua", "VIIRS"],
        label="Sensors",
    )
    # DBSCAN settings: how close (in km / days) detections must be to join a fire.
    space_km = mo.ui.slider(1, 15, step=0.5, value=3, label="Spatial scale (km)", show_value=True)
    time_days = mo.ui.slider(1, 15, step=1, value=5, label="Temporal scale (days)", show_value=True)
    min_samples = mo.ui.slider(1, 20, step=1, value=5, label="Min detections per cluster", show_value=True)
    mo.hstack([year, sensors], justify="start", gap=2)
    return min_samples, sensors, space_km, time_days, year


@app.cell
def _(mo, pd, year):
    # The date slider depends on the selected year, so it lives in its own cell.
    # It works in "day of year" (1 = Jan 1) and we show the real dates next to it.
    _y = int(year.value)
    n_days = 366 if pd.Timestamp(_y, 12, 31).day_of_year == 366 else 365
    day_range = mo.ui.range_slider(
        start=1, stop=n_days, step=1, value=[1, n_days], full_width=True,
        label="Day of year range",
    )
    day_range
    return day_range, n_days


@app.cell
def _(day_range, mo, pd, year):
    # Convert the slider's day-of-year numbers into real dates.
    year_start = pd.Timestamp(int(year.value), 1, 1)
    date_lo = year_start + pd.Timedelta(days=day_range.value[0] - 1)
    date_hi = year_start + pd.Timedelta(days=day_range.value[1] - 1, hours=23, minutes=59)
    mo.md(f"Selected dates: **{date_lo:%Y-%m-%d}** to **{date_hi:%Y-%m-%d}**")
    return date_hi, date_lo, year_start


@app.cell
def _(min_samples, mo, space_km, time_days):
    mo.hstack([space_km, time_days, min_samples], justify="start", gap=2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Filter detections
    """)
    return


@app.cell
def _(date_hi, date_lo, detections, mo, sensors, year, year_start):
    mo.stop(len(sensors.value) == 0, mo.md("**Pick at least one sensor.**"))

    sel = detections[
        (detections["year"] == int(year.value))
        & detections["sensor_label"].isin(sensors.value)
        & (detections["datetime"] >= date_lo)
        & (detections["datetime"] <= date_hi)
    ].copy()
    # "day" = days since Jan 1 (fractional), used for colors and clustering.
    sel["day"] = (sel["datetime"] - year_start).dt.total_seconds() / 86400.0
    sel = sel.reset_index(drop=True)

    mo.stop(sel.empty, mo.md("**No detections match these filters.** Try another year or wider dates."))

    _counts = (
        sel.groupby("sensor_label")
        .agg(detections=("frp", "size"), total_frp_MW=("frp", "sum"))
        .round(1)
        .reset_index()
        .rename(columns={"sensor_label": "sensor"})
    )
    mo.vstack([mo.md(f"**{len(sel):,}** detections after filtering. Counts per sensor:"),
               mo.ui.table(_counts, selection=None)])
    return (sel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Map colored by date
    Dark purple = early in the selected range, yellow = late. Large hollow circles are
    MODIS (dashed = Terra); small filled dots are VIIRS. Stars are known ignition points.
    Click any point for details.
    """)
    return


@app.cell
def _():
    # Map display limit: above this many points we draw a random sample.
    MAX_POINTS = 15000
    SEED = 42
    return MAX_POINTS, SEED


@app.cell
def _(LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, folium, known_fires, year):
    # Helper functions shared by both maps (defined once, used twice).
    def new_map():
        """Make an empty folium map zoomed to the bounding box."""
        _m = folium.Map(tiles="OpenStreetMap", control_scale=True)
        _m.fit_bounds([[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]])
        return _m

    def add_known_fires(fmap):
        """Add a labelled star for each known fire of the selected year."""
        _kf = known_fires[(known_fires["start"].dt.year == int(year.value)) & known_fires["lat"].notna()]
        for _, _f in _kf.iterrows():
            folium.Marker(
                [_f["lat"], _f["lon"]],
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
                tooltip=f"{_f['fire']} fire ignition ({_f['start']:%b %d} - {_f['end']:%b %d})",
            ).add_to(fmap)

    def popup_text(r):
        """HTML shown when a point is clicked."""
        return (
            f"<b>{r.sensor_label}</b><br>date: {r.datetime:%Y-%m-%d}<br>"
            f"time (UTC): {r.datetime:%H:%M}<br>FRP: {r.frp:.1f} MW<br>"
            f"confidence: {r.confidence}<br>day/night: {r.daynight}"
        )

    return add_known_fires, new_map, popup_text


@app.cell
def _(MAX_POINTS, SEED, sel):
    # Sample for plotting only (clustering below uses ALL selected points).
    if len(sel) > MAX_POINTS:
        plot_df = sel.sample(MAX_POINTS, random_state=SEED).sort_values("datetime")
        sample_note = (f"Showing a random sample of {MAX_POINTS:,} of {len(sel):,} points "
                       f"(seed={SEED}) to keep the map responsive.")
    else:
        plot_df = sel
        sample_note = f"Showing all {len(sel):,} points."
    return plot_df, sample_note


@app.cell
def _(add_known_fires, bcm, folium, mo, new_map, pd, plot_df, popup_text, sample_note, sel, year_start):
    _map = new_map()
    _lo, _hi = float(sel["day"].min()), float(sel["day"].max())
    if _hi <= _lo:  # only one day of data: avoid a zero-width color scale
        _hi = _lo + 1
    # Continuous viridis-like colormap over "days since Jan 1".
    _cmap = bcm.linear.viridis.scale(_lo, _hi)
    _d0 = year_start + pd.Timedelta(days=_lo)
    _d1 = year_start + pd.Timedelta(days=_hi)
    _cmap.caption = f"Acquisition date: {_d0:%b %d} (dark) -> {_d1:%b %d} (yellow); scale = days since Jan 1"
    _cmap.add_to(_map)

    for _r in plot_df.itertuples():
        _color = _cmap(_r.day)
        if _r.sensor_label.startswith("MODIS"):
            # MODIS: larger hollow circle (dashed outline for Terra).
            _dash = "4 4" if _r.sensor_label.endswith("Terra") else None
            folium.CircleMarker([_r.latitude, _r.longitude], radius=7, color=_color,
                                weight=2, fill=False, dash_array=_dash,
                                popup=folium.Popup(popup_text(_r), max_width=220)).add_to(_map)
        else:
            # VIIRS: small filled dot.
            folium.CircleMarker([_r.latitude, _r.longitude], radius=2, color=_color,
                                weight=0, fill=True, fill_color=_color, fill_opacity=0.9,
                                popup=folium.Popup(popup_text(_r), max_width=220)).add_to(_map)
    add_known_fires(_map)

    mo.vstack([mo.md(f"*{sample_note}*"), mo.Html(_map._repr_html_())])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Automatic fire grouping (DBSCAN)
    Each detection becomes a point `[north-km, east-km, days]`; we divide each axis by
    its scale (the sliders above) so that "1 unit" means "about one spatial scale
    or one temporal scale apart". DBSCAN then joins points that are within 1 unit of
    each other, chaining them into clusters. Points that join nothing are **noise (-1)**.
    Clusters are numbered by first detection date.
    """)
    return


@app.cell
def _(DBSCAN, min_samples, np, sel, space_km, time_days):
    # Convert degrees to km (flat-earth approximation is fine for ~80 km).
    _lat0 = np.radians(sel["latitude"].mean())
    _north_km = sel["latitude"].to_numpy() * 110.574
    _east_km = sel["longitude"].to_numpy() * 111.320 * np.cos(_lat0)

    # Scale: divide by the chosen km / days so eps = 1 means "one scale unit".
    _X = np.column_stack([
        _north_km / space_km.value,
        _east_km / space_km.value,
        sel["day"].to_numpy() / time_days.value,
    ])
    _raw = DBSCAN(eps=1.0, min_samples=min_samples.value).fit_predict(_X)

    # Renumber real clusters 0,1,2... in order of first detection; keep noise = -1.
    _first = {c: sel["day"].to_numpy()[_raw == c].min() for c in set(_raw) if c != -1}
    _order = {c: i for i, c in enumerate(sorted(_first, key=_first.get))}
    clustered = sel.assign(cluster=[(_order[c] if c != -1 else -1) for c in _raw])
    return (clustered,)


@app.cell
def _(np):
    def haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance in km between points (works on numpy arrays too)."""
        _p1, _p2 = np.radians(lat1), np.radians(lat2)
        _dphi = _p2 - _p1
        _dl = np.radians(lon2) - np.radians(lon1)
        _a = np.sin(_dphi / 2) ** 2 + np.cos(_p1) * np.cos(_p2) * np.sin(_dl / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(_a))

    return (haversine_km,)


@app.cell
def _(clustered, haversine_km, known_fires, mo, np, pd):
    MATCH_KM = 25  # centroid must be within this distance of the ignition point

    _rows = []
    for _cid, _g in clustered[clustered["cluster"] != -1].groupby("cluster"):
        _clat, _clon = _g["latitude"].mean(), _g["longitude"].mean()
        _first, _last = _g["datetime"].min(), _g["datetime"].max()

        # Best match = known fire whose dates overlap the cluster's dates AND whose
        # ignition point is within MATCH_KM of the cluster centroid (closest wins).
        _best, _best_d = "unknown", np.inf
        for _, _f in known_fires.iterrows():
            if pd.isna(_f["lat"]):
                continue  # no usable ignition point
            _overlap = (_first <= _f["end"]) and (_last >= _f["start"])
            _d = haversine_km(_clat, _clon, _f["lat"], _f["lon"])
            if _overlap and _d <= MATCH_KM and _d < _best_d:
                _best, _best_d = _f["fire"], _d

        _modis = _g[_g["sensor_label"].str.startswith("MODIS")]
        _viirs = _g[_g["sensor_label"] == "VIIRS"]
        _rows.append({
            "cluster": _cid,
            "first_date": _first.date(),
            "last_date": _last.date(),
            "n_modis": len(_modis),
            "n_viirs": len(_viirs),
            "frp_modis_MW": round(_modis["frp"].sum(), 1),
            "frp_viirs_MW": round(_viirs["frp"].sum(), 1),
            "centroid_lat": round(_clat, 4),
            "centroid_lon": round(_clon, 4),
            "matched_fire": _best,
            "dist_km": round(_best_d, 1) if _best != "unknown" else None,
        })

    summary = pd.DataFrame(_rows, columns=[
        "cluster", "first_date", "last_date", "n_modis", "n_viirs", "frp_modis_MW",
        "frp_viirs_MW", "centroid_lat", "centroid_lon", "matched_fire", "dist_km"])
    n_noise = int((clustered["cluster"] == -1).sum())

    mo.vstack([
        mo.md(f"## 5. Cluster summary\n**{len(summary)}** clusters; **{n_noise:,}** noise points (label -1) "
              f"out of {len(clustered):,}. A cluster is matched to a known fire if the dates overlap "
              f"and its centroid is within {MATCH_KM} km of the ignition point."),
        mo.ui.table(summary, selection=None, page_size=25),
    ])
    return n_noise, summary


@app.cell
def _(matplotlib, summary):
    # One color per cluster, shared by the cluster map and the timeline.
    _pal = list(matplotlib.colormaps["tab20"].colors) + list(matplotlib.colormaps["tab20b"].colors)
    cluster_colors = {-1: "#999999"}
    for _i, _cid in enumerate(summary["cluster"]):
        cluster_colors[int(_cid)] = matplotlib.colors.to_hex(_pal[_i % len(_pal)])

    # Human-readable label, e.g. "C3 - Carr" or "C7 - unknown".
    cluster_labels = {-1: "noise (-1)"}
    for _r in summary.itertuples():
        cluster_labels[int(_r.cluster)] = f"C{_r.cluster} - {_r.matched_fire}"
    return cluster_colors, cluster_labels


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Map colored by cluster
    Gray = noise. Labels show the cluster number and the matched known fire.
    Compare with the date map above.
    """)
    return


@app.cell
def _(MAX_POINTS, SEED, add_known_fires, cluster_colors, cluster_labels, clustered, folium, mo, new_map, popup_text, summary):
    _map = new_map()
    _pts = clustered
    if len(_pts) > MAX_POINTS:
        _pts = _pts.sample(MAX_POINTS, random_state=SEED)  # same seed as the date map

    for _r in _pts.itertuples():
        _color = cluster_colors[_r.cluster]
        _txt = popup_text(_r) + f"<br><b>{cluster_labels[_r.cluster]}</b>"
        if _r.sensor_label.startswith("MODIS"):
            _dash = "4 4" if _r.sensor_label.endswith("Terra") else None
            folium.CircleMarker([_r.latitude, _r.longitude], radius=7, color=_color, weight=2,
                                fill=False, dash_array=_dash,
                                popup=folium.Popup(_txt, max_width=220)).add_to(_map)
        else:
            folium.CircleMarker([_r.latitude, _r.longitude], radius=2, color=_color, weight=0,
                                fill=True, fill_color=_color, fill_opacity=0.9,
                                popup=folium.Popup(_txt, max_width=220)).add_to(_map)

    # Text label at each cluster's centroid.
    for _r in summary.itertuples():
        folium.Marker(
            [_r.centroid_lat, _r.centroid_lon],
            icon=folium.DivIcon(html=(
                f'<div style="font-size:11px;font-weight:bold;white-space:nowrap;color:#000;'
                f'text-shadow:0 0 3px #fff,0 0 3px #fff;">{cluster_labels[int(_r.cluster)]}</div>')),
        ).add_to(_map)
    add_known_fires(_map)

    # Simple HTML legend.
    _items = "".join(
        f'<div><span style="background:{cluster_colors[int(_c)]};display:inline-block;width:10px;'
        f'height:10px;margin-right:4px;"></span>{cluster_labels[int(_c)]}</div>'
        for _c in list(summary["cluster"]) + [-1])
    _map.get_root().html.add_child(folium.Element(
        '<div style="position:fixed;bottom:20px;left:20px;z-index:9999;background:white;'
        'padding:6px 8px;border:1px solid #888;font-size:11px;max-height:300px;overflow:auto;">'
        f"{_items}</div>"))

    mo.Html(_map._repr_html_())
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Timeline: daily detections per cluster
    Shaded bands are the known fires' start-to-containment dates.
    """)
    return


@app.cell
def _(cluster_colors, cluster_labels, clustered, known_fires, plt, year):
    _fig, _ax = plt.subplots(figsize=(12, 4.5))

    # Daily counts per cluster (rows = day, columns = cluster).
    _daily = (clustered.assign(date=clustered["datetime"].dt.normalize())
              .groupby(["date", "cluster"]).size().unstack(fill_value=0))
    _bottom = None
    for _c in _daily.columns:
        _vals = _daily[_c]
        _ax.bar(_daily.index, _vals, bottom=_bottom, width=1.0, color=cluster_colors[int(_c)],
                label=cluster_labels[int(_c)])
        _bottom = _vals if _bottom is None else _bottom + _vals

    # Shade each known fire of this year.
    _kf = known_fires[known_fires["start"].dt.year == int(year.value)]
    for _, _f in _kf.iterrows():
        _ax.axvspan(_f["start"], _f["end"], color="red", alpha=0.10)
        _ax.text(_f["start"], _ax.get_ylim()[1] * 0.97, f" {_f['fire']}", color="darkred",
                 va="top", fontsize=9, fontweight="bold")

    _ax.set_xlabel("Date")
    _ax.set_ylabel("Detections per day")
    _ax.set_title(f"Daily detections by cluster, {year.value} (red bands = known fires)")
    _ax.legend(fontsize=7, ncol=2, loc="upper right", bbox_to_anchor=(1.0, 0.85))
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(known_fires, mo, n_noise, summary, year):
    # Build the summary text from the computed results (nothing hard-coded).
    _lines = [f"## Summary for {year.value}", ""]
    _matched = summary[summary["matched_fire"] != "unknown"]
    _unmatched = summary[summary["matched_fire"] == "unknown"]

    for _name, _g in _matched.groupby("matched_fire"):
        _desc = "; ".join(
            f"cluster {r.cluster} ({r.first_date} to {r.last_date}, {r.n_modis} MODIS + {r.n_viirs} VIIRS, "
            f"{r.dist_km} km from ignition)" for r in _g.itertuples())
        _lines.append(f"- **{_name}** <- {_desc}")

    _expected = known_fires[known_fires["start"].dt.year == int(year.value)]["fire"].tolist()
    _missed = [f for f in _expected if f not in set(_matched["matched_fire"])]
    if _missed:
        _lines.append(f"- Known fires with **no** matching cluster: {', '.join(_missed)}")
    if len(_unmatched):
        _lines.append("")
        _lines.append("**Clusters that did not match any known fire:**")
        for r in _unmatched.itertuples():
            _lines.append(f"- cluster {r.cluster}: {r.first_date} to {r.last_date}, "
                          f"{r.n_modis + r.n_viirs} detections, centroid ({r.centroid_lat}, {r.centroid_lon})")
    elif len(summary):
        _lines.append("")
        _lines.append("Every cluster matched a known fire.")
    _lines.append("")
    _lines.append(f"{n_noise:,} detections were noise (not part of any cluster).")
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
