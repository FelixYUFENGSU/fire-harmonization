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
    # Which real fire does each satellite detection belong to?

    **Why this notebook exists.** Our project *harmonizes* wildfire measurements from two
    satellite families: **MODIS** (Aqua/Terra, ~1 km pixels) and **VIIRS** (~375 m pixels).
    To compare them fairly we need to know that a MODIS detection and a VIIRS detection
    are looking at the **same real fire**. Here we sort detections near Redding, CA into
    fires, then check the groups against fires we already know about
    (Bagley, Bully, Carr, Hirz, Delta, Zogg).

    **Words used below**
    - **Detection**: one satellite pixel flagged as "burning" at one moment.
    - **FRP** (Fire Radiative Power): how much heat energy the fire gives off, in megawatts (MW). Bigger = more intense.
    - **Bounding box**: a rectangle on the map defined by min/max latitude and longitude.
    - **Colormap**: a rule that turns a number (here, a date) into a color.
    - **DBSCAN**: a clustering method that groups points that sit close together and calls isolated points "noise".
    - **Epsilon**: DBSCAN's "how close is close" distance.
    - **Cluster**: one group DBSCAN found; we hope each one is one real fire.

    *Tip: every code cell has a short explanation above it and a "what this shows" note below it.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. Setup

    ### Step: the region of interest (bounding box)
    We only care about detections northwest of Redding (Whiskeytown, Shasta Lake, Lakehead,
    Shasta-Trinity National Forest). The cell below sets the **bounding box**; edit the four
    numbers and everything downstream re-runs. A small box keeps the maps fast and focused.
    """)
    return


@app.cell
def _():
    # ------------------------------------------------------------------
    # EDIT ME: region of interest (degrees).
    # ------------------------------------------------------------------
    LAT_MIN, LAT_MAX = 40.40, 41.05
    LON_MIN, LON_MAX = -122.90, -122.10
    return LAT_MAX, LAT_MIN, LON_MAX, LON_MIN


@app.cell
def _(LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, mo, np):
    # Rough size of the box in km (1 degree of latitude is about 111 km).
    _h_km = (LAT_MAX - LAT_MIN) * 111.0
    _w_km = (LON_MAX - LON_MIN) * 111.0 * np.cos(np.radians((LAT_MAX + LAT_MIN) / 2))
    mo.md(f"""
    **What this means:** the box runs from latitude {LAT_MIN} to {LAT_MAX} and longitude
    {LON_MIN} to {LON_MAX}. That is roughly **{_w_km:.0f} km wide by {_h_km:.0f} km tall**
    (about {_w_km * _h_km:,.0f} km²). Any fire whose ignition lies outside this rectangle
    (for example Bagley, which started near the east edge) may be only partly visible.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step: the list of known fires
    To judge whether our automatic groups are real fires we need a "answer key": the fires
    that officially burned here, with start date, containment date and ignition point.
    It is a small table so you can add rows later. Bagley's ignition point is uncertain
    (likely outside the box), so its lat/lon are left empty and it is matched by dates only
    in the timeline, not by distance.
    """)
    return


@app.cell
def _(mo, pd):
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
    # "End" is the containment date, so include that whole day.
    known_fires["end"] = pd.to_datetime(known_fires["end"]) + pd.Timedelta(hours=23, minutes=59)

    # Show a small, plain-date version (6 rows).
    mo.ui.table(
        known_fires.assign(start=known_fires["start"].dt.date, end=known_fires["end"].dt.date),
        selection=None, page_size=10,
    )
    return (known_fires,)


@app.cell
def _(known_fires, mo):
    _by_year = known_fires.groupby(known_fires["start"].dt.year)["fire"].apply(", ".join)
    _txt = "; ".join(f"{y}: {names}" for y, names in _by_year.items())
    mo.md(f"""
    **How to read it:** each row is one fire. `start`/`end` bound when it burned and
    `lat`/`lon` is where it started. Fires by year: {_txt}. Later we call a cluster a
    "match" only if its dates overlap a row's dates **and** its center is near that row's
    ignition point. Fires in years we did not select are simply ignored in that year's maps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step: load the satellite data
    The CSVs hold *every* detection in California for 2012–2020, which is far more than we
    need. The loader (`load_sensor`) reads the MODIS and VIIRS files, and we immediately
    keep only rows inside the bounding box. We also label each row with a **sensor label**
    ("MODIS Aqua", "MODIS Terra" or "VIIRS") and keep just the columns we use, which keeps
    memory use and output sizes small.
    """)
    return


@app.cell
def _(LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, load_sensor, mo, pd):
    # Data lives in <repo>/data (the loader searches sub-folders too).
    _data_dir = mo.notebook_dir().parent / "data"

    _parts = []
    for _name in ("modis", "viirs"):
        _df = load_sensor(_name, _data_dir)
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
    return (detections,)


@app.cell
def _(detections, mo):
    _per_label = detections["sensor_label"].value_counts()
    _per_year = detections["year"].value_counts().sort_index()
    _lbl = ", ".join(f"{k}: {v:,}" for k, v in _per_label.items())
    _yrs = ", ".join(f"{y}: {n:,}" for y, n in _per_year.items())
    mo.md(f"""
    **What we loaded:** {len(detections):,} detections inside the box across all years
    ({_lbl}). Per year: {_yrs}. Big years (like 2018, the Carr/Hirz/Delta season) stand out
    immediately. Nothing is plotted yet; this is the pool we filter in the next steps.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Controls
    These widgets let you ask "what if?" without touching code. Whenever you change one,
    every cell that depends on it re-runs automatically.
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
    # Map size limit: at most this many points PER SENSOR are drawn on each map.
    max_points = mo.ui.slider(1000, 10000, step=500, value=5000, label="Max points per sensor on maps", show_value=True)
    mo.vstack([
        mo.hstack([year, sensors], justify="start", gap=2),
        mo.hstack([space_km, time_days, min_samples], justify="start", gap=2),
        max_points,
    ])
    return max_points, min_samples, sensors, space_km, time_days, year


@app.cell
def _(max_points, min_samples, mo, sensors, space_km, time_days, year):
    mo.md(f"""
    **What each control does** (current values in bold):
    - **Year** (**{year.value}**): which fire season to study.
    - **Sensors** (**{', '.join(sensors.value) or 'none'}**): which instruments to include.
      Aqua and Terra are two MODIS satellites; VIIRS is the newer instrument. Comparing
      them is the point of the project.
    - **Spatial scale** (**{space_km.value} km**) and **Temporal scale** (**{time_days.value} days**):
      how far apart in space/time two detections can be and still count as "the same fire".
      Larger values merge fires together; smaller values split them apart.
    - **Min detections per cluster** (**{min_samples.value}**): a group smaller than this is
      treated as noise.
    - **Max points per sensor on maps** (**{max_points.value:,}**): a drawing limit only. It never
      changes the clustering; it just keeps the maps light enough to display. (Each map is
      also limited to 15,000 points in total, so with three sensors each gets at most 5,000.)
    """)
    return


@app.cell
def _(mo, pd, year):
    # The date slider depends on the selected year, so it lives in its own cell.
    # It works in "day of year" (1 = Jan 1); the next cell turns it into real dates.
    _y = int(year.value)
    _n_days = 366 if pd.Timestamp(_y, 12, 31).day_of_year == 366 else 365
    day_range = mo.ui.range_slider(
        start=1, stop=_n_days, step=1, value=[1, _n_days], full_width=True,
        label="Day of year range",
    )
    day_range
    return (day_range,)


@app.cell
def _(day_range, pd, year):
    # Convert the slider's day-of-year numbers into real dates.
    year_start = pd.Timestamp(int(year.value), 1, 1)
    date_lo = year_start + pd.Timedelta(days=day_range.value[0] - 1)
    date_hi = year_start + pd.Timedelta(days=day_range.value[1] - 1, hours=23, minutes=59)
    return date_hi, date_lo, year_start


@app.cell
def _(date_hi, date_lo, mo):
    mo.md(f"""
    **Date range:** the slider above counts days from Jan 1 (day 1). Your selection is
    **{date_lo:%b %d, %Y} to {date_hi:%b %d, %Y}** ({(date_hi - date_lo).days + 1} days).
    Drag the handles to zoom in on one fire's window, e.g. late July to August for Carr.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Filter detections
    Now we apply the controls: keep only detections from the chosen **year**, **sensors**
    and **date range** (the bounding box was applied at load time). We also add a `day`
    column (days since Jan 1) because both the map colors and the clustering need a plain
    number for "when". The table counts detections and sums FRP per sensor.
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

    counts = (
        sel.groupby("sensor_label")
        .agg(detections=("frp", "size"), total_frp_MW=("frp", "sum"))
        .round(1)
        .reset_index()
        .rename(columns={"sensor_label": "sensor"})
    )
    mo.ui.table(counts, selection=None, page_size=10)
    return counts, sel


@app.cell
def _(counts, mo, sel, year):
    _n_modis = int(counts.loc[counts["sensor"].str.startswith("MODIS"), "detections"].sum())
    _n_viirs = int(counts.loc[counts["sensor"] == "VIIRS", "detections"].sum())
    _txt = f"We kept **{len(sel):,}** detections in {year.value}: **{_n_modis:,} MODIS** and **{_n_viirs:,} VIIRS**. "
    if _n_modis and _n_viirs:
        _r = _n_viirs / _n_modis
        _txt += (
            f"VIIRS has about **{_r:.1f}×** as many detections as MODIS. That is expected: VIIRS pixels "
            f"are ~375 m across versus ~1 km for MODIS, so it can see smaller flames and splits one fire "
            f"into many more pixels. "
        )
        _frp_m = counts.loc[counts["sensor"].str.startswith("MODIS"), "total_frp_MW"].sum()
        _frp_v = counts.loc[counts["sensor"] == "VIIRS", "total_frp_MW"].sum()
        _txt += (f"Total FRP is {_frp_m:,.0f} MW (MODIS) vs {_frp_v:,.0f} MW (VIIRS); "
                 f"comparing those totals fairly is exactly what harmonization is for.")
    else:
        _txt += "Select both a MODIS and the VIIRS sensor to compare them."
    mo.md(_txt)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Map colored by date
    ### Step: prepare helper functions and a size-limited sample
    Drawing every detection as its own map object makes the page huge (the notebook would
    refuse to show it). So the maps use a smaller recipe: each sensor's points go into
    **one GeoJSON layer** (a compact list of coordinates), coordinates are rounded to
    4 decimals (~10 m), and each point keeps only the few fields the popup needs. If a
    sensor has more points than the slider's cap, we draw a **random sample** with a fixed
    seed (so the same points appear every time). The sample is for *drawing only*;
    clustering below uses every point.
    """)
    return


@app.cell
def _(LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, folium, known_fires, pd, year):
    SEED = 42  # fixed random seed so samples are reproducible
    MAX_TOTAL = 15000  # hard limit of points per map, so a map never exceeds marimo's output limit

    def sample_per_sensor(df, cap):
        """Return (sampled rows, {sensor: (shown, total)}), at most `cap` rows per sensor.

        The cap is lowered further if needed so that all sensors together stay <= MAX_TOTAL.
        """
        cap = min(cap, MAX_TOTAL // max(df["sensor_label"].nunique(), 1))
        parts, info = [], {}
        for name, g in df.groupby("sensor_label"):
            info[name] = (min(len(g), cap), len(g))
            parts.append(g.sample(cap, random_state=SEED) if len(g) > cap else g)
        return pd.concat(parts), info

    def sample_note(info):
        """One-line text such as 'MODIS Aqua: 800 of 800; VIIRS: 5,000 of 12,000 (sampled)'."""
        bits = []
        for name, (shown, total) in info.items():
            bits.append(f"{name}: {shown:,} of {total:,}" + (" (random sample)" if shown < total else ""))
        return "; ".join(bits)

    def new_map():
        """Make an empty folium map zoomed to the bounding box."""
        m = folium.Map(tiles="OpenStreetMap", control_scale=True)
        m.fit_bounds([[LAT_MIN, LON_MIN], [LAT_MAX, LON_MAX]])
        return m

    def add_known_fires(fmap):
        """Add a labelled star for each known fire of the selected year."""
        kf = known_fires[(known_fires["start"].dt.year == int(year.value)) & known_fires["lat"].notna()]
        for _, f in kf.iterrows():
            folium.Marker(
                [f["lat"], f["lon"]],
                icon=folium.Icon(color="red", icon="star", prefix="fa"),
                tooltip=f"{f['fire']} fire ignition ({f['start']:%b %d} - {f['end']:%b %d})",
            ).add_to(fmap)

    # Each point gets this little script, which colors it from its own "c" property.
    # (Using a script avoids storing a separate style entry for every point.)
    COLOR_JS = folium.JsCode(
        "function(f, layer){layer.setStyle({color: f.properties.c, fillColor: f.properties.c});}"
    )

    def add_points(fmap, df, with_group=False):
        """Draw df as ONE compact GeoJSON layer per sensor.

        df needs: latitude, longitude, datetime, sensor_label, frp, confidence
        and `color` (a hex string); `group` too if with_group=True.
        MODIS = big hollow circle (dashed for Terra); VIIRS = small filled dot.
        """
        fields = ["s", "d", "f", "q"]
        aliases = ["Sensor", "Date MM-DD HH:MM (UTC)", "FRP (MW)", "Confidence"]
        if with_group:
            fields.append("g")
            aliases.append("Cluster")
        for name, g in df.groupby("sensor_label"):
            feats = []
            for r in g.itertuples():
                props = {
                    "s": name,
                    "d": f"{r.datetime:%m-%d %H:%M}",
                    "f": round(float(r.frp), 1),
                    "q": str(r.confidence),
                    "c": r.color,
                }
                if with_group:
                    props["g"] = r.group
                feats.append({
                    "type": "Feature",
                    "geometry": {"type": "Point",
                                 "coordinates": [round(float(r.longitude), 4), round(float(r.latitude), 4)]},
                    "properties": props,
                })
            if name.startswith("MODIS"):
                marker = folium.CircleMarker(
                    radius=7, weight=2, fill=False,
                    dash_array="4 4" if name.endswith("Terra") else None)
            else:
                marker = folium.CircleMarker(radius=2, weight=0, fill=True, fill_opacity=0.9)
            folium.GeoJson(
                {"type": "FeatureCollection", "features": feats},
                name=name, marker=marker, on_each_feature=COLOR_JS,
                popup=folium.GeoJsonPopup(fields=fields, aliases=aliases, max_width=220),
            ).add_to(fmap)

    return add_known_fires, add_points, new_map, sample_note, sample_per_sensor


@app.cell
def _(max_points, sample_per_sensor, sel):
    # Pick the points we will DRAW (clustering later still uses all of `sel`).
    plot_df, plot_info = sample_per_sensor(sel, max_points.value)
    return plot_df, plot_info


@app.cell
def _(add_known_fires, add_points, bcm, mo, new_map, pd, plot_df, sel, year_start):
    _map = new_map()
    _lo, _hi = float(sel["day"].min()), float(sel["day"].max())
    if _hi <= _lo:  # only one day of data: avoid a zero-width color scale
        _hi = _lo + 1
    # Continuous viridis colormap over "days since Jan 1".
    _cmap = bcm.linear.viridis.scale(_lo, _hi)
    _d0 = year_start + pd.Timedelta(days=_lo)
    _d1 = year_start + pd.Timedelta(days=_hi)
    _cmap.caption = f"Acquisition date: {_d0:%b %d} (dark purple) -> {_d1:%b %d} (yellow)"
    _cmap.add_to(_map)

    # Give every drawn point its color (first 7 characters = "#rrggbb").
    _colored = plot_df.assign(color=[_cmap(_d)[:7] for _d in plot_df["day"]])
    add_points(_map, _colored)
    add_known_fires(_map)
    mo.Html(_map._repr_html_())
    return


@app.cell
def _(mo, pd, plot_info, sample_note, sel, year_start):
    _d0 = year_start + pd.Timedelta(days=float(sel["day"].min()))
    _d1 = year_start + pd.Timedelta(days=float(sel["day"].max()))
    mo.md(f"""
    **How to read the map.** Color = date: **dark purple is early, yellow is late**
    ({_d0:%b %d} to {_d1:%b %d}). Big hollow circles are MODIS (dashed = Terra); small
    filled dots are VIIRS; red stars are the known ignition points. Click a point to see
    its sensor, time, FRP and confidence.

    **What to look for:** a patch of one color is a fire burning during one period. A
    smooth purple-to-yellow gradient across a region means the fire spread over time.
    Separate blobs with different colors are probably different fires.

    *Drawn points: {sample_note(plot_info)}. Sampling only affects this picture, not the
    clustering. Use the "Max points per sensor" slider to draw more or fewer.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. Automatic fire grouping (DBSCAN)
    ### Step: cluster detections in space *and* time
    Looking at colors is subjective, so we let **DBSCAN** do the grouping. Each detection
    becomes a point `[north-km, east-km, days since Jan 1]`. We divide each axis by its
    scale from the sliders, so "1 unit" means "one spatial scale away" or "one temporal
    scale away". DBSCAN with **epsilon = 1** then links points within 1 unit of each other
    and grows groups by chaining. Points with too few neighbours become **noise (-1)**.
    Clusters are then numbered 0, 1, 2… by first detection date.
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
def _(clustered, mo, space_km, time_days):
    _sizes = clustered.loc[clustered["cluster"] != -1, "cluster"].value_counts()
    _noise = int((clustered["cluster"] == -1).sum())
    _txt = (f"DBSCAN found **{len(_sizes)}** clusters and labelled **{_noise:,}** of "
            f"{len(clustered):,} detections as noise (-1), using {space_km.value} km and "
            f"{time_days.value} days. ")
    if len(_sizes):
        _txt += (f"The biggest cluster has {int(_sizes.iloc[0]):,} detections "
                 f"(cluster {int(_sizes.index[0])}); the smallest has {int(_sizes.iloc[-1]):,}. ")
    _txt += ("Big clusters are usually real wildfires; tiny ones may be small or prescribed burns. "
             "If two real fires are merged into one cluster, shrink the scales; if one fire is "
             "split into pieces, enlarge them.")
    mo.md(_txt)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step: a distance helper
    To decide whether a cluster is a known fire we need the distance between the cluster's
    center and the fire's ignition point. The **haversine formula** gives the
    great-circle distance (the shortest path over the Earth's surface) in km. This cell
    just defines the function, so it has no visible output.
    """)
    return


@app.cell
def _(np):
    def haversine_km(lat1, lon1, lat2, lon2):
        """Great-circle distance in km between points (works on numpy arrays too)."""
        p1, p2 = np.radians(lat1), np.radians(lat2)
        dphi = p2 - p1
        dl = np.radians(lon2) - np.radians(lon1)
        a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
        return 2 * 6371.0088 * np.arcsin(np.sqrt(a))

    return (haversine_km,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Cluster summary
    ### Step: describe each cluster and match it to a known fire
    For every cluster we compute its first/last detection date, how many MODIS and VIIRS
    detections it has, total FRP per sensor, and its **centroid** (average position).
    The **best-matching known fire** must satisfy two rules: the dates overlap, and the
    centroid is within 25 km of the ignition point; if several qualify, the closest wins.
    Otherwise the cluster is called "unknown". The table is paged (10 rows at a time).
    """)
    return


@app.cell
def _(clustered, haversine_km, known_fires, mo, np, pd):
    MATCH_KM = 25  # centroid must be within this distance of the ignition point

    _rows = []
    for _cid, _g in clustered[clustered["cluster"] != -1].groupby("cluster"):
        _clat, _clon = _g["latitude"].mean(), _g["longitude"].mean()
        _first, _last = _g["datetime"].min(), _g["datetime"].max()

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

    mo.ui.table(summary, selection=None, page_size=10)
    return MATCH_KM, n_noise, summary


@app.cell
def _(MATCH_KM, mo, n_noise, summary):
    _m = summary[summary["matched_fire"] != "unknown"]
    _txt = (f"There are **{len(summary)}** clusters; **{len(_m)}** matched a known fire "
            f"({_m['matched_fire'].nunique()} distinct fires) and **{len(summary) - len(_m)}** are "
            f"\"unknown\". {n_noise:,} detections were noise. ")
    if len(summary):
        _big = summary.assign(n=summary["n_modis"] + summary["n_viirs"]).sort_values("n", ascending=False).iloc[0]
        _txt += (f"The largest cluster is #{int(_big['cluster'])} ({_big['first_date']} to {_big['last_date']}, "
                 f"{int(_big['n']):,} detections), matched to **{_big['matched_fire']}**. ")
    _txt += (f"`dist_km` is how far the cluster center is from the ignition point (must be ≤ {MATCH_KM} km). "
             "A cluster can be \"unknown\" because the fire isn't in our list, or because two fires "
             "were merged and the center landed too far from both ignition points.")
    mo.md(_txt)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Step: choose a color for each cluster
    The second map and the timeline both need one distinct color per cluster (gray for
    noise) and a readable label such as "C1 - Carr". We build these lookups once here so the
    two figures agree with each other. No visible output.
    """)
    return


@app.cell
def _(matplotlib, summary):
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
def _(cluster_colors, mo):
    _n = len(cluster_colors) - 1
    mo.md(f"""
    We prepared **{_n}** cluster colors plus gray for noise. With more than 40 clusters the
    palette repeats, so neighbouring clusters can share a color; click a point to see its
    exact cluster label.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Map colored by cluster
    ### Step: draw the same detections, now colored by DBSCAN cluster
    This map uses the same points (same seed and cap) as the date map, but each point takes
    its **cluster's color**, with a text label at each cluster's center showing the matched
    fire. Flip between the two maps to see whether the groups DBSCAN chose line up with the
    color patches you saw by date.
    """)
    return


@app.cell
def _(add_known_fires, add_points, cluster_colors, cluster_labels, clustered, folium, max_points, mo, new_map, sample_per_sensor, summary):
    _map = new_map()
    _pts, cluster_info = sample_per_sensor(clustered, max_points.value)  # same seed as the date map
    _pts = _pts.assign(
        color=_pts["cluster"].map(cluster_colors),
        group=_pts["cluster"].map(cluster_labels),
    )
    add_points(_map, _pts, with_group=True)

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
    return (cluster_info,)


@app.cell
def _(cluster_info, mo, n_noise, sample_note, summary):
    _unk = int((summary["matched_fire"] == "unknown").sum())
    mo.md(f"""
    **How to read the map.** Each color is one cluster, with its label ("C#" = cluster
    number, then the matched fire or "unknown"). **Gray points are noise**: isolated
    detections that DBSCAN did not attach to any fire ({n_noise:,} in total). Hollow
    circles are MODIS and dots are VIIRS, as before.

    **What to look for:** each real fire should be one compact color blob containing both
    MODIS circles and VIIRS dots, near a red star. Here, **{len(summary) - _unk}** clusters have a
    matched fire name and **{_unk}** say "unknown". Two stars inside one blob means DBSCAN
    merged two fires; one star spread over several colors means it split one fire.

    *Drawn points: {sample_note(cluster_info)}.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Timeline: detections per day, per cluster
    ### Step: see how each fire evolves through the year
    A stacked bar chart: each bar is one day, and each color slice is the number of
    detections from one cluster that day. Red shaded bands mark the known fires' official
    start-to-containment dates. If our clusters are real fires, a cluster's bars should
    rise and fall inside the matching band.
    """)
    return


@app.cell
def _(cluster_colors, cluster_labels, clustered, known_fires, plt, year):
    _fig, _ax = plt.subplots(figsize=(10, 4))

    # Daily counts per cluster (rows = day, columns = cluster).
    daily_counts = (clustered.assign(date=clustered["datetime"].dt.normalize())
                    .groupby(["date", "cluster"]).size().unstack(fill_value=0))
    _bottom = None
    for _c in daily_counts.columns:
        _vals = daily_counts[_c]
        _ax.bar(daily_counts.index, _vals, bottom=_bottom, width=1.0, color=cluster_colors[int(_c)],
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
    plt.close(_fig)  # free the figure's memory; the finished picture is still shown below
    _fig
    return (daily_counts,)


@app.cell
def _(cluster_labels, daily_counts, known_fires, mo, year):
    _tot = daily_counts.sum(axis=1)
    _peak_day = _tot.idxmax()
    _top_c = int(daily_counts.loc[_peak_day].idxmax())
    _kf = known_fires[known_fires["start"].dt.year == int(year.value)]
    _bands = ", ".join(f"{r.fire} ({r.start:%b %d}–{r.end:%b %d})" for r in _kf.itertuples()) or "none this year"
    mo.md(f"""
    **How to read the chart.** The busiest day was **{_peak_day:%b %d}** with
    **{int(_tot.max()):,}** detections, mostly from **{cluster_labels[_top_c]}**. Red bands
    for {year.value}: {_bands}.

    **What to look for:** a cluster whose bars sit inside its band is behaving like that
    fire. Bars that start before the band are early detections (or a different fire);
    bars after it are smoldering or a neighbouring fire. Gray bars are noise. Gaps are days
    with no detections, often because of smoke/cloud or no satellite overpass.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 8. Summary
    The final cell is written from the results above (nothing is hard-coded), so it updates
    whenever you change the year, sensors or sliders.
    """)
    return


@app.cell
def _(MATCH_KM, known_fires, mo, n_noise, summary, year):
    _lines = [f"### Summary for {year.value}", ""]
    _matched = summary[summary["matched_fire"] != "unknown"]
    _unmatched = summary[summary["matched_fire"] == "unknown"]

    _lines.append(f"DBSCAN found **{len(summary)}** clusters; **{len(_matched)}** matched a known fire "
                  f"(dates overlap and centroid within {MATCH_KM} km of ignition).")
    _lines.append("")
    for _name, _g in _matched.groupby("matched_fire"):
        _desc = "; ".join(
            f"cluster {r.cluster} ({r.first_date} to {r.last_date}, {r.n_modis:,} MODIS + {r.n_viirs:,} VIIRS, "
            f"{r.dist_km} km from ignition)" for r in _g.itertuples())
        _lines.append(f"- **{_name}** ← {_desc}")

    _expected = known_fires[known_fires["start"].dt.year == int(year.value)]["fire"].tolist()
    _missed = [f for f in _expected if f not in set(_matched["matched_fire"])]
    if _missed:
        _lines.append(f"- Known fires with **no** matching cluster: {', '.join(_missed)}. Possible reasons: "
                      "its ignition is outside the box, it is merged into a neighbouring cluster, or the "
                      "sliders are too strict.")
    if len(_unmatched):
        _lines.append("")
        _lines.append("**Clusters that did not match any known fire:**")
        for r in _unmatched.itertuples():
            _lines.append(f"- cluster {r.cluster}: {r.first_date} to {r.last_date}, "
                          f"{r.n_modis + r.n_viirs:,} detections, centroid ({r.centroid_lat}, {r.centroid_lon})")
    elif len(summary):
        _lines.append("")
        _lines.append("Every cluster matched a known fire.")
    _lines.append("")
    _lines.append(f"{n_noise:,} detections were noise (not part of any cluster).")

    # What this means for the MODIS-VIIRS comparison.
    _lines.append("")
    _lines.append("**What this means for harmonizing MODIS and VIIRS.** Once a cluster is tied to one real "
                  "fire, its MODIS and VIIRS detections describe the *same* event, so we can compare them "
                  "directly:")
    if len(_matched):
        for _name, _g in _matched.groupby("matched_fire"):
            _nm, _nv = int(_g["n_modis"].sum()), int(_g["n_viirs"].sum())
            _fm, _fv = _g["frp_modis_MW"].sum(), _g["frp_viirs_MW"].sum()
            if _nm and _nv:
                _lines.append(f"- {_name}: VIIRS/MODIS = {_nv / _nm:.1f}× in detection count ({_nv:,} vs {_nm:,}) "
                              f"and {_fv / _fm:.2f}× in total FRP ({_fv:,.0f} vs {_fm:,.0f} MW).")
            else:
                _lines.append(f"- {_name}: only one sensor family is selected or present, so no ratio.")
        _lines.append("")
        _lines.append("If the count ratio is much larger than the FRP ratio, VIIRS is seeing many small, "
                      "low-power pixels that MODIS's coarser pixels blend together. Those per-fire ratios "
                      "are the starting point for a correction that puts both sensors on one scale.")
    else:
        _lines.append("- No cluster matched a known fire with the current settings, so there is nothing to "
                      "compare yet. Try another year or adjust the sliders.")
    mo.md("\n".join(_lines))
    return


if __name__ == "__main__":
    app.run()
