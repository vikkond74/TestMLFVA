"""
ML Exclusion FVA Analyzer — Simple edition
==========================================
Upload a forecast extract at any level and judge whether ML exclusions are
justified: does the user/consensus forecast beat the ML engine's own (shadow)
forecast against shipped actuals?

Simple methodology (deliberately):
    Variance = Forecast − Shipped          (signed, units)
    Bias %   = Variance / Shipped          (positive = over-forecast)
    FCA      = 1 − |Variance| / Shipped    (floored at 0)
    Verdict  = compare FCA of User vs ML
"""

import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="ML Exclusion FVA Analyzer", page_icon="🎯",
                   layout="wide")
st.title("🎯 ML Exclusion FVA Analyzer")
st.caption("Is each ML exclusion earning its keep? Variance = FC − Shipped · "
           "Bias % = Variance / Shipped · FCA = 1 − |Variance| / Shipped.")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_file(file_bytes, file_name):
    if file_name.lower().endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(file_bytes))
    else:
        df = pd.read_csv(io.BytesIO(file_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    return df


def as_str(series):
    """NA-safe string conversion (pandas 3.x astype(str) keeps NA)."""
    return series.astype(str).fillna("(blank)")


def parse_periods(vals):
    """Interpret period values as calendar months, tolerating MIXED types in
    one column. Returns parsed datetimes aligned to vals, or None if values
    carry no calendar info (e.g. relative period numbers 1..12)."""
    s = pd.Series(list(vals))
    num = pd.to_numeric(s, errors="coerce")
    if (num.notna().all() and num.between(1, 600).all()
            and (num % 1 == 0).all()):
        return None
    parsed = pd.Series(pd.NaT, index=s.index)
    is_yyyymm = num.notna() & num.between(190001, 209912)
    if is_yyyymm.any():
        parsed[is_yyyymm] = pd.to_datetime(
            num[is_yyyymm].astype(int).astype(str), format="%Y%m",
            errors="coerce")
    rest = parsed.isna()
    if rest.any():
        ext = s[rest].astype(str).str.strip().str.extract(
            r"^(?P<m>\d{1,2})[./\-](?P<y>\d{4})$")
        ok = ext["m"].notna() & ext["y"].notna()
        if ok.any():
            mm = ext.loc[ok, "m"].astype(int)
            idx = ext.index[ok][mm.between(1, 12)]
            parsed.loc[idx] = pd.to_datetime(
                ext.loc[idx, "y"] + "-" + ext.loc[idx, "m"].str.zfill(2),
                format="%Y-%m", errors="coerce")
    rest = parsed.isna()
    if rest.any():
        try:
            parsed[rest] = pd.to_datetime(s[rest].astype(str), format="mixed",
                                          dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            parsed[rest] = pd.to_datetime(s[rest].astype(str), dayfirst=True,
                                          errors="coerce")
    return parsed if parsed.notna().mean() >= 0.9 else None


def group_metrics(frame, by, fca_level, month_col):
    """Metrics for each group in `by` (or one overall row if `by` is empty).

    FCA is computed the statistically sound way: absolute errors at the FCA
    calculation level (e.g. SKU) x month, summed, divided by summed shipments
    -- equivalent to a volume-weighted average of item-level FCAs. No netting.
    Variance and Bias are net by design (over/under SHOULD offset there).
    """
    grain = list(by)
    grain += [c for c in fca_level if c not in grain]
    if month_col not in grain:
        grain.append(month_col)
    base = (frame.groupby(grain, dropna=False)
            .agg(ml=("ml", "sum"), user=("user", "sum"), act=("act", "sum"))
            .reset_index())
    base["em"] = (base["ml"] - base["act"]).abs()
    base["eu"] = (base["user"] - base["act"]).abs()
    if by:
        g = (base.groupby(list(by), dropna=False)
             .agg(ml=("ml", "sum"), user=("user", "sum"), act=("act", "sum"),
                  em=("em", "sum"), eu=("eu", "sum"))
             .reset_index())
    else:
        g = pd.DataFrame([{"ml": base["ml"].sum(), "user": base["user"].sum(),
                           "act": base["act"].sum(), "em": base["em"].sum(),
                           "eu": base["eu"].sum()}])
    a = g["act"]
    out = g[list(by)].copy() if by else pd.DataFrame(index=g.index)
    out["ML FC"] = g["ml"]
    out["User FC"] = g["user"]
    out["Shipped"] = a
    out["Variance ML"] = g["ml"] - a
    out["Variance User"] = g["user"] - a
    out["Bias % ML"] = pd.Series(np.where(a > 0, (g["ml"] - a) / a, np.nan))
    out["Bias % User"] = pd.Series(np.where(a > 0, (g["user"] - a) / a, np.nan))
    out["FCA ML"] = pd.Series(np.where(a > 0, 1 - g["em"] / a, np.nan)).clip(lower=0)
    out["FCA User"] = pd.Series(np.where(a > 0, 1 - g["eu"] / a, np.nan)).clip(lower=0)
    return out


def add_verdict(df_, thr):
    diff = df_["FCA User"] - df_["FCA ML"]
    df_["FCA User − ML"] = diff
    df_["Verdict"] = np.select(
        [diff >= thr, diff <= -thr],
        ["👤 User adds value", "🤖 ML more accurate"], default="≈ Tie")
    df_.loc[diff.isna(), "Verdict"] = "⚪ No demand"
    return df_


NUM_COLS = ["ML FC", "User FC", "Shipped", "Variance ML", "Variance User"]
PCT_COLS = ["Bias % ML", "Bias % User", "FCA ML", "FCA User", "FCA User − ML"]
COLUMN_CONFIG = {
    **{c: st.column_config.NumberColumn(format="localized") for c in NUM_COLS},
    **{c: st.column_config.NumberColumn(format="percent") for c in PCT_COLS},
    "Variance ML": st.column_config.NumberColumn(
        format="localized", help="ML FC − Shipped (positive = over-forecast)."),
    "Variance User": st.column_config.NumberColumn(
        format="localized", help="User FC − Shipped (positive = over-forecast)."),
    "FCA User − ML": st.column_config.NumberColumn(
        format="percent", help="Positive = the user forecast adds value."),
}


# -----------------------------------------------------------------------------
# 1 · Upload
# -----------------------------------------------------------------------------
SAMPLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample_data.csv")


@st.cache_data(show_spinner=False)
def load_sample_bytes():
    """Read the bundled sample if present; else synthesize a tiny one so the
    'try sample' button always works regardless of deployment."""
    try:
        with open(SAMPLE_PATH, "rb") as fh:
            return fh.read()
    except OSError:
        rng = np.random.default_rng(7)
        rows = []
        for i in range(1, 21):
            brand = ["ACME", "BORA", "CIRA"][i % 3]
            ctry = ["DE", "FR", "PL"][i % 3]
            excl = "Y" if i % 2 else "N"
            base = int(rng.integers(20, 300))
            for m in range(1, 13):
                act = max(0, int(base * (1 + 0.3 * np.sin(m)) + rng.normal(0, base * .1)))
                ml = max(0, int(act * rng.normal(1.1, .15)))
                usr = max(0, int(act * rng.normal(1.0, .12)))
                rows.append(f"{brand},RE{i:06d}{ctry},{excl},{m},{ml},{usr},{act}")
        header = ("SapCode,SPU Id,ML Exclusion,Month,ML Forecast,"
                  "Global Lag 0 PBU Consensus FC,Shipped Units")
        return ("\n".join([header] + rows)).encode()


uploaded = st.file_uploader(
    "Upload your extract (CSV or Excel, long format: one row per item per month)",
    type=["csv", "xlsx", "xlsm", "xls"],
)

if uploaded is None:
    st.info("**New here?** This tool checks whether each ML exclusion is "
            "earning its keep — i.e. whether the human/consensus forecast "
            "actually beats the ML engine against what shipped. Upload your "
            "own extract, or try the sample below to see how it works.")
    c1, c2 = st.columns(2)
    use_sample = c1.button("▶️ Load sample data", type="primary",
                           width="stretch")
    c2.download_button("⬇️ Download sample file", load_sample_bytes(),
                       "sample_data.csv", "text/csv",
                       width="stretch",
                       help="A fully synthetic example (random brands, SPUs, "
                            "values) you can open in Excel to see the expected "
                            "layout.")
    with st.expander("What columns does my file need?"):
        st.markdown(
            "- **Dimension column(s)** at any level (SKU, SPU, country …)\n"
            "- **ML exclusion flag** — Y/N\n"
            "- **Month / period**\n"
            "- **ML forecast** — the engine's own (shadow) output\n"
            "- **User / consensus forecast**\n"
            "- **Shipped units** — actuals\n\n"
            "One row per item per month. Column names are auto-detected and "
            "re-mappable in the sidebar.")
    if not use_sample:
        st.stop()
    raw = load_file(load_sample_bytes(), "sample_data.csv")
    st.success("Loaded the synthetic sample — explore the views below, then "
               "upload your own file when ready.")
else:
    raw = load_file(uploaded.getvalue(), uploaded.name)

all_cols = list(raw.columns)

# -----------------------------------------------------------------------------
# 2 · Column mapping
# -----------------------------------------------------------------------------
st.sidebar.header("1 · Column mapping")


def guess_sequential(columns):
    taken, out = set(), {}
    specs = [
        ("excl",  ["exclusion", "excluded", "excl flag"]),
        ("month", ["month", "period", "fiscper", "date"]),
        ("act",   ["shipped", "actual", "sales qty", "deliver", "sales"]),
        ("ml",    ["ml forecast", "ml fc", "stat fc", "statistical", "engine",
                   "shadow"]),
        ("user",  ["consensus", "user forecast", "user fc", "final fc",
                   "demand plan", "adopted", "pbu"]),
        ("ml",    ["ml"]),
        ("user",  ["user", "final"]),
    ]
    for slot, kws in specs:
        if slot in out:
            continue
        for col in columns:
            if col in taken:
                continue
            if any(kw in col.lower() for kw in kws):
                out[slot] = col
                taken.add(col)
                break
    return out


guesses = guess_sequential(all_cols)


def sel(label, slot, key):
    guess = guesses.get(slot)
    idx = all_cols.index(guess) if guess in all_cols else 0
    return st.sidebar.selectbox(label, all_cols, index=idx, key=key)


col_excl = sel("ML exclusion flag (Y/N)", "excl", "c_excl")
col_month = sel("Month / period", "month", "c_month")
col_ml = sel("ML forecast (shadow / engine output)", "ml", "c_ml")
col_user = sel("User / consensus forecast", "user", "c_user")
col_act = sel("Shipped units (actuals)", "act", "c_act")

mapped = [col_excl, col_month, col_ml, col_user, col_act]
if len(set(mapped)) < len(mapped):
    st.error("⛔ **Same column mapped to two roles.** Each of the five roles "
             "in the sidebar must point at a different column.")
    st.stop()

_flag_vals = set(raw[col_excl].dropna().astype(str).str.strip().str.upper()
                 .unique())
if not (_flag_vals & {"Y", "YES", "TRUE", "1", "X"}):
    st.warning(f"⚠️ The exclusion-flag column **{col_excl}** contains no "
               f"Y/Yes/True/1/X values — every item will be treated as not "
               f"excluded. Values found: `{sorted(_flag_vals)[:10]}`.")

for _role, _c in [("ML forecast", col_ml), ("User forecast", col_user),
                  ("Shipped units", col_act)]:
    if pd.to_numeric(raw[_c], errors="coerce").notna().mean() < 0.5:
        st.warning(f"⚠️ **{_role}** is mapped to `{_c}`, but most of its "
                   "values are not numeric — this looks like a wrong mapping.")

with st.expander("🧭 Column mapping in use", expanded=False):
    st.table(pd.DataFrame({
        "Role": ["Exclusion flag", "Month", "ML forecast", "User forecast",
                 "Shipped units"],
        "Column": mapped,
        "Sample": [str(raw[c].dropna().iloc[0]) if raw[c].notna().any()
                   else "—" for c in mapped],
    }))

# -----------------------------------------------------------------------------
# 3 · Period handling: normalize, completed months only, last-12 cap
# -----------------------------------------------------------------------------
_pvals = list(raw[col_month].dropna().unique())
_pnorm = parse_periods(_pvals)
if _pnorm is not None:
    _pmap = {v: (f"{d.year}-{d.month:02d}" if pd.notna(d) else None)
             for v, d in zip(_pvals, _pnorm)}
    _before = len(raw)
    raw = raw.copy()
    raw[col_month] = raw[col_month].map(_pmap)
    raw = raw[raw[col_month].notna()]
    if len(raw) < _before:
        st.warning(f"⚠️ {_before - len(raw)} row(s) dropped: period value "
                   "could not be interpreted as a month.")
else:
    _pnum_full = pd.to_numeric(raw[col_month], errors="coerce")
    if _pnum_full.notna().mean() >= 0.9:
        raw = raw[_pnum_full.notna()].copy()
        raw[col_month] = pd.to_numeric(raw[col_month])
    else:
        raw = raw.copy()
        raw[col_month] = raw[col_month].astype(str)

_period_vals = list(raw[col_month].dropna().unique())
_parsed = parse_periods(_period_vals)
if _parsed is not None:
    _cutoff = pd.Timestamp.today().normalize().replace(day=1)
    _keep = {v for v, p in zip(_period_vals, _parsed)
             if pd.notna(p) and p < _cutoff}
    _dropped = [v for v in _period_vals if v not in _keep]
    if _dropped:
        raw = raw[raw[col_month].isin(_keep)]
        st.info(f"🗓️ **{len(_dropped)} period(s) excluded** — only completed "
                f"months are scored (today is {pd.Timestamp.today():%d %b %Y})."
                f" Dropped: {', '.join(str(d) for d in sorted(_dropped)[:8])}"
                f"{'…' if len(_dropped) > 8 else ''}.")
    if raw.empty:
        st.error("No completed months left — the file only contains "
                 "current/future periods.")
        st.stop()
else:
    _nums = pd.to_numeric(pd.Series(_period_vals), errors="coerce")
    _cur_m = pd.Timestamp.today().month
    _cur_y = pd.Timestamp.today().year
    if (_nums.notna().all() and _nums.between(1, 12).all()
            and (_nums < _cur_m).all()):
        st.caption(f"🗓️ Period numbers {int(_nums.min())}–{int(_nums.max())} "
                   f"read as calendar months of {_cur_y}; all completed.")
    elif _nums.notna().all() and _nums.between(1, 12).all():
        interp = st.radio(
            f"⚠️ Month numbers reach {int(_nums.max())}, but only months "
            f"1–{_cur_m - 1} of {_cur_y} are completed. How should they be "
            "read?",
            [f"Calendar months of {_cur_y} → drop months ≥ {_cur_m} as "
             "incomplete",
             "Sequence numbers of past periods → keep all"], index=0)
        if interp.startswith("Calendar"):
            _keep_nums = {v for v, n in zip(_period_vals, _nums) if n < _cur_m}
            _dropped = [v for v in _period_vals if v not in _keep_nums]
            raw = raw[raw[col_month].isin(_keep_nums)]
            if _dropped:
                st.info(f"🗓️ Dropped month(s) "
                        f"{', '.join(str(d) for d in sorted(_dropped))} as "
                        "not yet completed.")
            if raw.empty:
                st.error("No completed months left after the cutoff.")
                st.stop()
    else:
        st.warning(f"⚠️ **Current/future months could NOT be auto-dropped** — "
                   f"the period column `{col_month}` carries no recognizable "
                   f"calendar information (values look like: "
                   f"`{', '.join(str(v) for v in _period_vals[:5])}`…). Use "
                   "the sidebar control to drop trailing incomplete periods.")
    _drop_n = st.sidebar.number_input(
        "Drop trailing period(s) as incomplete", min_value=0, max_value=6,
        value=0)
    if _drop_n:
        _keep_periods = sorted(raw[col_month].dropna().unique())[:-_drop_n]
        raw = raw[raw[col_month].isin(_keep_periods)]
        st.info(f"🗓️ Last {_drop_n} period(s) dropped as incomplete.")

st.sidebar.header("2 · Settings")
months_all = sorted(raw[col_month].dropna().unique().tolist())
months_sorted = months_all[-12:]
if len(months_all) > 12:
    st.sidebar.caption(f"ℹ️ {len(months_all)} periods in file — limited to "
                       f"the most recent 12.")
dim_candidates = [c for c in all_cols
                  if c not in {col_excl, col_month, col_ml, col_user, col_act}]
fca_level = st.sidebar.multiselect(
    "FCA calculation level", dim_candidates,
    default=dim_candidates[:1],
    help="FCA is computed from absolute errors at THIS level × month, then "
         "volume-weighted up (no netting across items). Default is the first "
         "dimension column (e.g. SapCode). Add columns to go finer — e.g. "
         "SapCode + Country scores each brand×country separately, which "
         "exposes mistakes that a SapCode-only view would let cancel out. "
         "Pick only the SKU column for SKU-level FCA.")
if not fca_level:
    st.sidebar.error("Pick at least one column for the FCA level.")
    st.stop()
with st.sidebar.expander("How the FCA level changes the number"):
    st.markdown(
        "FCA = 1 − Σ|Forecast − Shipped| ÷ ΣShipped, with the errors summed "
        "at the level you choose × month.\n\n"
        "- **SapCode only:** a brand's SPUs and countries are added together "
        "first, then compared to shipped. Over-forecasts on one SPU hide "
        "under-forecasts on another → a more forgiving, higher number.\n"
        "- **SapCode + Country:** each brand×country is scored separately and "
        "then volume-weighted up. Misses can no longer cancel across "
        "countries → a stricter, usually lower number that reflects how "
        "wrong the plan was *where it mattered*.\n\n"
        "Finer level = less netting = more honest. Default is the first "
        "dimension column.")
fca_threshold = st.sidebar.slider(
    "Verdict materiality threshold (pp of FCA)", 1, 25, 5,
    help="How much higher one side's FCA must be before it counts as a clear "
         "win rather than a tie.") / 100.0

# -----------------------------------------------------------------------------
# 4 · Prepare data
# -----------------------------------------------------------------------------
df = raw[raw[col_month].isin(months_sorted)].copy()
for c in [col_ml, col_user, col_act]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
df["_excl"] = (df[col_excl].astype(str).str.strip().str.upper()
               .map(lambda v: "Y" if v in ("Y", "YES", "TRUE", "1", "X")
                    else "N"))
df = df.rename(columns={col_ml: "ml", col_user: "user", col_act: "act"})

# -----------------------------------------------------------------------------
# 5 · Data-quality checks
# -----------------------------------------------------------------------------
excl_rows = df[df["_excl"] == "Y"]
nonzero_excl = excl_rows[(excl_rows["ml"] != 0) | (excl_rows["user"] != 0)]
ident = ((nonzero_excl["ml"] == nonzero_excl["user"]).mean()
         if len(nonzero_excl) else np.nan)
with st.expander("🔍 Data-quality checks",
                 expanded=bool(pd.notna(ident) and ident > 0.5)):
    c1, c2, c3 = st.columns(3)
    c1.metric("Excluded rows where ML = user FC",
              "—" if pd.isna(ident) else f"{ident:0.0%}",
              help="If the ML column merely echoes the user forecast on "
                   "excluded items, the engine's shadow forecast was not "
                   "captured and the comparison is blind exactly where it "
                   "matters.")
    c2.metric("Rows with zero shipments", f"{(df['act'] == 0).mean():0.0%}")
    c3.metric("Periods analysed", f"{len(months_sorted)}")
    if pd.notna(ident) and ident > 0.5:
        st.error("More than half of the non-zero excluded rows have an "
                 "identical ML and user forecast — the ML column likely holds "
                 "the *adopted* forecast, not the engine's shadow forecast. "
                 "Fix the extract before acting on verdicts.")

# -----------------------------------------------------------------------------
# 6 · Scorecard (All / Y / N)
# -----------------------------------------------------------------------------
st.subheader("Whole-dataset scorecard")
st.caption(
    f"Totals across the last {len(months_sorted)} completed month(s), split "
    "into all items, ML-**excluded** (Y), and **not excluded** (N). "
    f"**Forecast accuracy (FCA) is calculated at "
    f"`{' + '.join(fca_level)}` level** (change it in the sidebar): absolute "
    "errors are measured at that level × month and volume-weighted up, so "
    "offsetting misses are not cancelled out. Variance and Bias %, by "
    "contrast, are net totals."
)
with st.expander("ℹ️ How the verdict is decided"):
    st.markdown(
        "For the **excluded (Y)** items we compare two forecast accuracies "
        "against what actually shipped:\n\n"
        "- **FCA User** — accuracy of the human/consensus forecast that "
        "*replaced* ML on excluded items\n"
        "- **FCA ML** — accuracy the ML engine's own (shadow) forecast "
        "*would* have had\n\n"
        f"The gap is **FVA = FCA User − FCA ML**. If the user is higher by "
        f"more than the materiality threshold (currently "
        f"**{fca_threshold:.0%}**), the human **adds value** and the "
        "exclusion is justified 👤. If ML is higher by more than the "
        "threshold, **ML would have been more accurate** 🤖 and the "
        "exclusion is a removal candidate. Inside the threshold it's a tie. "
        "The threshold is adjustable in the sidebar.")
rows = []
for label, frame in [("All items", df),
                     ("Excluded (Y)", df[df["_excl"] == "Y"]),
                     ("Not excluded (N)", df[df["_excl"] == "N"])]:
    m = group_metrics(frame, [], fca_level, col_month)
    m.insert(0, "Scope", label)
    rows.append(m)
sc = add_verdict(pd.concat(rows, ignore_index=True), fca_threshold)
st.dataframe(sc, width="stretch", hide_index=True,
             column_config=COLUMN_CONFIG)

y = sc[sc["Scope"] == "Excluded (Y)"].iloc[0]
if pd.isna(y["FCA User − ML"]):
    st.info("No shipped volume on excluded items — no overall verdict.")
elif y["FCA User − ML"] >= fca_threshold:
    st.success(f"**Excluded items overall: 👤 the user forecast adds value** — "
               f"FCA {y['FCA User']:.0%} vs {y['FCA ML']:.0%} for ML.")
elif y["FCA User − ML"] <= -fca_threshold:
    st.error(f"**Excluded items overall: 🤖 ML would be more accurate** — "
             f"FCA {y['FCA ML']:.0%} vs {y['FCA User']:.0%} for the user "
             "forecast. Review the exclusion list.")
else:
    st.warning(f"**Excluded items overall: ≈ tie** — FCA difference "
               f"({y['FCA User − ML']:+.1%}) is below the "
               f"{fca_threshold:.0%} threshold.")

# -----------------------------------------------------------------------------
# 7 · Summed-up analysis by any column
# -----------------------------------------------------------------------------
st.subheader("Summed-up analysis by…")
cset1, cset2 = st.columns([1.6, 2.4])
with cset1:
    measure_cols = {col_month, "ml", "user", "act", col_ml, col_user, col_act}
    bd_options = [c for c in df.columns
                  if c not in measure_cols and c != "_excl"]
    bd_options = [col_excl] + [c for c in bd_options if c != col_excl]
    bcol = st.selectbox("Break down by", bd_options, index=0)
    excl_scope = st.radio("ML exclusion scope",
                          ["Both (split Y/N)", "Y only", "N only"],
                          horizontal=True)
    view_mode = st.radio("View", ["Per month", "Aggregated"], horizontal=True)
with cset2:
    _pp = parse_periods(months_sorted)
    if _pp is not None:
        _ymap = {v: (d.year if pd.notna(d) else None)
                 for v, d in zip(months_sorted, _pp)}
        _years = sorted({yv for yv in _ymap.values() if yv is not None})
        sel_years = st.multiselect("Years", _years, default=_years)
        period_opts = [m for m in months_sorted
                       if _ymap.get(m) in set(sel_years)]
    else:
        period_opts = months_sorted
    sel_periods = st.multiselect("Months / periods", period_opts,
                                 default=period_opts)

if not sel_periods:
    st.info("Select at least one period to see the summed-up analysis.")
    st.stop()

dfb = df[df[col_month].isin(sel_periods)]
if excl_scope == "Y only":
    dfb = dfb[dfb["_excl"] == "Y"]
elif excl_scope == "N only":
    dfb = dfb[dfb["_excl"] == "N"]
if dfb.empty:
    st.warning("No rows match the selected periods and exclusion scope.")
    st.stop()

split_by_excl = excl_scope.startswith("Both") and bcol != col_excl
keys = [bcol, "_excl"] if split_by_excl else [bcol]
group_keys = keys + ([col_month] if view_mode == "Per month" else [])

bsum = group_metrics(dfb, group_keys, fca_level, col_month)
bsum = add_verdict(bsum, fca_threshold)

# Order: biggest groups first; Y before N; months in order
vol_order = (bsum.groupby(bcol)["Shipped"].sum()
             .sort_values(ascending=False).index.tolist())
bsum["_r"] = bsum[bcol].map({v: i for i, v in enumerate(vol_order)})
sort_keys, sort_asc = ["_r"], [True]
if split_by_excl:
    sort_keys.append("_excl"); sort_asc.append(False)
if view_mode == "Per month":
    sort_keys.append(col_month); sort_asc.append(True)
bsum = bsum.sort_values(sort_keys, ascending=sort_asc).drop(columns="_r")

st.caption(f"Scope: **{excl_scope}** · View: **{view_mode}** · "
           f"{len(sel_periods)} period(s)")
disp = bsum.rename(columns={"_excl": "Excluded"})
show_cols = ([str(bcol)]
             + (["Excluded"] if split_by_excl else [])
             + ([str(col_month)] if view_mode == "Per month" else [])
             + NUM_COLS + PCT_COLS + ["Verdict"])
st.dataframe(disp[show_cols], width="stretch", hide_index=True,
             column_config=COLUMN_CONFIG)
st.download_button(
    "⬇️ Download this table (CSV)",
    disp[show_cols].to_csv(index=False).encode(),
    "summed_up_analysis.csv", "text/csv")

# ---- FCA chart (always aggregated view) --------------------------------------
csum = group_metrics(dfb, keys, fca_level, col_month)
csum = csum.sort_values("Shipped", ascending=False)
if view_mode == "Per month":
    st.caption("Chart shows the selected periods aggregated.")

if split_by_excl:
    top_groups = (csum.groupby(bcol)["Shipped"].sum()
                  .sort_values(ascending=False).head(12).index.tolist())
    plot_bd = csum[csum[bcol].isin(top_groups) & csum["FCA ML"].notna()]
    if not plot_bd.empty:
        long = plot_bd.melt(id_vars=[bcol, "_excl"],
                            value_vars=["FCA ML", "FCA User"],
                            var_name="side", value_name="fca")
        figb = px.bar(long, x=bcol, y="fca", color="side", barmode="group",
                      facet_col="_excl",
                      color_discrete_map={"FCA ML": "#ef4444",
                                          "FCA User": "#3b82f6"},
                      category_orders={bcol: [str(g) for g in top_groups],
                                       "_excl": ["Y", "N"]},
                      labels={"fca": "FCA", "side": ""})
        figb.for_each_annotation(lambda a: a.update(
            text="Excluded (Y)" if a.text.endswith("Y")
            else ("Not excluded (N)" if a.text.endswith("N") else a.text)))
        figb.update_layout(height=440, legend=dict(orientation="h", y=1.12))
        figb.update_yaxes(tickformat=".0%")
        st.plotly_chart(figb, width="stretch")
        st.caption("Top 12 groups by shipped volume.")
else:
    plot_bd = csum[csum["FCA ML"].notna()].head(20)
    if len(plot_bd) > 1:
        figb = go.Figure()
        figb.add_bar(x=as_str(plot_bd[bcol]), y=plot_bd["FCA ML"],
                     name="FCA ML", marker_color="#ef4444")
        figb.add_bar(x=as_str(plot_bd[bcol]), y=plot_bd["FCA User"],
                     name="FCA User", marker_color="#3b82f6")
        figb.update_layout(barmode="group", height=420,
                           yaxis_tickformat=".0%", yaxis_title="FCA",
                           xaxis_title=str(bcol),
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(figb, width="stretch")
        st.caption("Top 20 groups by shipped volume.")

# -----------------------------------------------------------------------------
# 8 · Item drill-in (level chosen by the user; same metric logic as above)
# -----------------------------------------------------------------------------
st.subheader("🔬 Item drill-in")
d1, d2 = st.columns([1, 3])
with d1:
    drill_flag = st.selectbox("ML Exclusion", ["Y", "N", "All"], index=0,
                              key="drill_flag")
with d2:
    drill_cols = st.multiselect(
        "Drill level — pick the column(s) that define an item",
        dim_candidates, default=dim_candidates[:1],
        help="Items are listed at exactly this level: pick SPU for SPU-level "
             "items, SPU + SKU for that combination, SPU + Cluster for that "
             "one. Variance/Bias stay net per item; FCA still comes from "
             "absolute errors at the FCA calculation level × month "
             "(sidebar), volume-weighted — identical logic to the tables "
             "above.")

if not drill_cols:
    st.info("Pick at least one column to drill into.")
else:
    dfd = df[df[col_month].isin(sel_periods)]
    if drill_flag != "All":
        dfd = dfd[dfd["_excl"] == drill_flag]
    if dfd.empty:
        st.warning("No rows match this exclusion scope and period selection.")
    else:
        items = group_metrics(dfd, drill_cols, fca_level, col_month)
        items = add_verdict(items, fca_threshold)
        items = items.sort_values("FCA User − ML", na_position="last")
        st.caption(f"{len(items)} item(s) at level "
                   f"**{' › '.join(drill_cols)}** · scope **{drill_flag}** · "
                   "sorted worst FVA first.")
        st.dataframe(items, width="stretch", hide_index=True, height=380,
                     column_config=COLUMN_CONFIG)
        st.download_button(
            "⬇️ Download item table (CSV)",
            items.to_csv(index=False).encode(),
            "item_drill_in.csv", "text/csv")

        # ---- Monthly detail for one item ------------------------------------
        lbl = as_str(items[drill_cols[0]])
        for c in drill_cols[1:]:
            lbl = lbl + " | " + as_str(items[c])
        items = items.assign(_label=lbl)
        pick = st.selectbox("Monthly detail for", items["_label"].tolist(),
                            key="drill_pick")
        sel_it = items[items["_label"] == pick].iloc[0]
        mask = pd.Series(True, index=dfd.index)
        for c in drill_cols:
            mask &= as_str(dfd[c]) == str(sel_it[c])
        per_month = group_metrics(dfd[mask], [col_month], fca_level,
                                  col_month)
        per_month = add_verdict(per_month, fca_threshold)
        per_month = per_month.sort_values(col_month)

        figd = go.Figure()
        figd.add_bar(x=per_month[col_month], y=per_month["Shipped"],
                     name="Shipped", marker_color="#94a3b8", opacity=0.55)
        figd.add_scatter(x=per_month[col_month], y=per_month["ML FC"],
                         name="ML FC", mode="lines+markers",
                         line=dict(color="#ef4444", width=2))
        figd.add_scatter(x=per_month[col_month], y=per_month["User FC"],
                         name="User FC", mode="lines+markers",
                         line=dict(color="#3b82f6", width=2))
        figd.update_layout(height=400, xaxis_title="Month",
                           yaxis_title="Units",
                           legend=dict(orientation="h", y=1.08))
        st.plotly_chart(figd, width="stretch")
        st.dataframe(per_month, width="stretch", hide_index=True,
                     column_config=COLUMN_CONFIG)

st.caption("Methodology: Variance = FC − Shipped and Bias % = Variance / "
           "Shipped are NET (over/under offset by design). FCA is NOT "
           "netted: absolute errors are computed at the FCA calculation "
           "level × month (see sidebar), summed, and divided by summed "
           "shipments — equivalent to volume-weighting item-level FCAs.")
