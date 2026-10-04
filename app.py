"""
Sri Lanka Car Price Dashboard
-----------------------------
- Tab 1: Predict a car's price with the trained pipeline (Model.pkl).
         Every input is restricted to values that actually occur in formatted_data.csv,
         and the options cascade (Brand -> Model -> Condition -> Fuel -> ... ) so an
         impossible combination (e.g. a petrol-only BMW 318i as Diesel, a 10,000cc engine)
         can never be submitted.
- Tab 2: Interactive EDA (Plotly) with cross-filters, group comparisons, scatter explorer,
         depreciation curves, correlation heatmap, market map and CSV download.

Run with:  streamlit run app.py
Requires:  streamlit pandas numpy joblib scikit-learn plotly
Place this file next to Model.pkl and formatted_data.csv.
"""

import pickle
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_PATH = BASE_DIR / "Model.pkl"
DATA_PATH = BASE_DIR / "formatted_data.csv"
CURRENT_YEAR = datetime.now().year

NUMERIC_FEATURES = [
    "Car_Age", "Mileage_numeric", "Engine (cc)", "Number of Cylinders",
    "Power Steering", "Power Shutters", "Power Mirrors",
]
CATEGORICAL_FEATURES = [
    "Brand", "Model_cleaned", "Gear", "Fuel Type", "Condition", "Body Type",
]
FEATURE_ORDER = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MIN_REAL_ENGINE_CC = 500          # anything smaller in the data is a scraping error (e.g. 1 cc)
NOISE_MIN_COUNT = 5               # an engine size / cylinder value is "real" for a model if it
NOISE_MIN_SHARE = 0.04            # appears >= 5 times OR in >= 4% of that model's listings

st.set_page_config(page_title="Sri Lanka Car Price Dashboard", layout="wide")

# Hide Deploy button from top right
st.markdown(
    """
    <style>
    .stAppDeployButton,
    [data-testid="stAppDeployButton"],
    .stDeployButton,
    [data-testid="stDeployButton"] {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_model(path: Path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    """Reproduce the cleaning / feature-engineering used to train the model so the
    allowed input values and the EDA reflect exactly what the model was fit on."""
    df = pd.read_csv(path)

    df = df.drop_duplicates()
    df = df.dropna(subset=["Price_numeric", "Mileage_numeric", "Number of Cylinders",
                           "Engine (cc)", "YOM"])
    df = df[(df["Engine (cc)"] > 0) & (df["Engine (cc)"] < 10_000)]
    df = df[df["Condition"] != "Antique"]
    df = df[df["Price_numeric"] >= 50_000]

    model_counts = df["Model_cleaned"].value_counts()
    df = df[df["Model_cleaned"].isin(model_counts[model_counts >= 2].index)]

    df["strat_col"] = list(zip(df["Model_cleaned"], df["Condition"]))
    strat_counts = df["strat_col"].value_counts()
    df = df[df["strat_col"].isin(strat_counts[strat_counts >= 3].index)]
    df = df.drop(columns="strat_col")

    df["Car_Age"] = CURRENT_YEAR - df["YOM"]

    for col in ["Power Steering", "Power Shutters", "Power Mirrors"]:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0}).fillna(0)
        )

    # Extra columns used only for display / EDA
    df["Price_M"] = df["Price_numeric"] / 1e6
    df["Location"] = df["Location"].astype(str).str.strip().str.title()
    df["Manufacturing Country"] = df["Manufacturing Country"].astype(str).str.strip()
    return df.reset_index(drop=True)


@st.cache_data
def build_model_labels(_df: pd.DataFrame) -> dict:
    """Model_cleaned ('fitgp5sgradesafety') -> readable name ('Fit GP5 S Grade Safety')."""
    labels = _df.groupby("Model_cleaned")["Model"].agg(lambda s: s.value_counts().index[0]).to_dict()
    seen = {}
    for key, lab in labels.items():
        seen.setdefault(lab, []).append(key)
    for lab, keys in seen.items():
        if len(keys) > 1:                                  # disambiguate duplicate labels
            for k in keys:
                labels[k] = f"{lab} [{k}]"
    return labels


try:
    model = load_model(MODEL_PATH)
    df = load_data(DATA_PATH)
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a required file: {e}\n\n"
        "Make sure `Model.pkl` and `formatted_data.csv` sit in the same folder as this script."
    )
    st.stop()

MODEL_LABELS = build_model_labels(df)
df["Model_label"] = df["Model_cleaned"].map(MODEL_LABELS)


# --------------------------------------------------------------------------
# Helpers for "only possible values"
# --------------------------------------------------------------------------
def common_values(sub: pd.DataFrame, col: str, min_count: int = 1, min_share: float = 0.0):
    """Sorted values of `col` seen in `sub`, dropping rare (noisy) ones."""
    vc = sub[col].dropna().value_counts()
    if vc.empty:
        return []
    keep = vc[(vc >= min_count) | (vc / vc.sum() >= min_share)]
    if keep.empty:
        keep = vc
    return sorted(keep.index)


def valid_engine_sizes(brand: str, model_key: str, fuel: str) -> list:
    """Engine sizes that exist for this model + fuel (falls back to model, then brand)."""
    real = df[df["Engine (cc)"] >= MIN_REAL_ENGINE_CC]
    for sub in (
        real[(real["Model_cleaned"] == model_key) & (real["Fuel Type"] == fuel)],
        real[real["Model_cleaned"] == model_key],
        real[(real["Brand"] == brand) & (real["Fuel Type"] == fuel)],
        real,
    ):
        vals = common_values(sub, "Engine (cc)", NOISE_MIN_COUNT, NOISE_MIN_SHARE)
        if vals:
            return [int(v) for v in vals]
    return []


def valid_cylinders(brand: str, model_key: str, fuel: str, cc: int) -> list:
    """Cylinder counts that exist for this model + fuel + engine size."""
    real = df[df["Engine (cc)"] >= MIN_REAL_ENGINE_CC]
    for sub in (
        real[(real["Model_cleaned"] == model_key) & (real["Fuel Type"] == fuel) & (real["Engine (cc)"] == cc)],
        real[(real["Model_cleaned"] == model_key) & (real["Fuel Type"] == fuel)],
        real[(real["Model_cleaned"] == model_key)],
        real[(real["Brand"] == brand) & (real["Engine (cc)"] == cc)],
        real[real["Engine (cc)"] == cc],
        real,
    ):
        vals = common_values(sub, "Number of Cylinders", NOISE_MIN_COUNT, NOISE_MIN_SHARE)
        if vals:
            return [int(v) for v in vals]
    return [4]


def mileage_cap(condition: str) -> int:
    """Highest sensible mileage for a condition (99th percentile, rounded up to 1,000 km)."""
    m = df.loc[df["Condition"] == condition, "Mileage_numeric"]
    return int(max(1_000, np.ceil(m.quantile(0.99) / 1_000) * 1_000))


def mode_or_first(series: pd.Series, allowed: list):
    """Most common value of `series` that is in `allowed` (default selection)."""
    vc = series[series.isin(allowed)].value_counts()
    return vc.index[0] if not vc.empty else allowed[0]


def pick_index(options: list, preferred) -> int:
    return options.index(preferred) if preferred in options else 0



def year_input(default: int, key: str) -> int:
    """Slider spanning fixed range 1980 to 2026."""
    min_year = 1980
    max_year = 2026
    clipped_default = int(np.clip(default, min_year, max_year))
    return st.slider(
        "Year of Manufacture",
        min_value=min_year,
        max_value=max_year,
        value=clipped_default,
        step=1,
        key=key,
    )


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.title("Sri Lanka Car Price Dashboard")
tab_predict, tab_eda = st.tabs(["Price Prediction", "Exploratory Data Analysis"])

# ============================== PREDICTION TAB ==============================
with tab_predict:
    st.subheader("Estimate a car's market price")
    st.caption(
        "Every option below is built from real listings. Choices narrow automatically as you "
        "pick a brand and model, so only valid combinations can be selected."
    )

    col1, col2, col3 = st.columns(3)

    # ---- Column 1: identity -------------------------------------------------
    with col1:
        brand = st.selectbox("Brand", sorted(df["Brand"].unique()), key="p_brand")
        brand_df = df[df["Brand"] == brand]

        model_keys = sorted(brand_df["Model_cleaned"].unique(), key=lambda k: MODEL_LABELS[k].lower())
        model_key = st.selectbox(
            "Model", model_keys, format_func=lambda k: MODEL_LABELS[k], key=f"p_model_{brand}"
        )
        model_df = brand_df[brand_df["Model_cleaned"] == model_key]

        conditions = common_values(model_df, "Condition")
        condition = st.selectbox(
            "Condition", conditions,
            index=pick_index(conditions, mode_or_first(model_df["Condition"], conditions)),
            key=f"p_cond_{model_key}",
        )
        cond_df = model_df[model_df["Condition"] == condition]

    # ---- Column 3 first (fuel drives engine options in column 2) -----------
    with col3:
        fuels = common_values(model_df, "Fuel Type")
        fuel_type = st.selectbox(
            "Fuel Type", fuels,
            index=pick_index(fuels, mode_or_first(model_df["Fuel Type"], fuels)),
            key=f"p_fuel_{model_key}",
        )
        fuel_df = model_df[model_df["Fuel Type"] == fuel_type]

        gears = common_values(fuel_df, "Gear")
        gear = st.selectbox(
            "Gear", gears,
            index=pick_index(gears, mode_or_first(fuel_df["Gear"], gears)),
            key=f"p_gear_{model_key}_{fuel_type}",
        )

        bodies = common_values(model_df, "Body Type")
        body_type = st.selectbox(
            "Body Type", bodies,
            index=pick_index(bodies, mode_or_first(model_df["Body Type"], bodies)),
            key=f"p_body_{model_key}",
        )

    # ---- Column 2: numbers --------------------------------------------------
    with col2:
        yom_lo = int(cond_df["YOM"].min())
        yom_hi = int(min(cond_df["YOM"].max(), CURRENT_YEAR))
        yom = year_input(
            int(cond_df["YOM"].median()), key=f"p_yom_{model_key}_{condition}"
        )

        cap = mileage_cap(condition)
        step = 10 if cap <= 1_000 else 500 if cap <= 100_000 else 1_000
        default_km = int(np.clip(round(cond_df["Mileage_numeric"].median() / step) * step, 0, cap))
        mileage = st.number_input(
            f"Mileage (km) — 0 to {cap:,}", min_value=0, max_value=cap,
            value=default_km, step=step, key=f"p_km_{model_key}_{condition}",
        )

        cc_options = valid_engine_sizes(brand, model_key, fuel_type)
        default_cc = mode_or_first(
            fuel_df.loc[fuel_df["Engine (cc)"].isin(cc_options), "Engine (cc)"].astype(int), cc_options
        )
        engine_cc = st.selectbox(
            "Engine Size (cc)", cc_options,
            index=pick_index(cc_options, default_cc),
            format_func=lambda x: f"{x:,} cc",
            key=f"p_cc_{model_key}_{fuel_type}",
        )

        cyl_options = valid_cylinders(brand, model_key, fuel_type, engine_cc)
        cylinders = st.selectbox(
            "Number of Cylinders", cyl_options,
            key=f"p_cyl_{model_key}_{fuel_type}_{engine_cc}",
        )

    # ---- Extra features (default = what is typical for this model) ---------
    st.markdown("**Extra features**")
    c1, c2, c3 = st.columns(3)
    power_steering = c1.checkbox(
        "Power Steering", value=bool(model_df["Power Steering"].mode().iloc[0]), key=f"p_ps_{model_key}")
    power_shutters = c2.checkbox(
        "Power Shutters", value=bool(model_df["Power Shutters"].mode().iloc[0]), key=f"p_psh_{model_key}")
    power_mirrors = c3.checkbox(
        "Power Mirrors", value=bool(model_df["Power Mirrors"].mode().iloc[0]), key=f"p_pm_{model_key}")

    st.divider()

    if st.button("Predict Price", type="primary", width="stretch"):
        car_age = CURRENT_YEAR - yom
        input_row = {
            "Car_Age": car_age,
            "Mileage_numeric": mileage,
            "Engine (cc)": engine_cc,
            "Number of Cylinders": cylinders,
            "Power Steering": int(power_steering),
            "Power Shutters": int(power_shutters),
            "Power Mirrors": int(power_mirrors),
            "Brand": brand,
            "Model_cleaned": model_key,
            "Gear": gear,
            "Fuel Type": fuel_type,
            "Condition": condition,
            "Body Type": body_type,
        }
        input_df = pd.DataFrame([input_row], columns=FEATURE_ORDER)

        # Model was trained on log1p(Price) -> back-transform the prediction
        pred_price = float(np.expm1(model.predict(input_df)[0]))

        st.success(f"### Estimated Price: Rs. {pred_price:,.0f}")
        st.caption(f"Car age used: {car_age} years  •  Model: {MODEL_LABELS[model_key]}")
        st.info(
            f"**{brand} {MODEL_LABELS[model_key]}** — based on {len(model_df):,} listings "
            f"({len(cond_df):,} in “{condition}” condition). "
            f"Seen model years {yom_lo}–{yom_hi}; fuel: {', '.join(fuels)}; body: {', '.join(bodies)}."
        )

        # Sanity check against similar real listings
        similar = cond_df[(cond_df["YOM"] - yom).abs() <= 2]
        basis = "same model, condition, ±2 years"
        if len(similar) < 3:
            similar, basis = cond_df, "same model and condition"
        q1, med, q3 = similar["Price_numeric"].quantile([0.25, 0.5, 0.75])

        k1, k2, k3 = st.columns(3)
        k1.metric("Similar listings", f"{len(similar):,}", help=basis)
        k2.metric("Their median price", f"Rs. {med:,.0f}")
        k3.metric("Their middle 50% range", f"{q1/1e6:.1f}M – {q3/1e6:.1f}M")
        if not (q1 * 0.8 <= pred_price <= q3 * 1.2):
            st.warning("The estimate is well outside the price range of similar listings — treat it with caution.")

# ================================== EDA TAB ==================================
with tab_eda:
    st.subheader("Explore the cleaned dataset")

    # ----------------------------- Filters ---------------------------------
    def reset_filters():
        for k in [k for k in st.session_state if k.startswith("f_")]:
            del st.session_state[k]

    with st.expander("Filters (they apply to every chart below)", expanded=True):
        r1 = st.columns(4)
        f_brand = r1[0].multiselect("Brand", sorted(df["Brand"].unique()), key="f_brand")

        model_pool = df[df["Brand"].isin(f_brand)] if f_brand else df
        f_model = r1[1].multiselect(
            "Model", sorted(model_pool["Model_cleaned"].unique(), key=lambda k: MODEL_LABELS[k].lower()),
            format_func=lambda k: MODEL_LABELS[k], key=f"f_model_{'|'.join(sorted(f_brand))}",
        )
        f_cond = r1[2].multiselect("Condition", sorted(df["Condition"].unique()), key="f_cond")
        f_fuel = r1[3].multiselect("Fuel Type", sorted(df["Fuel Type"].unique()), key="f_fuel")

        r2 = st.columns(4)
        f_gear = r2[0].multiselect("Gear", sorted(df["Gear"].unique()), key="f_gear")
        f_body = r2[1].multiselect("Body Type", sorted(df["Body Type"].unique()), key="f_body")
        f_country = r2[2].multiselect("Manufacturing Country", sorted(df["Manufacturing Country"].unique()), key="f_country")
        f_loc = r2[3].multiselect("Location", sorted(df["Location"].unique()), key="f_loc")

        r3 = st.columns(4)
        p_lo, p_hi = float(np.floor(df["Price_M"].min() * 10) / 10), float(np.ceil(df["Price_M"].max() * 10) / 10)
        f_price = r3[0].slider("Price (Rs. millions)", p_lo, p_hi, (p_lo, p_hi), step=0.1, key="f_price")
        y_lo, y_hi = int(df["YOM"].min()), int(df["YOM"].max())
        f_yom = r3[1].slider("Year of Manufacture", y_lo, y_hi, (y_lo, y_hi), key="f_yom")
        m_hi = int(np.ceil(df["Mileage_numeric"].max() / 5_000) * 5_000)
        f_km = r3[2].slider("Mileage (km)", 0, m_hi, (0, m_hi), step=5_000, key="f_km")
        c_lo = int(np.floor(df["Engine (cc)"].min() / 50) * 50)
        c_hi = int(np.ceil(df["Engine (cc)"].max() / 50) * 50)
        f_cc = r3[3].slider("Engine (cc)", c_lo, c_hi, (c_lo, c_hi), step=50, key="f_cc")

        r4 = st.columns([2, 2, 1])
        f_cyl = r4[0].multiselect(
            "Cylinders", sorted(df["Number of Cylinders"].unique()),
            format_func=lambda x: str(int(x)), key="f_cyl")
        f_power = r4[1].multiselect(
            "Must have", ["Power Steering", "Power Shutters", "Power Mirrors"], key="f_power")
        r4[2].write("")
        r4[2].button("↺ Reset filters", on_click=reset_filters, width="stretch")

    eda_df = df
    for col, sel in [("Brand", f_brand), ("Model_cleaned", f_model), ("Condition", f_cond),
                     ("Fuel Type", f_fuel), ("Gear", f_gear), ("Body Type", f_body),
                     ("Manufacturing Country", f_country), ("Location", f_loc),
                     ("Number of Cylinders", f_cyl)]:
        if sel:
            eda_df = eda_df[eda_df[col].isin(sel)]
    eda_df = eda_df[
        eda_df["Price_M"].between(*f_price) & eda_df["YOM"].between(*f_yom)
        & eda_df["Mileage_numeric"].between(*f_km) & eda_df["Engine (cc)"].between(*f_cc)
    ]
    for feat in f_power:
        eda_df = eda_df[eda_df[feat] == 1]

    if eda_df.empty:
        st.warning("No listings match these filters. Loosen a filter or press **Reset filters**.")
    else:
        k = st.columns(5)
        k[0].metric("Listings", f"{len(eda_df):,}", f"{len(eda_df)/len(df):.0%} of data", delta_color="off")
        k[1].metric("Median price", f"Rs. {eda_df['Price_numeric'].median():,.0f}")
        k[2].metric("Average price", f"Rs. {eda_df['Price_numeric'].mean():,.0f}")
        k[3].metric("Median mileage", f"{eda_df['Mileage_numeric'].median():,.0f} km")
        k[4].metric("Median age", f"{eda_df['Car_Age'].median():.0f} yrs")

        GROUPS = {
            "Brand": "Brand", "Model": "Model_label", "Fuel Type": "Fuel Type", "Condition": "Condition",
            "Gear": "Gear", "Body Type": "Body Type", "Manufacturing Country": "Manufacturing Country",
            "Location": "Location", "Cylinders": "Number of Cylinders",
        }
        AXES = {
            "Price (Rs. M)": "Price_M", "Mileage (km)": "Mileage_numeric", "Car age (yrs)": "Car_Age",
            "Year of manufacture": "YOM", "Engine (cc)": "Engine (cc)", "Cylinders": "Number of Cylinders",
        }
        COLORS = ["None"] + [g for g in GROUPS if g != "Model"]
        HOVER = ["Brand", "Model_label", "YOM", "Mileage_numeric", "Fuel Type", "Condition", "Price_numeric"]

        def color_arg(name):
            return None if name == "None" else GROUPS[name]

        t_over, t_cmp, t_rel, t_dep, t_corr, t_loc, t_data = st.tabs([
            "Overview", "Compare groups", "Scatter explorer", "Depreciation",
            "Correlation", "Market map", "Data",
        ])

        # ---------------------------- Overview ----------------------------
        with t_over:
            a, b, c = st.columns([1, 1, 1])
            hist_color = color_arg(a.selectbox("Colour by", COLORS, key="o_color"))
            bins = b.slider("Bins", 10, 100, 40, key="o_bins")
            log_x = c.checkbox("Log price axis", key="o_log")
            fig = px.histogram(
                eda_df, x="Price_M", nbins=bins, color=hist_color, marginal="box", log_x=log_x,
                barmode="overlay", opacity=0.75, labels={"Price_M": "Price (Rs. millions)"},
                title="Price distribution",
            )
            st.plotly_chart(fig, width="stretch")

            comp = st.selectbox("Show composition of", [g for g in GROUPS if g != "Model"], key="o_comp")
            comp_df = eda_df[GROUPS[comp]].value_counts().reset_index()
            comp_df.columns = [comp, "Listings"]
            left, right = st.columns(2)
            left.plotly_chart(px.pie(comp_df.head(12), names=comp, values="Listings", hole=0.45,
                                     title=f"Listings by {comp} (top 12)"), width="stretch")
            right.plotly_chart(px.bar(comp_df.head(12), x=comp, y="Listings",
                                      title=f"Listings by {comp}"), width="stretch")

        # -------------------------- Compare groups -------------------------
        with t_cmp:
            a, b, c, d = st.columns(4)
            grp = a.selectbox("Group by", list(GROUPS), key="c_grp")
            metric = b.selectbox("Metric", ["Median price", "Average price", "Listings",
                                            "Median mileage", "Median age"], key="c_metric")
            top_n = c.slider("Show top N", 3, 30, 10, key="c_top")
            min_n = d.slider("Min listings per group", 1, 50, 5, key="c_min")

            gcol = GROUPS[grp]
            agg = (eda_df.groupby(gcol)
                   .agg(Listings=("Price_numeric", "size"), **{
                       "Median price": ("Price_M", "median"), "Average price": ("Price_M", "mean"),
                       "Median mileage": ("Mileage_numeric", "median"), "Median age": ("Car_Age", "median")})
                   .query("Listings >= @min_n").reset_index())
            if agg.empty:
                st.info("No group has enough listings — lower the minimum.")
            else:
                agg = agg.sort_values(metric, ascending=False).head(top_n)
                fig = px.bar(agg, x=metric, y=gcol, orientation="h", text_auto=".2f" if "price" in metric else ".0f",
                             hover_data=["Listings"], title=f"{metric} by {grp}"
                             + (" (Rs. millions)" if "price" in metric else ""))
                fig.update_layout(yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, width="stretch")

                shape = st.radio("Price spread", ["Box", "Violin"], horizontal=True, key="c_shape")
                sub = eda_df[eda_df[gcol].isin(agg[gcol])]
                plot = px.box if shape == "Box" else px.violin
                st.plotly_chart(plot(sub, x=gcol, y="Price_M", color=gcol,
                                     labels={"Price_M": "Price (Rs. millions)"},
                                     category_orders={gcol: list(agg[gcol])}), width="stretch")

        # ------------------------- Scatter explorer ------------------------
        with t_rel:
            a, b, c, d = st.columns(4)
            xa = a.selectbox("X axis", list(AXES), index=1, key="s_x")
            ya = b.selectbox("Y axis", list(AXES), index=0, key="s_y")
            col_by = color_arg(c.selectbox("Colour by", COLORS, index=2, key="s_color"))
            n_pts = d.slider("Max points", 200, max(300, len(eda_df)), min(2_000, len(eda_df)), key="s_n")
            l1, l2 = st.columns(2)
            log_sx = l1.checkbox("Log X", key="s_logx")
            log_sy = l2.checkbox("Log Y", key="s_logy")
            samp = eda_df.sample(min(n_pts, len(eda_df)), random_state=1)
            fig = px.scatter(samp, x=AXES[xa], y=AXES[ya], color=col_by, opacity=0.6,
                             hover_data=HOVER, log_x=log_sx, log_y=log_sy,
                             labels={AXES[xa]: xa, AXES[ya]: ya}, title=f"{ya} vs {xa}")
            st.plotly_chart(fig, width="stretch")
            st.caption("Drag to zoom, double-click to reset, click legend entries to hide groups.")

        # --------------------------- Depreciation --------------------------
        with t_dep:
            a, b, c = st.columns(3)
            dep_by = a.selectbox("Separate lines by", ["None"] + [g for g in GROUPS if g not in ("Model", "Location", "Cylinders")],
                                 index=1, key="d_by")
            stat = b.radio("Statistic", ["Median", "Mean"], horizontal=True, key="d_stat")
            min_pts = c.slider("Min listings per point", 1, 20, 3, key="d_min")
            keys = ["Car_Age"] + ([] if dep_by == "None" else [GROUPS[dep_by]])
            dep = (eda_df.groupby(keys)["Price_M"].agg(["median" if stat == "Median" else "mean", "size"])
                   .rename(columns={"median": "Price", "mean": "Price", "size": "Listings"}).reset_index())
            dep = dep[dep["Listings"] >= min_pts]
            if dep.empty:
                st.info("Not enough data per point — lower the minimum.")
            else:
                fig = px.line(dep, x="Car_Age", y="Price", color=None if dep_by == "None" else GROUPS[dep_by],
                              markers=True, hover_data=["Listings"],
                              labels={"Price": f"{stat} price (Rs. M)", "Car_Age": "Car age (years)"},
                              title="How price falls with age")
                st.plotly_chart(fig, width="stretch")

        # ---------------------------- Correlation --------------------------
        with t_corr:
            num_cols = ["YOM", "Car_Age", "Mileage_numeric", "Engine (cc)", "Number of Cylinders",
                        "Price_numeric", "Power Steering", "Power Shutters", "Power Mirrors"]
            chosen = st.multiselect("Variables", num_cols, default=num_cols[:6], key="corr_cols")
            method = st.radio("Method", ["pearson", "spearman"], horizontal=True, key="corr_m")
            if len(chosen) >= 2:
                corr = eda_df[chosen].corr(method=method)
                st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                          zmin=-1, zmax=1, aspect="auto"), width="stretch")
            else:
                st.info("Pick at least two variables.")

        # ---------------------------- Market map ---------------------------
        with t_loc:
            a, b = st.columns(2)
            geo = a.selectbox("Region view", ["Location", "Manufacturing Country"], key="g_by")
            gtop = b.slider("Show top N by listings", 5, 30, 15, key="g_top")
            g = (eda_df.groupby(geo).agg(Listings=("Price_M", "size"), Median_price=("Price_M", "median"))
                 .reset_index().sort_values("Listings", ascending=False).head(gtop))
            fig = px.bar(g, x=geo, y="Listings", color="Median_price", color_continuous_scale="Viridis",
                         labels={"Median_price": "Median price (Rs. M)"},
                         title=f"Listings by {geo} (colour = median price)")
            st.plotly_chart(fig, width="stretch")

        # ------------------------------- Data ------------------------------
        with t_data:
            show_cols = ["Brand", "Model_label", "YOM", "Condition", "Fuel Type", "Gear", "Body Type",
                         "Engine (cc)", "Number of Cylinders", "Mileage_numeric", "Location",
                         "Manufacturing Country", "Price_numeric"]
            st.dataframe(eda_df[show_cols].sort_values("Price_numeric", ascending=False),
                         width="stretch", hide_index=True)
            st.download_button("⬇ Download filtered data (CSV)",
                               eda_df[show_cols].to_csv(index=False).encode("utf-8"),
                               "filtered_cars.csv", "text/csv")