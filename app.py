"""
Sri Lanka Car Price Dashboard
-----------------------------
- Tab 1: Predict a car's price using the trained Random Forest pipeline (Model.pkl)
- Tab 2: Exploratory Data Analysis on formatted_data.csv

Run with:  streamlit run app.py
Place this file in the same folder as Model.pkl and formatted_data.csv
(i.e. your "Sri-Lanka-Car-Market..." project root).
"""

import pickle
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

st.set_page_config(page_title="Sri Lanka Car Price Dashboard", layout="wide")


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
    """Reproduce the same cleaning/feature-engineering steps used to train the model,
    so dropdown options and EDA reflect exactly what the model was fit on."""
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

    df["Car_Age"] = CURRENT_YEAR - df["YOM"]

    for col in ["Power Steering", "Power Shutters", "Power Mirrors"]:
        df[col] = (
            df[col].astype(str).str.strip().str.lower()
            .map({"yes": 1, "no": 0}).fillna(0)
        )

    return df


try:
    model = load_model(MODEL_PATH)
    df = load_data(DATA_PATH)
except FileNotFoundError as e:
    st.error(
        f"Couldn't find a required file: {e}\n\n"
        f"Make sure `Model.pkl` and `formatted_data.csv` sit in the same folder as this script."
    )
    st.stop()


# --------------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------------
st.title("🚗 Sri Lanka Car Price Dashboard")

tab_predict, tab_eda = st.tabs(["💰 Price Prediction", "📊 Exploratory Data Analysis"])

# ============================== PREDICTION TAB ==============================
with tab_predict:
    st.subheader("Estimate a car's market price")

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", sorted(df["Brand"].dropna().unique()))
        models_for_brand = sorted(
            df.loc[df["Brand"] == brand, "Model_cleaned"].dropna().unique()
        )
        model_cleaned = st.selectbox("Model", models_for_brand)
        condition = st.selectbox("Condition", sorted(df["Condition"].dropna().unique()))

    with col2:
        yom = st.number_input(
            "Year of Manufacture", min_value=1980, max_value=CURRENT_YEAR, value=2018, step=1
        )
        mileage = st.number_input(
            "Mileage (km)", min_value=0, value=60_000, step=1_000
        )
        engine_cc = st.number_input(
            "Engine Size (cc)", min_value=100, max_value=9_999, value=1_500, step=50
        )

    with col3:
        cyl_options = sorted(df["Number of Cylinders"].dropna().unique())
        cylinders = st.selectbox(
            "Number of Cylinders", cyl_options,
            format_func=lambda x: str(int(x)) if float(x).is_integer() else str(x),
        )
        gear = st.selectbox("Gear", sorted(df["Gear"].dropna().unique()))
        fuel_type = st.selectbox("Fuel Type", sorted(df["Fuel Type"].dropna().unique()))
        body_type = st.selectbox("Body Type", sorted(df["Body Type"].dropna().unique()))

    st.markdown("**Extra features**")
    c1, c2, c3 = st.columns(3)
    power_steering = c1.checkbox("Power Steering", value=True)
    power_shutters = c2.checkbox("Power Shutters", value=True)
    power_mirrors = c3.checkbox("Power Mirrors", value=True)

    st.divider()

    if st.button("Predict Price", type="primary", use_container_width=True):
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
            "Model_cleaned": model_cleaned,
            "Gear": gear,
            "Fuel Type": fuel_type,
            "Condition": condition,
            "Body Type": body_type,
        }
        input_df = pd.DataFrame([input_row], columns=FEATURE_ORDER)

        # Model was trained on log1p(Price) -> back-transform the prediction
        pred_log_price = model.predict(input_df)[0]
        pred_price = np.expm1(pred_log_price)

        st.success(f"### Estimated Price: Rs. {pred_price:,.0f}")
        st.caption(f"Car age used: {car_age} years  •  Model input: {model_cleaned}")

# ================================== EDA TAB ==================================
with tab_eda:
    st.subheader("Explore the cleaned dataset")

    with st.sidebar:
        st.header("EDA Filters")
        brand_filter = st.multiselect(
            "Brand", sorted(df["Brand"].unique()), default=[]
        )
        condition_filter = st.multiselect(
            "Condition", sorted(df["Condition"].unique()), default=[]
        )

    eda_df = df.copy()
    if brand_filter:
        eda_df = eda_df[eda_df["Brand"].isin(brand_filter)]
    if condition_filter:
        eda_df = eda_df[eda_df["Condition"].isin(condition_filter)]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Listings", f"{len(eda_df):,}")
    m2.metric("Median Price", f"Rs. {eda_df['Price_numeric'].median():,.0f}")
    m3.metric("Median Mileage", f"{eda_df['Mileage_numeric'].median():,.0f} km")
    m4.metric("Median Age", f"{eda_df['Car_Age'].median():.0f} yrs")

    st.markdown("#### Price distribution")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.histplot(eda_df["Price_numeric"], bins=50, kde=True, color="#4C72B0", ax=ax)
    ax.set_xlabel("Price (LKR)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))
    st.pyplot(fig)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Price by Brand (top 10 by listing count)")
        top_brands = eda_df["Brand"].value_counts().head(10).index
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(
            data=eda_df[eda_df["Brand"].isin(top_brands)],
            x="Brand", y="Price_numeric", showfliers=False, ax=ax,
        )
        ax.set_ylabel("Price (LKR)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col_b:
        st.markdown("#### Price by Fuel Type")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.boxplot(data=eda_df, x="Fuel Type", y="Price_numeric", showfliers=False, ax=ax)
        ax.set_ylabel("Price (LKR)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))
        plt.xticks(rotation=20)
        st.pyplot(fig)

    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Mileage vs Price")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(
            data=eda_df.sample(min(2000, len(eda_df)), random_state=1),
            x="Mileage_numeric", y="Price_numeric", alpha=0.4, ax=ax,
        )
        ax.set_xlabel("Mileage (km)")
        ax.set_ylabel("Price (LKR)")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x/1e6:.0f}M"))
        st.pyplot(fig)

    with col_d:
        st.markdown("#### Correlation heatmap")
        numeric_cols = ["YOM", "Mileage_numeric", "Engine (cc)",
                         "Number of Cylinders", "Price_numeric", "Car_Age"]
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(eda_df[numeric_cols].corr(), annot=True, fmt=".2f",
                    cmap="coolwarm", center=0, ax=ax)
        st.pyplot(fig)

    st.markdown("#### Top 15 models by listing count")
    top_models = eda_df["Model_cleaned"].value_counts().head(15)
    fig, ax = plt.subplots(figsize=(10, 4))
    top_models.plot(kind="bar", ax=ax, color="#55A868")
    ax.set_ylabel("Listings")
    plt.xticks(rotation=60)
    st.pyplot(fig)

    with st.expander("View raw filtered data"):
        st.dataframe(eda_df, use_container_width=True)