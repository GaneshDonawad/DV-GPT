import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ----------------------------
# Page config & CSS (polish)
# ----------------------------
st.set_page_config(page_title="📊 Hybrid Interactive Dashboard", layout="wide", page_icon="📈")

st.markdown(
    """
    <style>
    .big-title { font-size:42px; font-weight:800; text-align:center; color:#0b6b4f; margin-bottom:6px; }
    .subtitle { text-align:center; color:#6b6b6b; margin-top:0; margin-bottom:18px; }
    .card { padding:18px; border-radius:12px; background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02)); border:1px solid rgba(255,255,255,0.06); box-shadow:0 6px 18px rgba(0,0,0,0.12);}
    .small { font-size:13px; color:#9aa0a6; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<div class='big-title'>📊 Hybrid Interactive Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Fixed professional filters + dynamic auto filters. Upload Excel/CSV and explore.</div>", unsafe_allow_html=True)

# ----------------------------
# File uploader (Excel or CSV)
# ----------------------------
with st.sidebar.expander("📂 Upload data (Excel / CSV)"):
    uploaded = st.file_uploader("Upload .xlsx or .csv file", type=["xlsx", "xls", "csv"], accept_multiple_files=False)
    sample_button = st.button("Load sample data (small)")

# ----------------------------
# Helper: load data
# ----------------------------
@st.cache_data
def load_dataframe(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.type in ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/vnd.ms-excel"):
            df = pd.read_excel(uploaded_file)
        else:
            # assume csv
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        # try common encodings / engines fallback
        try:
            df = pd.read_csv(uploaded_file, encoding="utf-8")
        except:
            df = pd.read_csv(uploaded_file, encoding="latin1")
    return df

# sample fallback data (very small)
def sample_data():
    countries = ["India","China","United States","Germany","France","Japan","Australia","Canada","Brazil","South Africa","Russia","Mexico","Saudi Arabia","South Korea","United Kingdom"]
    years = list(range(2018,2025))
    rows = []
    for c in countries:
        for y in years:
            rows.append({
                "Country": c,
                "Region": "Asia" if c in ["India","China","Japan","South Korea"] else ("Europe" if c in ["Germany","France","United Kingdom"] else ("North America" if c in ["United States","Canada","Mexico"] else ("South America" if c in ["Brazil"] else ("Africa" if c in ["South Africa"] else ("Oceania" if c in ["Australia"] else "Middle East"))))),
                "Year": y,
                "GDP (Billion USD)": round(np.random.uniform(50,20000),2),
                "Population (Millions)": round(np.random.uniform(3,1400),2),
                "Employment Rate (%)": round(np.random.uniform(40,80),2),
                "Internet Usage (%)": round(np.random.uniform(30,95),2),
                "Economic Inflation (%)": round(np.random.uniform(0.5,12),2),
                "Energy Consumption (GWh)": round(np.random.uniform(1000,2000000),2),
            })
    return pd.DataFrame(rows)

# load frame
if sample_button:
    df = sample_data()
else:
    df = load_dataframe(uploaded)

if df is None:
    st.warning("Upload an Excel/CSV file to start. You can also click 'Load sample data' in the sidebar for a demo.")
    st.stop()

# Normalize column names (strip)
df.columns = [str(c).strip() for c in df.columns]

# ----------------------------
# Auto-detect data types
# ----------------------------
# try to detect a few common fields (Country, Year, Region, GDP, Population)
col_lower = {c.lower(): c for c in df.columns}

def safe_get(col_options):
    for k in col_options:
        if k in col_lower:
            return col_lower[k]
    return None

country_col = safe_get(["country","nation","country name"])
region_col = safe_get(["region","continent"])
year_col = safe_get(["year","yr"])
gdp_col = safe_get(["gdp","gdp (billion usd)","gdp (million current us$)"])
pop_col = safe_get(["population","population (millions)","population (thousands)","pop"])
# if year column exists but is not numeric, try to parse
if year_col and not np.issubdtype(df[year_col].dtype, np.number):
    try:
        df[year_col] = pd.to_numeric(df[year_col], errors='coerce').astype('Int64')
    except:
        pass

# ----------------------------
# Sidebar: HYBRID FILTERS
# ----------------------------
st.sidebar.markdown("## 🔎 Filters (Hybrid)")

# --- Fixed professional filters (5) ---
st.sidebar.markdown("### 🔐 Fixed filters")
fixed_country = st.sidebar.multiselect("Country", options=sorted(df[country_col].dropna().unique()) if country_col else sorted(df.columns[:10].astype(str).unique()), default=(sorted(df[country_col].unique()) if country_col else None))
fixed_year = None
if year_col:
    years_sorted = sorted(df[year_col].dropna().astype(int).unique())
    fixed_year = st.sidebar.multiselect("Year", options=years_sorted, default=years_sorted)
else:
    fixed_year = st.sidebar.multiselect("Year", options=sorted(df.columns[:6].astype(str).unique()), default=None)

fixed_region = st.sidebar.multiselect("Region", options=sorted(df[region_col].dropna().unique()) if region_col else [], default=(sorted(df[region_col].unique()) if region_col else None))

# GDP range filter (if detected)
if gdp_col:
    min_gdp, max_gdp = float(df[gdp_col].min(skipna=True)), float(df[gdp_col].max(skipna=True))
    gdp_range = st.sidebar.slider("GDP Range (min - max)", min_value=min_gdp, max_value=max_gdp, value=(min_gdp, max_gdp))
else:
    gdp_range = None

# Population range filter (if detected)
if pop_col:
    min_pop, max_pop = float(df[pop_col].min(skipna=True)), float(df[pop_col].max(skipna=True))
    pop_range = st.sidebar.slider("Population Range (min - max)", min_value=min_pop, max_value=max_pop, value=(min_pop, max_pop))
else:
    pop_range = None

st.sidebar.markdown("---")

# --- Dynamic auto-filters (generated from remaining columns) ---
st.sidebar.markdown("### ⚙️ Auto-generated filters")
max_auto = st.sidebar.number_input("Max auto filters to show", min_value=5, max_value=30, value=12, step=1, help="Limit number of auto filters shown to keep UI tidy.")

auto_filters = {}
auto_show_count = 0
for col in df.columns:
    if col in [country_col, region_col, year_col, gdp_col, pop_col]:
        continue
    if auto_show_count >= max_auto:
        break

    series = df[col].dropna()
    # if mostly numeric
    if pd.api.types.is_numeric_dtype(series):
        try:
            mn, mx = float(series.min()), float(series.max())
            if mn == mx:
                continue
            val = st.sidebar.slider(f"{col} (range)", min_value=mn, max_value=mx, value=(mn, mx), step=(mx-mn)/100 if (mx-mn)!=0 else 1.0)
            auto_filters[col] = ("range", val)
            auto_show_count += 1
        except Exception:
            continue
    # if datetime-like
    elif pd.api.types.is_datetime64_any_dtype(series) or (series.apply(lambda x: isinstance(x, (datetime, pd.Timestamp))).any()):
        # convert
        try:
            ser_dt = pd.to_datetime(series, errors='coerce').dropna()
            mn_dt, mx_dt = ser_dt.min().date(), ser_dt.max().date()
            val = st.sidebar.date_input(f"{col} (date range)", value=(mn_dt, mx_dt))
            auto_filters[col] = ("date", val)
            auto_show_count += 1
        except Exception:
            continue
    else:
        # categorical/text - if unique values not too many
        uniques = series.astype(str).unique()
        if len(uniques) <= 120:
            sel = st.sidebar.multiselect(f"{col} (select)", options=sorted(uniques), default=None)
            auto_filters[col] = ("multi", sel)
            auto_show_count += 1
        else:
            # show a search box for large cardinality columns
            search_val = st.sidebar.text_input(f"Search {col}", value="")
            if search_val:
                sel = sorted([u for u in uniques if search_val.lower() in str(u).lower()])
                sel = st.sidebar.multiselect(f"{col} (select found)", options=sel, default=None)
                auto_filters[col] = ("multi", sel)
                auto_show_count += 1

st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Use the fixed filters for best results and the auto filters to zoom-in on other columns.")

# ----------------------------
# Apply filters to dataframe
# ----------------------------
def apply_filters(df):
    df2 = df.copy()
    # fixed filters
    if country_col and fixed_country:
        df2 = df2[df2[country_col].isin(fixed_country)]
    if year_col and fixed_year:
        df2 = df2[df2[year_col].isin(fixed_year)]
    if region_col and fixed_region:
        df2 = df2[df2[region_col].isin(fixed_region)]
    if gdp_col and gdp_range is not None:
        df2 = df2[df2[gdp_col].between(gdp_range[0], gdp_range[1])]
    if pop_col and pop_range is not None:
        df2 = df2[df2[pop_col].between(pop_range[0], pop_range[1])]

    # auto filters
    for col, (typ, val) in auto_filters.items():
        if val is None or (typ == "multi" and len(val) == 0):
            continue
        if typ == "range":
            mn, mx = val
            df2 = df2[pd.to_numeric(df2[col], errors='coerce').between(mn, mx, inclusive="both")]
        elif typ == "date":
            d1, d2 = val
            ser = pd.to_datetime(df2[col], errors='coerce').dt.date
            df2 = df2[ser.between(d1, d2)]
        elif typ == "multi":
            # treat as strings
            df2 = df2[df2[col].astype(str).isin([str(x) for x in val])]
    return df2

filtered = apply_filters(df)

# ----------------------------
# Main layout (tabs)
# ----------------------------
tabs = st.tabs(["📁 Data", "📊 Visuals", "🔎 Filters & Download", "📈 Insights"])

# ---------- TAB 1: Data ----------
with tabs[0]:
    st.subheader("📥 Data Preview")
    st.markdown(f"<div class='card'>Showing <b>{filtered.shape[0]}</b> rows and <b>{filtered.shape[1]}</b> columns after filters.</div>", unsafe_allow_html=True)
    st.dataframe(filtered.reset_index(drop=True), use_container_width=True)

# ---------- TAB 2: Visuals ----------
with tabs[1]:
    st.subheader("📊 Visualizations")

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    try:
        k1.metric("Rows (filtered)", f"{filtered.shape[0]:,}")
        if pop_col:
            k2.metric("Total Population", f"{filtered[pop_col].sum():,.2f}")
        else:
            k2.metric("Total Population", "N/A")
        if gdp_col:
            k3.metric("Avg GDP", f"{filtered[gdp_col].mean():,.2f}")
        else:
            k3.metric("Avg GDP", "N/A")
        k4.metric("Columns", f"{filtered.shape[1]}")
    except Exception:
        k1.metric("Rows (filtered)", f"{filtered.shape[0]:,}")
        k2.metric("Total Population", "N/A")
        k3.metric("Avg GDP", "N/A")
        k4.metric("Columns", f"{filtered.shape[1]}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # GDP trend if column exists
    if gdp_col and year_col:
        with col1:
            st.markdown("#### 📈 GDP Trend")
            fig = px.line(filtered, x=year_col, y=gdp_col, color=country_col if country_col else None, markers=True)
            st.plotly_chart(fig, use_container_width=True)
    else:
        with col1:
            st.info("GDP or Year column not detected — upload a dataset with fields named like 'GDP' and 'Year' for trend charts.")

    # Population bar (latest year)
    with col2:
        st.markdown("#### 👥 Population Comparison (latest year)")
        try:
            latest_year_val = int(filtered[year_col].max()) if year_col else None
            latest_df = filtered[filtered[year_col] == latest_year_val] if year_col else filtered
            if pop_col:
                fig2 = px.bar(latest_df.sort_values(pop_col, ascending=False).head(15), x=country_col if country_col else latest_df.columns[0], y=pop_col, text=pop_col)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Population column not detected.")
        except Exception as e:
            st.info("Not enough data for latest-year population chart.")

    st.markdown("---")
    # Scatter: GDP vs Internet usage (if available)
    scatter_left, scatter_right = st.columns(2)
    with scatter_left:
        if gdp_col and ("Internet" in df.columns or any("internet" in c.lower() for c in df.columns)):
            internet_col = next((c for c in df.columns if "internet" in c.lower()), None)
            if internet_col:
                fig3 = px.scatter(filtered, x=gdp_col, y=internet_col, color=country_col, size=pop_col if pop_col else None, hover_name=country_col)
                st.plotly_chart(fig3, use_container_width=True)
    with scatter_right:
        # Correlation heatmap for numeric cols (top 10)
        st.markdown("#### 🔥 Correlation (numeric columns)")
        numeric = filtered.select_dtypes(include=[np.number]).dropna(axis=1, how='all')
        if numeric.shape[1] >= 2:
            corr = numeric.corr()
            # limit to top 12 numeric columns for performance
            cols_to_show = corr.columns[:12]
            fig4 = px.imshow(corr.loc[cols_to_show, cols_to_show], text_auto=True)
            st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("Not enough numeric columns for correlation heatmap.")

# ---------- TAB 3: Filters & Download ----------
with tabs[2]:
    st.subheader("🔎 Current Filters Summary")
    st.write("Fixed filters:")
    st.write({"Country": fixed_country, "Year": fixed_year, "Region": fixed_region, "GDP_range": gdp_range, "Population_range": pop_range})
    st.write("Auto filters (applied):")
    st.write({k:v for k,v in auto_filters.items() if v[1] is not None and (not (isinstance(v[1], list) and len(v[1])==0))})

    st.markdown("---")
    st.subheader("⬇ Download filtered data")
    csv = filtered.to_csv(index=False).encode("utf-8")
    st.download_button("Download CSV (filtered)", csv, file_name="filtered_dataset.csv", mime="text/csv")
    st.markdown("You can also export displayed table manually if needed.")

# ---------- TAB 4: Insights ----------
with tabs[3]:
    st.subheader("📈 Auto Insights")
    try:
        st.write("### Descriptive statistics (numeric columns)")
        st.dataframe(filtered.select_dtypes(include=[np.number]).describe().T, use_container_width=True)
    except:
        st.info("Not enough numeric columns for descriptive stats.")

    st.markdown("### Column overview")
    for c in filtered.columns:
        with st.expander(f"{c} — type: {filtered[c].dtype}"):
            st.write("Missing:", int(filtered[c].isna().sum()))
            if pd.api.types.is_numeric_dtype(filtered[c]):
                st.write("Min / Max / Mean:", float(filtered[c].min(skipna=True)), "/", float(filtered[c].max(skipna=True)), "/", float(filtered[c].mean(skipna=True)))
            else:
                st.write("Unique values (top 15):", filtered[c].dropna().astype(str).unique()[:15])

st.markdown("<div class='small'>Tip: use the 'Max auto filters to show' control in the sidebar to increase or reduce the number of auto filters shown. The app is designed to adapt to any uploaded Excel.</div>", unsafe_allow_html=True)
