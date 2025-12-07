import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------
st.set_page_config(
    page_title="Global Economic Intelligence Dashboard",
    page_icon="🌍",
    layout="wide"
)

# -----------------------------------------------------------
# CUSTOM MULTICOLOR NEON CSS
# -----------------------------------------------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    font-family: 'Segoe UI', sans-serif;
}
.big-title {
    font-size: 50px;
    font-weight: 900;
    text-align: center;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #ff00ff, #00eaff, #00ff80);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.kpi-card {
    padding: 20px;
    border-radius: 15px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.25);
    backdrop-filter: blur(10px);
    color: white;
    text-align: center;
    box-shadow: 0 0 25px rgba(0,255,255,0.4);
    transition: 0.3s;
}
.kpi-card:hover {
    transform: scale(1.05);
    box-shadow: 0 0 35px rgba(255,0,255,0.7);
}
/* Style for the Clear Button to make it pop */
div.stButton > button:first-child {
    background-color: #ff4b4b;
    color: white;
    border-radius: 10px;
    border: none;
    font-weight: bold;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='big-title'>🌍 Global Economic Intelligence Dashboard</div>", unsafe_allow_html=True)

# -----------------------------------------------------------
# UPLOAD CSV/EXCEL
# -----------------------------------------------------------
file = st.file_uploader("📁 Upload CSV or Excel", type=["csv", "xlsx"])

if file is not None:
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = [c.strip() for c in df.columns]  # clean column names

    # -----------------------------------------------------------
    # SESSION STATE & RESET LOGIC
    # -----------------------------------------------------------
    # Define the Reset Function
    def reset_filters():
        st.session_state.country_key = []
        st.session_state.region_key = []
        st.session_state.year_key = []
        st.session_state.gdp_key = (float(df["GDP (Billion USD)"].min()), float(df["GDP (Billion USD)"].max()))
        st.session_state.growth_key = (float(df["Economic Growth (%)"].min()), float(df["Economic Growth (%)"].max()))
        st.session_state.inflation_key = (float(df["Inflation (%)"].min()), float(df["Inflation (%)"].max()))
        st.session_state.unemp_key = (float(df["Unemployment Rate (%)"].min()), float(df["Unemployment Rate (%)"].max()))
        st.session_state.pop_key = (float(df["Population (Millions)"].min()), float(df["Population (Millions)"].max()))
        st.session_state.exp_key = (float(df["Export Revenue (Billion USD)"].min()), float(df["Export Revenue (Billion USD)"].max()))
        st.session_state.imp_key = (float(df["Import Cost (Billion USD)"].min()), float(df["Import Cost (Billion USD)"].max()))
        st.session_state.trade_key = (float(df["Trade Balance (Billion USD)"].min()), float(df["Trade Balance (Billion USD)"].max()))
        st.session_state.dai_key = (float(df["Digital Adoption Index (0-1)"].min()), float(df["Digital Adoption Index (0-1)"].max()))
        st.session_state.carbon_key = (float(df["Carbon Emission (MT)"].min()), float(df["Carbon Emission (MT)"].max()))

    # -----------------------------------------------------------
    # LEFT FILTER PANEL
    # -----------------------------------------------------------
    with st.sidebar:
        st.subheader("🎛 Filters")
        
        # We add 'key' to every widget so we can reset them programmatically
        country = st.multiselect("Country", df["Country"].unique(), key="country_key")
        region = st.multiselect("Region", df["Region"].unique(), key="region_key")
        year = st.multiselect("Year", df["Year"].unique(), key="year_key")

        gdp = st.slider("GDP (Billion USD)",
                        float(df["GDP (Billion USD)"].min()), float(df["GDP (Billion USD)"].max()),
                        (float(df["GDP (Billion USD)"].min()), float(df["GDP (Billion USD)"].max())), key="gdp_key")
        growth = st.slider("Economic Growth (%)",
                           float(df["Economic Growth (%)"].min()), float(df["Economic Growth (%)"].max()),
                           (float(df["Economic Growth (%)"].min()), float(df["Economic Growth (%)"].max())), key="growth_key")
        inflation = st.slider("Inflation (%)",
                              float(df["Inflation (%)"].min()), float(df["Inflation (%)"].max()),
                              (float(df["Inflation (%)"].min()), float(df["Inflation (%)"].max())), key="inflation_key")
        unemployment = st.slider("Unemployment Rate (%)",
                                 float(df["Unemployment Rate (%)"].min()), float(df["Unemployment Rate (%)"].max()),
                                 (float(df["Unemployment Rate (%)"].min()), float(df["Unemployment Rate (%)"].max())), key="unemp_key")
        population = st.slider("Population (Millions)",
                               float(df["Population (Millions)"].min()), float(df["Population (Millions)"].max()),
                               (float(df["Population (Millions)"].min()), float(df["Population (Millions)"].max())), key="pop_key")
        export = st.slider("Export Revenue (Billion USD)",
                           float(df["Export Revenue (Billion USD)"].min()), float(df["Export Revenue (Billion USD)"].max()),
                           (float(df["Export Revenue (Billion USD)"].min()), float(df["Export Revenue (Billion USD)"].max())), key="exp_key")
        imp = st.slider("Import Cost (Billion USD)",
                        float(df["Import Cost (Billion USD)"].min()), float(df["Import Cost (Billion USD)"].max()),
                        (float(df["Import Cost (Billion USD)"].min()), float(df["Import Cost (Billion USD)"].max())), key="imp_key")
        trade = st.slider("Trade Balance (Billion USD)",
                          float(df["Trade Balance (Billion USD)"].min()), float(df["Trade Balance (Billion USD)"].max()),
                          (float(df["Trade Balance (Billion USD)"].min()), float(df["Trade Balance (Billion USD)"].max())), key="trade_key")
        dai = st.slider("Digital Adoption Index (0-1)",
                        float(df["Digital Adoption Index (0-1)"].min()), float(df["Digital Adoption Index (0-1)"].max()),
                        (float(df["Digital Adoption Index (0-1)"].min()), float(df["Digital Adoption Index (0-1)"].max())), key="dai_key")
        carbon = st.slider("Carbon Emission (MT)",
                           float(df["Carbon Emission (MT)"].min()), float(df["Carbon Emission (MT)"].max()),
                           (float(df["Carbon Emission (MT)"].min()), float(df["Carbon Emission (MT)"].max())), key="carbon_key")
        
        st.markdown("---")
        # CLEAR ALL FILTERS BUTTON
        st.button("❌ Clear All Filters", on_click=reset_filters)

    # -----------------------------------------------------------
    # FILTERING LOGIC
    # -----------------------------------------------------------
    fdf = df.copy()
    if country: fdf = fdf[fdf["Country"].isin(country)]
    if region: fdf = fdf[fdf["Region"].isin(region)]
    if year: fdf = fdf[fdf["Year"].isin(year)]

    fdf = fdf[
        (fdf["GDP (Billion USD)"].between(gdp[0], gdp[1])) &
        (fdf["Economic Growth (%)"].between(growth[0], growth[1])) &
        (fdf["Inflation (%)"].between(inflation[0], inflation[1])) &
        (fdf["Unemployment Rate (%)"].between(unemployment[0], unemployment[1])) &
        (fdf["Population (Millions)"].between(population[0], population[1])) &
        (fdf["Export Revenue (Billion USD)"].between(export[0], export[1])) &
        (fdf["Import Cost (Billion USD)"].between(imp[0], imp[1])) &
        (fdf["Trade Balance (Billion USD)"].between(trade[0], trade[1])) &
        (fdf["Digital Adoption Index (0-1)"].between(dai[0], dai[1])) &
        (fdf["Carbon Emission (MT)"].between(carbon[0], carbon[1]))
    ]

    # -----------------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------------
    st.subheader("📊 Key Performance Indicators (KPIs)")
    kpi_cols = st.columns(4)
    kpi_cols[0].markdown(f"<div class='kpi-card'><h3>Countries</h3><h2>{fdf['Country'].nunique()}</h2></div>", unsafe_allow_html=True)
    kpi_cols[1].markdown(f"<div class='kpi-card'><h3>Total GDP</h3><h2>{fdf['GDP (Billion USD)'].sum():,.0f} B</h2></div>", unsafe_allow_html=True)
    kpi_cols[2].markdown(f"<div class='kpi-card'><h3>Avg Growth</h3><h2>{fdf['Economic Growth (%)'].mean():.2f}%</h2></div>", unsafe_allow_html=True)
    kpi_cols[3].markdown(f"<div class='kpi-card'><h3>Population</h3><h2>{fdf['Population (Millions)'].sum():,.0f} M</h2></div>", unsafe_allow_html=True)

    # -----------------------------------------------------------
    # VISUALIZATIONS
    # -----------------------------------------------------------
    st.subheader("📈 Charts & Analytics")

    col1, col2 = st.columns(2)

    # LEFT COLUMN
    with col1:
        # 1. Line Chart
        fig_line = px.line(fdf, x="Country", y="Economic Growth (%)", color="Region", markers=True)
        st.plotly_chart(fig_line, use_container_width=True)

        # 2. Clustered Bar Chart
        fig_cluster = px.bar(fdf, x="Country", y=["GDP (Billion USD)", "Export Revenue (Billion USD)"],
                             barmode="group", title="Clustered Bar Chart")
        st.plotly_chart(fig_cluster, use_container_width=True)

        # 3. Stacked Column Chart
        fig_stack = px.bar(fdf, x="Region", y=["GDP (Billion USD)", "Export Revenue (Billion USD)"],
                           barmode="stack", title="Stacked Column Chart")
        st.plotly_chart(fig_stack, use_container_width=True)

        # 4. Pie Chart
        fig_pie = px.pie(fdf, names="Region", values="GDP (Billion USD)", title="Pie Chart")
        st.plotly_chart(fig_pie, use_container_width=True)

        # 5. Combo Chart
        st.markdown("### 🧬 Combo Chart: GDP vs Growth")
        combo_df = fdf.groupby("Country")[["GDP (Billion USD)", "Economic Growth (%)"]].mean().reset_index().head(10)
        
        fig_combo = make_subplots(specs=[[{"secondary_y": True}]])
        fig_combo.add_trace(
            go.Bar(x=combo_df["Country"], y=combo_df["GDP (Billion USD)"], name="GDP (Bar)", marker_color="#00eaff"),
            secondary_y=False
        )
        fig_combo.add_trace(
            go.Scatter(x=combo_df["Country"], y=combo_df["Economic Growth (%)"], name="Growth % (Line)", mode="lines+markers", line=dict(color="#ff00ff", width=3)),
            secondary_y=True
        )
        fig_combo.update_layout(title_text="GDP vs Economic Growth (Top 10)")
        st.plotly_chart(fig_combo, use_container_width=True)

        # 6. Waterfall Chart
        st.markdown("### 🌊 Waterfall Chart: Trade Balance")
        wf_export = fdf["Export Revenue (Billion USD)"].sum()
        wf_import = fdf["Import Cost (Billion USD)"].sum() * -1 
        wf_balance = fdf["Trade Balance (Billion USD)"].sum()
        
        fig_waterfall = go.Figure(go.Waterfall(
            name = "Trade Balance", orientation = "v",
            measure = ["relative", "relative", "total"],
            x = ["Total Exports", "Total Imports", "Net Trade Balance"],
            textposition = "outside",
            y = [wf_export, wf_import, wf_balance],
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        fig_waterfall.update_layout(title="Global Trade Balance Waterfall")
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # --- NEW VISUAL FOR LEFT COLUMN (Filling the vacancy) ---
        # 7. SUNBURST CHART
        st.markdown("### ☀️ Sunburst Chart: Regional Breakdown")
        fig_sun = px.sunburst(fdf, path=['Region', 'Country'], values='Population (Millions)',
                              color='Economic Growth (%)', color_continuous_scale='RdBu',
                              title="Population & Growth Hierarchy")
        st.plotly_chart(fig_sun, use_container_width=True)

    # RIGHT COLUMN
    with col2:
        # 1. Scatter Chart
        fig_scatter = px.scatter(fdf, x="GDP (Billion USD)", y="Population (Millions)",
                                 size="Economic Growth (%)", color="Region", title="Scatter Chart")
        st.plotly_chart(fig_scatter, use_container_width=True)

        # 2. Area Chart
        fig_area = px.area(fdf, x="Year", y="GDP (Billion USD)", color="Region", title="Area Chart")
        st.plotly_chart(fig_area, use_container_width=True)

        # 3. Donut Chart
        fig_donut = go.Figure(data=[go.Pie(labels=fdf['Region'], values=fdf['GDP (Billion USD)'], hole=.5)])
        fig_donut.update_layout(title="Regional GDP Distribution (Donut)")
        st.plotly_chart(fig_donut, use_container_width=True)

        # 4. Choropleth Map
        fig_map = px.choropleth(fdf, locations="Country", locationmode="country names",
                                color="GDP (Billion USD)", hover_name="Country",
                                color_continuous_scale="Plasma", title="GDP Heatmap")
        st.plotly_chart(fig_map, use_container_width=True)

        # 5. Tree Map
        st.markdown("### 🌳 Treemap: Region Hierarchy")
        fig_treemap = px.treemap(fdf, path=[px.Constant("World"), 'Region', 'Country'], values='GDP (Billion USD)',
                                 color='Economic Growth (%)', color_continuous_scale='RdBu',
                                 title="Global GDP Hierarchy")
        st.plotly_chart(fig_treemap, use_container_width=True)

        # 6. Funnel Chart
        st.markdown("### 🌪️ Funnel Chart: GDP Ranking")
        funnel_df = fdf.sort_values(by="GDP (Billion USD)", ascending=False).head(10)
        fig_funnel = px.funnel(funnel_df, x='GDP (Billion USD)', y='Country', title="Top 10 Economies Funnel")
        st.plotly_chart(fig_funnel, use_container_width=True)

        # 7. Ribbon Chart
        st.markdown("### 🎀 Ribbon Chart (Trend Flow)")
        fig_ribbon = px.area(fdf, x="Year", y="Export Revenue (Billion USD)", color="Region", 
                             line_group="Country", title="Export Trends Ribbon",
                             color_discrete_sequence=px.colors.qualitative.Bold)
        fig_ribbon.update_traces(line_shape='spline')
        st.plotly_chart(fig_ribbon, use_container_width=True)

    # Table / Matrix
    st.subheader("📋 Data Table")
    st.dataframe(fdf, use_container_width=True)

    # -----------------------------------------------------------
    # DOWNLOAD FILTERED DATA
    # -----------------------------------------------------------
    st.subheader("⬇ Download Filtered Data")
    st.download_button("Download CSV", fdf.to_csv(index=False).encode("utf-8"), "filtered_data.csv")

else:
    st.info("Upload your Excel/CSV file to begin.")
