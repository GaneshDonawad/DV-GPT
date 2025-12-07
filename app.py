import streamlit as st
import pandas as pd
import plotly.express as px

# ----------------------------------------------------
# Page Config
# ----------------------------------------------------
st.set_page_config(
    page_title="🌍 Global Country Dashboard",
    layout="wide",
)

# ----------------------------------------------------
# Upload Dataset (ADDED FEATURE)
# ----------------------------------------------------
st.sidebar.header("📂 Upload Excel File")

uploaded_file = st.sidebar.file_uploader("Upload your Excel dataset", type=["xlsx"])

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
else:
    st.warning("Please upload your Excel file to load the dashboard.")
    st.stop()

# ----------------------------------------------------
# Sidebar
# ----------------------------------------------------
st.sidebar.title("🔍 Dashboard Filters")

selected_countries = st.sidebar.multiselect(
    "Select Countries",
    df["Country"].unique(),
    default=df["Country"].unique()
)

selected_years = st.sidebar.multiselect(
    "Select Years",
    sorted(df["Year"].unique()),
    default=df["Year"].unique()
)

filtered_df = df[
    (df["Country"].isin(selected_countries)) &
    (df["Year"].isin(selected_years))
]

st.sidebar.markdown("----")
st.sidebar.info("Use filters to update all dashboard visuals in real-time.")

# ----------------------------------------------------
# Title
# ----------------------------------------------------
st.title("🌍 Global Country Mixed-Mode Analytics Dashboard")

st.markdown("""
This next-level interactive dashboard gives insights into **GDP**, **Population**,  
**Internet Usage**, **Employment**, **Inflation**, and more for 15 countries across multiple years.
""")

# ----------------------------------------------------
# KPI SECTION
# ----------------------------------------------------
st.subheader("📌 Global KPI Overview")

k1, k2, k3, k4 = st.columns(4)

k1.metric("Total Population", f"{filtered_df['Population (Millions)'].sum():,.1f} M")
k2.metric("Avg GDP", f"${filtered_df['GDP (Billion USD)'].mean():,.1f} B")
k3.metric("Avg Employment Rate", f"{filtered_df['Employment Rate (%)'].mean():.1f}%")
k4.metric("Avg Internet Usage", f"{filtered_df['Internet Usage (%)'].mean():.1f}%")

st.markdown("----")

# ----------------------------------------------------
# GDP Trend
# ----------------------------------------------------
st.subheader("📈 GDP Trend Over Years")

fig_gdp = px.line(
    filtered_df,
    x="Year",
    y="GDP (Billion USD)",
    color="Country",
    markers=True,
    template="plotly_white",
)

st.plotly_chart(fig_gdp, use_container_width=True)

# ----------------------------------------------------
# Population Bar Chart
# ----------------------------------------------------
st.subheader("👥 Population Comparison (Latest Year)")

latest_year = filtered_df["Year"].max()
latest_data = filtered_df[filtered_df["Year"] == latest_year]

fig_pop = px.bar(
    latest_data,
    x="Country",
    y="Population (Millions)",
    color="Population (Millions)",
    text_auto=True,
    template="plotly_white"
)

st.plotly_chart(fig_pop, use_container_width=True)

# ----------------------------------------------------
# Pie Chart - Internet Usage
# ----------------------------------------------------
st.subheader("🌐 Internet Usage Share (Latest Year)")

fig_net = px.pie(
    latest_data,
    names="Country",
    values="Internet Usage (%)",
    hole=0.4,
)

st.plotly_chart(fig_net, use_container_width=True)

# ----------------------------------------------------
# Scatter: Employment vs GDP
# ----------------------------------------------------
st.subheader("💼 Employment Rate vs GDP")

fig_emp = px.scatter(
    filtered_df,
    x="GDP (Billion USD)",
    y="Employment Rate (%)",
    size="Population (Millions)",
    color="Country",
    hover_name="Country",
    template="plotly_white",
)

st.plotly_chart(fig_emp, use_container_width=True)

# ----------------------------------------------------
# Top 10 GDP
# ----------------------------------------------------
st.subheader(f"🏆 Top 10 Countries by GDP ({latest_year})")

top10 = latest_data.nlargest(10, "GDP (Billion USD)")

fig_top = px.bar(
    top10,
    x="Country",
    y="GDP (Billion USD)",
    color="GDP (Billion USD)",
    text_auto=True,
    template="plotly_white",
)

st.plotly_chart(fig_top, use_container_width=True)

# ----------------------------------------------------
# Inflation Trend
# ----------------------------------------------------
st.subheader("📉 Inflation Rate Trend")

fig_inf = px.line(
    filtered_df,
    x="Year",
    y="Economic Inflation (%)",
    color="Country",
    markers=True,
    template="plotly_white",
)

st.plotly_chart(fig_inf, use_container_width=True)

st.markdown("----")

# ----------------------------------------------------
# Footer
# ----------------------------------------------------
st.caption("Built using Streamlit • Designed for DV Project Expo • By Ganesh")
