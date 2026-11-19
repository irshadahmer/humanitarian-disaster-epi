from pathlib import Path
import geopandas as gpd
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.outbreak_climate import fit_climate_outbreak_glm
from src.spatial_access import EMTSurgeParams, simulate_emt_14day_surge

st.set_page_config(
    page_title="Humanitarian & Disaster Epidemiology Toolkit",
    layout="wide",
)

PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"


@st.cache_data
def load_climate_panel() -> pd.DataFrame:
    return pd.read_parquet(PROCESSED_DIR / "cholera_climate_panel.parquet")


@st.cache_data
def load_mozambique_access() -> pd.DataFrame:
    gdf = gpd.read_file(PROCESSED_DIR / "mozambique_province_access.gpkg")
    return pd.DataFrame(gdf.drop(columns=["geometry"]))


def render_spatial_access_tab(df: pd.DataFrame) -> None:
    st.subheader("Humanitarian Health Access & Sphere Standards Gap Analysis (Mozambique)")
    st.caption("Sources: UN OCHA HDX / Healthsites.io (1,161 georeferenced facilities inside ADM1 boundaries), geoBoundaries ADM1, INE Mozambique")

    total_pop = int(df["population"].sum())
    total_fac = int(df["mapped_facilities"].sum())
    total_gap = int(df["sphere_facility_gap"].sum())
    worst_prov = df.iloc[0]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Mapped Health Facilities", f"{total_fac:,}")
    k2.metric("National Pop / Facility", f"{total_pop / total_fac:,.0f}", "Sphere target: 10,000")
    k3.metric("Total Sphere Facility Deficit", f"{total_gap:,} units")
    k4.metric("Highest-Need Province", worst_prov["shapeName"], f"{worst_prov['sphere_coverage_pct']:.1f}% coverage")

    fig = px.bar(
        df.sort_values("sphere_coverage_pct", ascending=True),
        x="sphere_coverage_pct",
        y="shapeName",
        orientation="h",
        color="sphere_coverage_pct",
        color_continuous_scale="RdYlGn",
        labels={
            "sphere_coverage_pct": "Sphere Standard Coverage (% of 1 Facility / 10,000 Pop)",
            "shapeName": "Province",
        },
        title="Provincial Primary Health Facility Coverage vs. Sphere Minimum Standard (100%)",
        height=460,
    )
    fig.add_vline(x=100, line_dash="dash", line_color="#0f172a", annotation_text="Sphere Minimum (100%)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Provincial Humanitarian Health Infrastructure Table")
    st.dataframe(
        df.rename(
            columns={
                "shapeName": "Province",
                "population": "Population",
                "mapped_facilities": "Mapped Facilities",
                "pop_per_facility": "Pop / Facility",
                "sphere_required_facilities": "Sphere Target Units",
                "sphere_facility_gap": "Unmet Facility Gap",
                "sphere_coverage_pct": "Sphere Coverage (%)",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


def render_climate_outbreak_tab(panel: pd.DataFrame) -> None:
    st.subheader("Climate-Driven Cholera Early-Warning & Outbreak Surveillance")
    st.caption("Sources: WHO Global Health Observatory (CHOLERA_0000000001/02) & Open-Meteo ERA5 Reanalysis (2000-2024)")

    summary = fit_climate_outbreak_glm(panel)
    m1, m2, m3 = st.columns(3)
    m1.metric("Notified Outbreak Corridor-Years", f"{summary['n_outbreak_years']}")
    m2.metric("IRR per +50mm Peak 24h Rainfall", f"{summary['irr_per_50mm_peak_day']:.2f}x")
    m3.metric("IRR per +1 SD Annual Rain", f"{summary['irr_per_sd_precip']:.2f}x", f"p = {summary['pval_precip']:.3f}")

    corridors = sorted(panel["iso3"].unique())
    selected_iso = st.selectbox(
        "Disaster Corridor",
        corridors,
        index=corridors.index("MOZ"),
        format_func=lambda k: f"{k} - {panel[panel['iso3'] == k]['corridor_name'].iloc[0]}",
    )

    sub = panel[panel["iso3"] == selected_iso].copy()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sub["year"],
            y=sub["cholera_cases"],
            name="Reported Cholera Cases",
            marker_color="#0284c7",
            yaxis="y1",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sub["year"],
            y=sub["max_daily_precip_mm"],
            name="Peak 24h Rainfall (mm)",
            mode="lines+markers",
            line=dict(color="#dc2626", width=2.5),
            yaxis="y2",
        )
    )
    fig.update_layout(
        title=f"Reported Cholera Cases vs. Peak 24-Hour Extreme Rainfall ({selected_iso}, 2000-2024)",
        xaxis_title="Year",
        yaxis=dict(title="Reported Cholera Cases", side="left"),
        yaxis2=dict(title="Max 24h Precipitation (mm)", overlaying="y", side="right"),
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_emt_simulator_tab() -> None:
    st.subheader("WHO Emergency Medical Team (EMT) 14-Day Surge Capacity Simulator")
    st.caption("Operational hospital bed & trauma/waterborne caseload simulator aligned with WHO EMT Minimum Data Set (MDS)")

    with st.sidebar:
        st.header("Disaster & EMT Parameters")
        affected_pop = st.slider("Disaster-Affected Population", 100_000, 1_500_000, 450_000, step=50_000)
        base_beds = st.slider("Pre-Disaster Provincial Beds", 100, 1_000, 380, step=20)
        damaged_pct = st.slider("Damaged / Non-Functional Beds (%)", 0, 70, 30, step=5) / 100.0
        emt_day = st.slider("WHO EMT Arrival Day", 1, 10, 4)
        emt_beds = st.slider("WHO EMT Added Surge Beds", 20, 300, 120, step=10)
        cholera_ar = st.slider("14-Day Cholera / AWD Attack Rate (%)", 0.10, 1.50, 0.45, step=0.05) / 100.0

    params = EMTSurgeParams(
        affected_population=affected_pop,
        baseline_hospital_beds=base_beds,
        damaged_bed_pct=damaged_pct,
        emt_arrival_day=emt_day,
        emt_added_beds=emt_beds,
        cholera_attack_rate_14d=cholera_ar,
    )
    df = simulate_emt_14day_surge(params)

    peak_row = df.loc[df["inpatient_census"].idxmax()]
    total_unmet_bed_days = df["unmet_bed_demand"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Peak Inpatient Census", f"{peak_row['inpatient_census']:.0f} pts", f"Day {int(peak_row['day'])}")
    c2.metric("Peak Bed Occupancy", f"{peak_row['occupancy_pct']:.1f}%")
    c3.metric("Peak Single-Day Overflow", f"{peak_row['unmet_bed_demand']:.0f} pts")
    c4.metric("Cumulative Unmet Bed-Days", f"{total_unmet_bed_days:,.0f}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["day"], y=df["new_trauma_cases"], name="Daily Trauma Admissions", marker_color="#f59e0b"))
    fig.add_trace(go.Bar(x=df["day"], y=df["new_awd_cholera_cases"], name="Daily AWD/Cholera Admissions", marker_color="#0284c7"))
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["inpatient_census"],
            name="Active Inpatient Census",
            mode="lines+markers",
            line=dict(color="#dc2626", width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["day"],
            y=df["available_beds"],
            name="Functional Bed Capacity",
            mode="lines",
            line=dict(color="#0f766e", dash="dash", width=3, shape="hv"),
        )
    )
    fig.update_layout(
        barmode="stack",
        xaxis_title="Days Post-Disaster Landfall",
        yaxis_title="Patients / Inpatient Beds",
        height=460,
    )
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Disaster Epidemiology, Outbreak Early-Warning & Humanitarian Access")
    panel = load_climate_panel()
    access_df = load_mozambique_access()

    tab1, tab2, tab3 = st.tabs([
        "1. Humanitarian Health Access & Sphere Gaps",
        "2. Climate-Cholera Early Warning",
        "3. WHO EMT 14-Day Surge Simulator",
    ])

    with tab1:
        render_spatial_access_tab(access_df)
    with tab2:
        render_climate_outbreak_tab(panel)
    with tab3:
        render_emt_simulator_tab()


if __name__ == "__main__":
    main()
