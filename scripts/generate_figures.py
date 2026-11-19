import sys
from pathlib import Path
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.spatial_access import EMTSurgeParams, simulate_emt_14day_surge

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
FIG_DIR = ROOT_DIR / "outputs" / "figures"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def plot_mozambique_access_map() -> None:
    gdf = gpd.read_file(PROCESSED_DIR / "mozambique_province_access.gpkg")
    hs = gpd.read_file(RAW_DIR / "mozambique_healthsites.geojson")
    if hs.crs != gdf.crs:
        hs = hs.to_crs(gdf.crs)

    fig, ax = plt.subplots(figsize=(7.5, 8.5))
    gdf.plot(
        column="sphere_coverage_pct",
        cmap="YlOrRd_r",
        linewidth=0.9,
        edgecolor="#1e293b",
        legend=True,
        legend_kwds={
            "label": "Sphere Standard Coverage (% of 1 Facility / 10,000 Pop)",
            "shrink": 0.65,
        },
        ax=ax,
    )
    hs.plot(ax=ax, color="#0f172a", markersize=4, alpha=0.45, label=f"HDX/OSM Health Facilities (n={int(gdf['mapped_facilities'].sum()):,})")

    for _, row in gdf.iterrows():
        centroid = row["geometry"].representative_point()
        ax.annotate(
            f"{row['shapeName']}\n({row['sphere_coverage_pct']:.1f}%)",
            xy=(centroid.x, centroid.y),
            ha="center",
            va="center",
            fontsize=7.5,
            fontweight="bold",
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax.set_title("Mozambique Health Access Gap vs. Sphere Humanitarian Standards", fontsize=11.5, fontweight="bold", pad=12)
    ax.set_xlabel("Longitude", fontsize=9)
    ax.set_ylabel("Latitude", fontsize=9)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "mozambique_sphere_access_map.png")
    plt.close(fig)


def plot_cholera_climate_surge() -> None:
    panel = pd.read_parquet(PROCESSED_DIR / "cholera_climate_panel.parquet")
    outbreaks = panel[panel["cholera_cases"] > 0].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.8))

    for iso3, grp in outbreaks.groupby("iso3"):
        ax1.scatter(
            grp["max_daily_precip_mm"],
            grp["cholera_cases"],
            s=65,
            alpha=0.85,
            label=iso3,
        )
    ax1.set_yscale("log")
    ax1.set_title("Peak 24h Rainfall vs. Reported Cholera Outbreaks", fontsize=10.5, fontweight="bold")
    ax1.set_xlabel("Max 24-Hour Precipitation (mm, ERA5)", fontsize=9.5)
    ax1.set_ylabel("Reported Cholera Cases (Log Scale)", fontsize=9.5)
    ax1.grid(True, linestyle=":", alpha=0.45)
    ax1.legend(title="Corridor", fontsize=8, frameon=True)

    agg = outbreaks.groupby("iso3", as_index=False).agg(
        total_cases=("cholera_cases", "sum"),
        mean_cfr=("cfr_pct", "mean"),
    ).sort_values("total_cases", ascending=True)

    bars = ax2.barh(agg["iso3"], agg["total_cases"] / 1000.0, color="#0284c7", edgecolor="#0369a1")
    ax2.set_title("Cumulative Notified Cholera Burden (WHO GHO)", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Cumulative Reported Cases (Thousands)", fontsize=9.5)
    ax2.grid(True, axis="x", linestyle=":", alpha=0.45)

    for bar, (_, row) in zip(bars, agg.iterrows()):
        ax2.text(
            bar.get_width() + 5,
            bar.get_y() + bar.get_height() / 2,
            f"CFR: {row['mean_cfr']:.1f}%",
            va="center",
            fontsize=8.5,
            fontweight="bold",
            color="#1e293b",
        )
    ax2.set_xlim(0, agg["total_cases"].max() / 1000.0 * 1.25)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "cholera_climate_surge.png")
    plt.close(fig)


def plot_emt_surge_capacity() -> None:
    df = simulate_emt_14day_surge(EMTSurgeParams())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(df["day"] - 0.18, df["new_trauma_cases"], width=0.35, color="#f59e0b", label="Daily Trauma Admissions")
    ax.bar(df["day"] + 0.18, df["new_awd_cholera_cases"], width=0.35, color="#0284c7", label="Daily AWD / Cholera Admissions")

    ax.plot(df["day"], df["inpatient_census"], color="#dc2626", marker="o", linewidth=2.2, label="Active Inpatient Census")
    ax.step(df["day"], df["available_beds"], where="mid", color="#0f766e", linestyle="--", linewidth=2.2, label="Usable Bed Capacity (EMT arrives Day 4)")

    ax.fill_between(
        df["day"],
        df["available_beds"],
        df["inpatient_census"],
        where=(df["inpatient_census"] > df["available_beds"]),
        color="#fecaca",
        alpha=0.5,
        interpolate=True,
        label="Unmet Surge Bed Deficit",
    )

    ax.set_title("14-Day Post-Cyclone WHO EMT Surge & Bed Bottleneck Simulation (Pop: 450,000)", fontsize=11, fontweight="bold", pad=12)
    ax.set_xlabel("Days Post-Disaster Landfall", fontsize=10)
    ax.set_ylabel("Patients / Inpatient Beds", fontsize=10)
    ax.set_xticks(df["day"])
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(loc="upper left", fontsize=8.5, frameon=True)

    fig.tight_layout()
    fig.savefig(FIG_DIR / "emt_surge_capacity.png")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_mozambique_access_map()
    plot_cholera_climate_surge()
    plot_emt_surge_capacity()
    print("Generated 3 PNG figures in outputs/figures/:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
