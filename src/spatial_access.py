from dataclasses import dataclass
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# INE Mozambique official population projections (2023/2024, Maputo Province + Maputo City combined to match ADM1 polygon)
MOZ_PROVINCE_POP = {
    "Nampula": 6_649_897,
    "Zambézia": 6_003_941,
    "Tete": 3_173_797,
    "Maputo": 3_622_400,
    "Cabo Delgado": 2_745_400,
    "Sofala": 2_674_800,
    "Manica": 2_298_700,
    "Niassa": 2_203_900,
    "Inhambane": 1_586_300,
    "Gaza": 1_484_600,
}

# Sphere Humanitarian Standard: minimum 1 primary health care unit per 10,000 population
SPHERE_POP_PER_FACILITY = 10_000


def compute_province_health_access() -> gpd.GeoDataFrame:
    hs = gpd.read_file(RAW_DIR / "mozambique_healthsites.geojson")
    adm1 = gpd.read_file(RAW_DIR / "mozambique_adm1.geojson")

    if hs.crs != adm1.crs:
        hs = hs.to_crs(adm1.crs)

    # Spatial join: assign each mapped health facility to its ADM1 province
    joined = gpd.sjoin(hs, adm1[["shapeName", "geometry"]], how="inner", predicate="within")
    counts = joined.groupby("shapeName").size().rename("mapped_facilities").reset_index()

    gdf = adm1[["shapeName", "geometry"]].merge(counts, on="shapeName", how="left")
    gdf["mapped_facilities"] = gdf["mapped_facilities"].fillna(0).astype(int)
    gdf["population"] = gdf["shapeName"].map(MOZ_PROVINCE_POP)

    gdf["pop_per_facility"] = np.where(
        gdf["mapped_facilities"] > 0,
        gdf["population"] / gdf["mapped_facilities"],
        np.nan,
    )
    gdf["sphere_required_facilities"] = np.ceil(gdf["population"] / SPHERE_POP_PER_FACILITY).astype(int)
    gdf["sphere_facility_gap"] = np.maximum(0, gdf["sphere_required_facilities"] - gdf["mapped_facilities"])
    gdf["sphere_coverage_pct"] = (gdf["mapped_facilities"] / gdf["sphere_required_facilities"]) * 100.0

    return gdf.sort_values("pop_per_facility", ascending=False).reset_index(drop=True)


@dataclass
class EMTSurgeParams:
    affected_population: int = 450_000
    baseline_hospital_beds: int = 380
    damaged_bed_pct: float = 0.30
    emt_arrival_day: int = 4
    emt_added_beds: int = 120
    cholera_attack_rate_14d: float = 0.0045   # 0.45% cumulative 14-day AWD/cholera attack rate
    trauma_rate_14d: float = 0.0020           # 0.20% acute flood/cyclone injury rate
    avg_length_of_stay_days: float = 3.5


def simulate_emt_14day_surge(params: EMTSurgeParams) -> pd.DataFrame:
    days = np.arange(1, 15)

    # Epi-curve weights across 14 days: trauma peaks Days 1-3; waterborne AWD/cholera peaks Days 5-9
    trauma_weights = np.exp(-0.5 * ((days - 2.0) / 1.4) ** 2)
    trauma_weights /= trauma_weights.sum()

    awd_weights = np.exp(-0.5 * ((days - 7.0) / 2.2) ** 2)
    awd_weights /= awd_weights.sum()

    new_trauma = params.affected_population * params.trauma_rate_14d * trauma_weights
    new_awd = params.affected_population * params.cholera_attack_rate_14d * awd_weights
    total_new_admissions = new_trauma + new_awd

    # Effective bed capacity before and after WHO EMT Type 2 / Cholera Treatment Center deployment
    usable_local_beds = params.baseline_hospital_beds * (1.0 - params.damaged_bed_pct)
    available_beds = np.where(
        days >= params.emt_arrival_day,
        usable_local_beds + params.emt_added_beds,
        usable_local_beds,
    )

    # Dynamic inpatient census with geometric discharge rate (1 / ALOS)
    discharge_rate = 1.0 / params.avg_length_of_stay_days
    census = np.zeros(len(days))
    for i in range(len(days)):
        prev_census = census[i - 1] * (1.0 - discharge_rate) if i > 0 else usable_local_beds * 0.65
        census[i] = prev_census + total_new_admissions[i]

    unmet_bed_demand = np.maximum(0.0, census - available_beds)
    occupancy_pct = (census / available_beds) * 100.0

    return pd.DataFrame({
        "day": days,
        "new_trauma_cases": np.round(new_trauma, 1),
        "new_awd_cholera_cases": np.round(new_awd, 1),
        "total_new_admissions": np.round(total_new_admissions, 1),
        "inpatient_census": np.round(census, 1),
        "available_beds": np.round(available_beds, 1),
        "unmet_bed_demand": np.round(unmet_bed_demand, 1),
        "occupancy_pct": np.round(occupancy_pct, 1),
    })


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    gdf = compute_province_health_access()
    out_gpkg = PROCESSED_DIR / "mozambique_province_access.gpkg"
    gdf.to_file(out_gpkg, driver="GPKG")

    print(f"Saved spatial access layer -> {out_gpkg.name}")
    print("\n=== Mozambique Provincial Health Facility Access vs. Sphere Standard (1 per 10,000) ===")
    cols = ["shapeName", "population", "mapped_facilities", "pop_per_facility", "sphere_coverage_pct", "sphere_facility_gap"]
    print(gdf[cols].to_string(index=False))

    surge_df = simulate_emt_14day_surge(EMTSurgeParams())
    surge_df.to_parquet(PROCESSED_DIR / "emt_14day_surge_baseline.parquet", index=False)
    peak_row = surge_df.loc[surge_df["inpatient_census"].idxmax()]
    print(
        f"\nEMT Surge Peak (Day {int(peak_row['day'])}): {peak_row['inpatient_census']:.0f} patients vs "
        f"{peak_row['available_beds']:.0f} beds ({peak_row['occupancy_pct']:.1f}% occupancy, "
        f"{peak_row['unmet_bed_demand']:.0f} overflow patients)"
    )


if __name__ == "__main__":
    main()
