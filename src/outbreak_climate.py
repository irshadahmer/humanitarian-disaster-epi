from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

ROOT_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"


def build_cholera_climate_panel() -> pd.DataFrame:
    cholera = pd.read_parquet(RAW_DIR / "who_cholera_surveillance.parquet")
    climate = pd.read_parquet(RAW_DIR / "era5_corridor_climate.parquet")

    cholera["iso3"] = cholera["iso3"].astype(str).str.strip()
    cholera["year"] = pd.to_numeric(cholera["year"], errors="coerce").astype("Int64")
    climate["iso3"] = climate["iso3"].astype(str).str.strip()
    climate["year"] = pd.to_numeric(climate["year"], errors="coerce").astype("Int64")

    panel = climate.merge(cholera, on=["iso3", "year"], how="left")
    panel["year"] = panel["year"].astype(int)
    panel["cholera_cases"] = panel["cholera_cases"].fillna(0.0)
    panel["cholera_deaths"] = panel["cholera_deaths"].fillna(0.0)
    panel["cfr_pct"] = np.where(
        panel["cholera_cases"] > 0,
        (panel["cholera_deaths"] / panel["cholera_cases"]) * 100.0,
        0.0,
    )
    panel["extreme_rain_flag"] = panel["precip_zscore"] >= 1.0
    return panel.sort_values(["iso3", "year"]).reset_index(drop=True)


def fit_climate_outbreak_glm(panel: pd.DataFrame) -> dict:
    df = panel[panel["cholera_cases"] > 0].copy()
    model = smf.glm(
        formula="cholera_cases ~ precip_zscore + max_daily_precip_mm + mean_temp_c + C(iso3)",
        data=df,
        family=sm.families.NegativeBinomial(alpha=1.0),
    ).fit()

    irr_precip = float(np.exp(model.params["precip_zscore"]))
    pval_precip = float(model.pvalues["precip_zscore"])
    irr_extreme_day = float(np.exp(model.params["max_daily_precip_mm"] * 50.0))

    normal_median = float(panel[~panel["extreme_rain_flag"]]["cholera_cases"].median())
    flood_median = float(panel[panel["extreme_rain_flag"]]["cholera_cases"].median())

    return {
        "n_outbreak_years": len(df),
        "irr_per_sd_precip": irr_precip,
        "pval_precip": pval_precip,
        "irr_per_50mm_peak_day": irr_extreme_day,
        "median_cases_normal_yr": normal_median,
        "median_cases_flood_anomaly_yr": flood_median,
    }


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_cholera_climate_panel()
    out_path = PROCESSED_DIR / "cholera_climate_panel.parquet"
    panel.to_parquet(out_path, index=False)

    summary = fit_climate_outbreak_glm(panel)
    print(f"Saved cholera-climate panel: {len(panel)} corridor-years -> {out_path.name}")
    print("\n=== Climate-Outbreak Epidemiological Summary (2000-2024) ===")
    print(f"Outbreak corridor-years analyzed : {summary['n_outbreak_years']}")
    print(f"Incidence Rate Ratio (+1 SD rain): {summary['irr_per_sd_precip']:.2f}x (p = {summary['pval_precip']:.3f})")
    print(f"IRR per +50mm peak 24h rainfall  : {summary['irr_per_50mm_peak_day']:.2f}x")
    print(f"Median cases (Normal vs Z>=1.0)  : {summary['median_cases_normal_yr']:,.0f} vs {summary['median_cases_flood_anomaly_yr']:,.0f}")


if __name__ == "__main__":
    main()
