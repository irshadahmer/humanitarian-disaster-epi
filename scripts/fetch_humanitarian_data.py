from pathlib import Path
import pandas as pd
import requests

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

WHO_INDICATORS = {
    "cholera_cases": "CHOLERA_0000000001",
    "cholera_deaths": "CHOLERA_0000000002",
}

CORRIDOR_COORDS = {
    "MOZ": {"name": "Mozambique (Beira / Sofala)", "lat": -19.84, "lon": 34.84},
    "MWI": {"name": "Malawi (Blantyre / Shire)", "lat": -15.79, "lon": 35.01},
    "YEM": {"name": "Yemen (Hodeidah / Sanaa)", "lat": 14.80, "lon": 42.95},
    "BGD": {"name": "Bangladesh (Coxs Bazar)", "lat": 21.43, "lon": 92.01},
    "HTI": {"name": "Haiti (Artibonite / Port-au-Prince)", "lat": 18.59, "lon": -72.31},
}

HDX_MOZ_HEALTHSITES_URL = (
    "https://data.humdata.org/dataset/33d3a4f8-976b-4f1e-92d8-621751c2d3b8/"
    "resource/8262bf23-921d-485e-9b68-20bc56aedd8b/download/mozambique.geojson"
)

MOZ_ADM1_GEOJSON_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/MOZ/ADM1/"
    "geoBoundaries-MOZ-ADM1_simplified.geojson"
)


def fetch_who_cholera() -> pd.DataFrame:
    frames = []
    for metric, code in WHO_INDICATORS.items():
        url = f"https://ghoapi.azureedge.net/api/{code}"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        records = resp.json().get("value", [])
        df = pd.DataFrame(records)
        df = df[df["SpatialDimType"] == "COUNTRY"][["SpatialDim", "TimeDim", "NumericValue"]].copy()
        df = df.rename(columns={"SpatialDim": "iso3", "TimeDim": "year", "NumericValue": metric})
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
        df[metric] = pd.to_numeric(df[metric], errors="coerce")
        df = df.groupby(["iso3", "year"], as_index=False)[metric].max()
        frames.append(df)

    merged = frames[0].merge(frames[1], on=["iso3", "year"], how="outer")
    merged = merged[merged["year"] >= 2000].sort_values(["iso3", "year"]).reset_index(drop=True)
    return merged


def fetch_era5_climate(start_date: str = "2000-01-01", end_date: str = "2024-12-31") -> pd.DataFrame:
    rows = []
    for iso3, info in CORRIDOR_COORDS.items():
        params = {
            "latitude": info["lat"],
            "longitude": info["lon"],
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["precipitation_sum", "temperature_2m_mean"],
            "timezone": "UTC",
        }
        resp = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=45)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        df = pd.DataFrame({
            "date": pd.to_datetime(daily["time"]),
            "precip_mm": daily["precipitation_sum"],
            "temp_c": daily["temperature_2m_mean"],
        })
        df["year"] = df["date"].dt.year
        annual = df.groupby("year", as_index=False).agg(
            annual_precip_mm=("precip_mm", "sum"),
            max_daily_precip_mm=("precip_mm", "max"),
            mean_temp_c=("temp_c", "mean"),
        )
        annual["iso3"] = iso3
        annual["corridor_name"] = info["name"]
        # Standardized precipitation anomaly (Z-score relative to 2000-2024 baseline)
        annual["precip_zscore"] = (
            annual["annual_precip_mm"] - annual["annual_precip_mm"].mean()
        ) / annual["annual_precip_mm"].std()
        rows.append(annual)

    return pd.concat(rows, ignore_index=True)


def download_file(url: str, dest: Path) -> None:
    resp = requests.get(url, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("1. Fetching WHO GHO Cholera Surveillance data...")
    cholera_df = fetch_who_cholera()
    cholera_df.to_parquet(RAW_DIR / "who_cholera_surveillance.parquet", index=False)
    print(f"   Saved {len(cholera_df):,} country-year rows")

    print("2. Fetching Open-Meteo ERA5 historical rainfall & temperature series...")
    climate_df = fetch_era5_climate()
    climate_df.to_parquet(RAW_DIR / "era5_corridor_climate.parquet", index=False)
    print(f"   Saved {len(climate_df):,} corridor-year rows")

    print("3. Downloading HDX Mozambique Healthsites GeoJSON...")
    hs_path = RAW_DIR / "mozambique_healthsites.geojson"
    download_file(HDX_MOZ_HEALTHSITES_URL, hs_path)
    print(f"   Saved {hs_path.name} ({hs_path.stat().st_size / 1024:.1f} KB)")

    print("4. Downloading Mozambique ADM1 Province Boundaries GeoJSON...")
    adm1_path = RAW_DIR / "mozambique_adm1.geojson"
    download_file(MOZ_ADM1_GEOJSON_URL, adm1_path)
    print(f"   Saved {adm1_path.name} ({adm1_path.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
