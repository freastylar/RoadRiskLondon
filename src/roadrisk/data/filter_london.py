from __future__ import annotations

import pandas as pd

from roadrisk.utils.validation import DataValidationError

LONDON_ONS_DISTRICTS = {
    "E09000001",
    "E09000002",
    "E09000003",
    "E09000004",
    "E09000005",
    "E09000006",
    "E09000007",
    "E09000008",
    "E09000009",
    "E09000010",
    "E09000011",
    "E09000012",
    "E09000013",
    "E09000014",
    "E09000015",
    "E09000016",
    "E09000017",
    "E09000018",
    "E09000019",
    "E09000020",
    "E09000021",
    "E09000022",
    "E09000023",
    "E09000024",
    "E09000025",
    "E09000026",
    "E09000027",
    "E09000028",
    "E09000029",
    "E09000030",
    "E09000031",
    "E09000032",
    "E09000033",
}

LONDON_BOROUGH_BY_ONS = {
    "E09000001": "City of London",
    "E09000002": "Barking and Dagenham",
    "E09000003": "Barnet",
    "E09000004": "Bexley",
    "E09000005": "Brent",
    "E09000006": "Bromley",
    "E09000007": "Camden",
    "E09000008": "Croydon",
    "E09000009": "Ealing",
    "E09000010": "Enfield",
    "E09000011": "Greenwich",
    "E09000012": "Hackney",
    "E09000013": "Hammersmith and Fulham",
    "E09000014": "Haringey",
    "E09000015": "Harrow",
    "E09000016": "Havering",
    "E09000017": "Hillingdon",
    "E09000018": "Hounslow",
    "E09000019": "Islington",
    "E09000020": "Kensington and Chelsea",
    "E09000021": "Kingston upon Thames",
    "E09000022": "Lambeth",
    "E09000023": "Lewisham",
    "E09000024": "Merton",
    "E09000025": "Newham",
    "E09000026": "Redbridge",
    "E09000027": "Richmond upon Thames",
    "E09000028": "Southwark",
    "E09000029": "Sutton",
    "E09000030": "Tower Hamlets",
    "E09000031": "Waltham Forest",
    "E09000032": "Wandsworth",
    "E09000033": "Westminster",
}

LONDON_BOROUGH_NAMES = {
    "city of london",
    "barking and dagenham",
    "barnet",
    "bexley",
    "brent",
    "bromley",
    "camden",
    "croydon",
    "ealing",
    "enfield",
    "greenwich",
    "hackney",
    "hammersmith and fulham",
    "haringey",
    "harrow",
    "havering",
    "hillingdon",
    "hounslow",
    "islington",
    "kensington and chelsea",
    "kingston upon thames",
    "lambeth",
    "lewisham",
    "merton",
    "newham",
    "redbridge",
    "richmond upon thames",
    "southwark",
    "sutton",
    "tower hamlets",
    "waltham forest",
    "wandsworth",
    "westminster",
}


def london_mask(df: pd.DataFrame, allow_coordinate_fallback: bool = False) -> pd.Series:
    if "local_authority_ons_district" in df.columns:
        mask = df["local_authority_ons_district"].astype(str).str.upper().isin(LONDON_ONS_DISTRICTS)
        if mask.any():
            return mask
    for column in ["local_authority_district", "local_authority_highway", "borough_name"]:
        if column in df.columns:
            values = df[column].astype(str).str.lower().str.replace(r"\s+", " ", regex=True).str.strip()
            mask = values.isin(LONDON_BOROUGH_NAMES)
            if mask.any():
                return mask
    if allow_coordinate_fallback and {"longitude", "latitude"}.issubset(df.columns):
        lon = pd.to_numeric(df["longitude"], errors="coerce")
        lat = pd.to_numeric(df["latitude"], errors="coerce")
        mask = lon.between(-0.55, 0.35) & lat.between(51.25, 51.75)
        if mask.any():
            return mask
    raise DataValidationError(
        "Could not derive a non-empty London filter from verified administrative fields"
    )


def filter_london_collisions(df: pd.DataFrame, allow_coordinate_fallback: bool = False) -> pd.DataFrame:
    filtered = df.loc[london_mask(df, allow_coordinate_fallback=allow_coordinate_fallback)].copy()
    if filtered.empty:
        raise DataValidationError("London collision filter returned no rows")
    return filtered
