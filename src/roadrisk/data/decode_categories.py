from __future__ import annotations

import pandas as pd

SEVERITY_MAP = {
    1: "Fatal",
    2: "Serious",
    3: "Slight",
    "1": "Fatal",
    "2": "Serious",
    "3": "Slight",
    "Fatal": "Fatal",
    "Serious": "Serious",
    "Slight": "Slight",
}

CASUALTY_TYPE_GROUPS = {
    0: "pedestrian",
    1: "cyclist",
    2: "motorcyclist",
    3: "motorcyclist",
    4: "motorcyclist",
    5: "motorcyclist",
    23: "motorcyclist",
    97: "motorcyclist",
    103: "motorcyclist",
    104: "motorcyclist",
    105: "motorcyclist",
    106: "motorcyclist",
    8: "car_occupant",
    9: "car_occupant",
    108: "car_occupant",
    109: "car_occupant",
    10: "bus_or_coach",
    11: "bus_or_coach",
    110: "bus_or_coach",
    16: "other",
    17: "other",
    18: "other",
    19: "other",
    20: "other",
    21: "other",
    22: "other",
    90: "other",
    98: "other",
    99: "other",
    113: "other",
}

CASUALTY_TYPE_LABELS = {
    0: "Pedestrian",
    1: "Cyclist",
    2: "Motorcycle 50cc and under",
    3: "Motorcycle 125cc and under",
    4: "Motorcycle over 125cc and up to 500cc",
    5: "Motorcycle over 500cc",
    8: "Taxi/private hire car occupant",
    9: "Car occupant",
    10: "Minibus occupant",
    11: "Bus or coach occupant",
    16: "Horse rider",
    17: "Agricultural vehicle occupant",
    18: "Tram occupant",
    19: "Van/goods vehicle under 3.5t occupant",
    20: "Goods vehicle 3.5t to 7.5t occupant",
    21: "Goods vehicle 7.5t and over occupant",
    22: "Mobility scooter rider",
    23: "Electric motorcycle rider or passenger",
    90: "Other vehicle occupant",
    97: "Motorcycle unknown cc",
    98: "Goods vehicle unknown weight occupant",
    99: "Unknown vehicle type",
    103: "Motorcycle scooter, 1979-1998",
    104: "Motorcycle, 1979-1998",
    105: "Motorcycle combination, 1979-1998",
    106: "Motorcycle over 125cc, 1999-2004",
    108: "Taxi, 1979-2004",
    109: "Car including private hire cars, 1979-2004",
    110: "Minibus/motor caravan, 1979-1998",
    113: "Goods over 3.5 tonnes, 1979-1998",
    -1: "Data missing or out of range",
}

CONDITION_FIELD_LABELS = {
    "speed_limit": "Speed limit",
    "road_type": "Road type",
    "junction_detail": "Junction type",
    "junction_control": "Junction control",
    "light_conditions": "Light conditions",
    "weather_conditions": "Weather",
    "road_surface_conditions": "Road surface",
    "urban_or_rural_area": "Urban/rural",
}

CONDITION_CODE_LABELS = {
    "road_type": {
        "1": "Roundabout",
        "2": "One way street",
        "3": "Dual carriageway",
        "6": "Single carriageway",
        "7": "Slip road",
        "9": "Unknown",
        "12": "One way street / slip road",
        "-1": "Data missing or out of range",
    },
    "junction_detail": {
        "0": "Not at or near junction",
        "13": "T or staggered junction",
        "16": "Crossroads",
        "17": "Junction with more than four arms",
        "18": "Private drive or entrance",
        "19": "Other junction",
        "99": "Unknown",
        "-1": "Data missing or out of range",
    },
    "junction_control": {
        "0": "Not at or near junction",
        "1": "Authorised person",
        "2": "Automatic traffic signal",
        "3": "Stop sign",
        "4": "Give way or uncontrolled",
        "9": "Unknown",
        "-1": "Data missing or out of range",
    },
    "light_conditions": {
        "1": "Daylight",
        "4": "Darkness - lights lit",
        "5": "Darkness - lights unlit",
        "6": "Darkness - no lighting",
        "7": "Darkness - lighting unknown",
        "-1": "Data missing or out of range",
    },
    "weather_conditions": {
        "1": "Fine, no high winds",
        "2": "Raining, no high winds",
        "3": "Snowing, no high winds",
        "4": "Fine with high winds",
        "5": "Raining with high winds",
        "6": "Snowing with high winds",
        "7": "Fog or mist",
        "8": "Other weather",
        "9": "Unknown",
        "-1": "Data missing or out of range",
    },
    "road_surface_conditions": {
        "1": "Dry",
        "2": "Wet or damp",
        "3": "Snow",
        "4": "Frost or ice",
        "5": "Flood over 3cm deep",
        "6": "Oil or diesel",
        "7": "Mud",
        "9": "Unknown",
        "-1": "Data missing or out of range",
    },
    "urban_or_rural_area": {
        "1": "Urban",
        "2": "Rural",
        "3": "Unallocated",
        "-1": "Data missing or out of range",
    },
    "speed_limit": {
        "99": "Unknown",
        "-1": "Data missing or out of range",
    },
}


def decode_severity(series: pd.Series) -> pd.Series:
    decoded = series.map(SEVERITY_MAP)
    decoded = decoded.fillna(series.astype(str).str.strip().str.title())
    decoded = decoded.where(decoded.isin(["Fatal", "Serious", "Slight"]))
    return decoded


def is_ksi_from_severity(series: pd.Series) -> pd.Series:
    decoded = decode_severity(series)
    return decoded.map({"Fatal": 1, "Serious": 1, "Slight": 0})


def casualty_group(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.map(CASUALTY_TYPE_GROUPS).fillna("other")


def casualty_type_label(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    return numeric.map(CASUALTY_TYPE_LABELS).fillna("Unknown or uncoded")


def condition_field_label(field: str) -> str:
    return CONDITION_FIELD_LABELS.get(field, field.replace("_", " ").title())


def condition_category_label(field: str, value: object) -> str:
    if pd.isna(value):
        return "Missing"
    value_text = str(value)
    if value_text.endswith(".0"):
        value_text = value_text[:-2]
    if field == "speed_limit":
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.notna(numeric) and int(numeric) > 0:
            return f"{int(numeric)} mph"
    return CONDITION_CODE_LABELS.get(field, {}).get(value_text, f"Unmapped value {value_text}")
