from __future__ import annotations

import json
from pathlib import Path
from urllib.request import Request, urlopen

from roadrisk.utils.paths import app_data_dir
from roadrisk.utils.validation import DataValidationError, require_file

ONS_LONDON_BOROUGH_BOUNDARIES_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "Local_Authority_Districts_December_2024_Boundaries_UK_BFC/FeatureServer/0/query"
    "?where=LAD24CD%20LIKE%20%27E09%25%27"
    "&outFields=LAD24CD,LAD24NM"
    "&outSR=4326"
    "&f=geojson"
)


def ensure_borough_boundaries(root: Path, force: bool = False) -> Path:
    path = app_data_dir(root) / "borough_boundaries.geojson"
    if path.exists() and path.stat().st_size > 0 and not force:
        return path
    request = Request(ONS_LONDON_BOROUGH_BOUNDARIES_URL, headers={"User-Agent": "RoadRiskLondon/0.1"})
    with urlopen(request, timeout=120) as response:
        data = json.loads(response.read().decode("utf-8"))
    features = data.get("features", [])
    if len(features) != 33:
        raise DataValidationError(f"Expected 33 London borough boundary features, found {len(features)}")
    path.write_text(json.dumps(data), encoding="utf-8")
    return require_file(path)
