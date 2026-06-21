"""Fetch the London drivable road network from OpenStreetMap (via osmnx).

Downloads road centrelines for Greater London, keeps the useful columns, and
saves them as a GeoParquet for the trip-risk road snapping step. Roads are the
real named segments people recognise (e.g. "A40", "Euston Road").

Usage:
    python scripts/14_fetch_london_roads.py
"""

from __future__ import annotations

from pathlib import Path

import osmnx as ox

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "data" / "raw" / "roads" / "london_roads.parquet"

# Only meaningful through-roads, not every service road / alley, so the map and
# snapping stay manageable and relevant to trips.
KEEP_HIGHWAY = {
    "motorway", "motorway_link", "trunk", "trunk_link",
    "primary", "primary_link", "secondary", "secondary_link",
    "tertiary", "tertiary_link", "residential", "unclassified",
}


def main() -> None:
    print("Fetching Greater London drivable road network from OSM (this can take a few minutes)...")
    G = ox.graph_from_place("Greater London, England", network_type="drive", simplify=True)
    edges = ox.graph_to_gdfs(G, nodes=False).reset_index(drop=True)

    def _norm(value):
        return value[0] if isinstance(value, list) else value

    edges["highway"] = edges["highway"].map(_norm)
    edges["name"] = edges.get("name").map(_norm) if "name" in edges.columns else None
    edges["ref"] = edges.get("ref").map(_norm) if "ref" in edges.columns else None
    edges = edges[edges["highway"].isin(KEEP_HIGHWAY)].copy()

    # Keep richer infrastructure tags for severity features, normalising list cells.
    for col in ["maxspeed", "cycleway", "lit", "lanes", "bridge", "tunnel"]:
        if col in edges.columns:
            edges[col] = edges[col].map(_norm)
    keep = [c for c in ["name", "ref", "highway", "length", "maxspeed", "cycleway",
                        "lit", "lanes", "bridge", "tunnel", "geometry"] if c in edges.columns]
    edges = edges[keep]
    # A display label: prefer the road ref (A40), else the name, else the class.
    edges["road_label"] = (
        edges["ref"].fillna("").astype(str).where(edges["ref"].notna(), "")
    )
    edges["road_label"] = edges.apply(
        lambda r: (str(r.get("ref")) if r.get("ref") else None)
        or (str(r.get("name")) if r.get("name") else None)
        or f"{r['highway'].title()} road",
        axis=1,
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(OUT)
    print(f"Saved {len(edges):,} road segments to {OUT}")
    print("Example roads:", edges["road_label"].dropna().unique()[:10].tolist())


if __name__ == "__main__":
    main()
