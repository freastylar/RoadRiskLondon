import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from kmodes.kmodes import KModes
import json
import numpy as np
from roadrisk.data.decode_categories import condition_category_label

processed_path = Path("data/processed/collisions_london.parquet") 
if not processed_path.exists():
    processed_files = list(Path("data/processed/").glob("*.parquet"))
    processed_path = processed_files[0] if processed_files else None

if not processed_path:
    raise FileNotFoundError("Processed collision data missing.")

df = pd.read_parquet(processed_path)

cluster_features = [
    'road_type', 
    'speed_limit', 
    'light_conditions', 
    'weather_conditions', 
    'junction_detail', 
    'urban_or_rural_area'
]

df_model = df[cluster_features].astype(str).copy()

print("Initializing K-Modes clustering...")
n_clusters = 9
km = KModes(n_clusters=n_clusters, init='Huang', n_init=3, random_state=42)
cluster_labels = km.fit_predict(df_model)

centroids = km.cluster_centroids_
mismatches = np.sum(df_model.values != centroids[cluster_labels], axis=1)

df['typology_cluster'] = np.where(mismatches >= 2, -1, cluster_labels)

summary_data = {}

for cluster_id in range(n_clusters):
    df_c = df[df['typology_cluster'] == cluster_id]
    total_incidents = len(df_c)
    if total_incidents == 0: continue
        
    ksi_count = df_c['accident_severity'].isin(['Fatal', 'Serious', 1, 2]).sum()
    ksi_ratio = (ksi_count / total_incidents) * 100
    
    modes = {}
    for col in cluster_features:
        raw_mode_value = df_c[col].mode()[0]
        modes[col] = condition_category_label(col, raw_mode_value)
        
    dynamic_title = f"Profile {cluster_id}: {modes['road_type']} ({modes['speed_limit']}) under {modes['light_conditions']}"
    
    summary_data[str(cluster_id)] = {
        "dynamic_title": dynamic_title,
        "total_incidents": int(total_incidents),
        "ksi_ratio": float(ksi_ratio),
        "modes": modes
    }

df_noise = df[df['typology_cluster'] == -1]
total_noise = len(df_noise)
if total_noise > 0:
    ksi_count_noise = df_noise['accident_severity'].isin(['Fatal', 'Serious', 1, 2]).sum()
    ksi_ratio_noise = (ksi_count_noise / total_noise) * 100
    
    summary_data["-1"] = {
        "dynamic_title": "Profile -1: Unclassified Anomalies (Outliers)",
        "total_incidents": int(total_noise),
        "ksi_ratio": float(ksi_ratio_noise),
        "modes": {col: "Various (No dominant pattern)" for col in cluster_features}
    }

output_dir = Path("data/app")
output_dir.mkdir(parents=True, exist_ok=True)

df_raw_out = df[['latitude', 'longitude', 'typology_cluster', 'accident_severity']].dropna(subset=['latitude', 'longitude']).copy()
df_raw_out['accident_severity'] = df_raw_out['accident_severity'].astype(str)
df_raw_out['is_hotspot'] = False
df_raw_out['weight'] = 1
df_raw_out['ksi_count'] = df_raw_out['accident_severity'].apply(lambda x: 1 if x in ['Fatal', 'Serious', '1', '2'] else 0)

df['lat_round'] = df['latitude'].round(3)
df['lon_round'] = df['longitude'].round(3)
df_agg_out = df.groupby(['lat_round', 'lon_round', 'typology_cluster']).agg(
    weight=('typology_cluster', 'count'),
    ksi_count=('accident_severity', lambda x: x.isin(['Fatal', 'Serious', 1, 2, '1', '2']).sum())
).reset_index()
df_agg_out = df_agg_out.rename(columns={'lat_round': 'latitude', 'lon_round': 'longitude'})
df_agg_out['is_hotspot'] = True
df_agg_out['accident_severity'] = 'Aggregated'

map_export_df = pd.concat([df_raw_out, df_agg_out], ignore_index=True)
map_export_df.to_parquet(output_dir / "typology_map_points.parquet")

with open(output_dir / "typology_summary.json", "w") as f:
    json.dump(summary_data, f, indent=4)

print("Typology artifacts successfully exported to data/app/.")