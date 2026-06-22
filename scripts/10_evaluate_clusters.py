import pandas as pd
from pathlib import Path
from kmodes.kmodes import KModes

# Locate processed London data from the pipeline
processed_path = Path("data/processed/collisions_london.parquet")
if not processed_path.exists():
    processed_files = list(Path("data/processed/").glob("*.parquet"))
    processed_path = processed_files[0] if processed_files else None

if not processed_path:
    raise FileNotFoundError("Processed collision data missing.")

df = pd.read_parquet(processed_path)

# Segment based on core environmental and infrastructure attributes
cluster_features = [
    'road_type', 
    'speed_limit', 
    'light_conditions', 
    'weather_conditions', 
    'junction_detail', 
    'urban_or_rural_area'
]
df_model = df[cluster_features].astype(str).copy()

# Evaluate cluster counts from 2 to 49 to trace the total mismatch cost curve
print("Evaluating K-Modes cost across different cluster configurations...")
for k in range(2, 50):
    km = KModes(n_clusters=k, init='Huang', n_init=1, random_state=42)
    km.fit(df_model)
    print(f"Clusters (K): {k} | Total Mismatch Cost: {km.cost_:.1f}")