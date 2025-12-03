import os
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
INPUT_FILE = r"C:/tree-canopy/data/merged_predictions.parquet"
OUTPUT_FILE = r"C:/tree-canopy/data/berlin_natural_shapes.parquet"

# 0.60 (60%) is the "Sweet Spot" for natural shapes.
# If Tree B is 60% covered by Tree A -> Delete Tree B.
# This gets rid of the "cut-off" halves without needing to clip geometries.
CONTAINMENT_THRESHOLD = 0.60 

def clean_overlaps_natural(gdf):
    """
    Removes significantly overlapping smaller trees to leave only the 
    dominant, full tree shapes. No clipping is performed.
    """
    print(f"Starting Natural Cleaning on {len(gdf)} trees...")
    print(f"Threshold: {CONTAINMENT_THRESHOLD * 100}% overlap triggers deletion.")
    
    # 1. Calculate Area and Sort (Largest First)
    # We always want to keep the big, full tree.
    if 'area_calc' not in gdf.columns:
        gdf['area_calc'] = gdf.geometry.area
    
    # Sort Largest -> Smallest
    gdf = gdf.sort_values(by='area_calc', ascending=False).reset_index(drop=True)
    
    sindex = gdf.sindex
    drop_indices = set()
    
    for i in tqdm(range(len(gdf)), desc="Removing Overlaps"):
        if i in drop_indices: continue
            
        current_poly = gdf.geometry.iloc[i]
        
        # Find neighbors
        possible_matches = list(sindex.intersection(current_poly.bounds))
        
        for j in possible_matches:
            # Only look at SMALLER trees (j > i)
            if j <= i or j in drop_indices: continue
            
            candidate_poly = gdf.geometry.iloc[j]
            
            if not current_poly.intersects(candidate_poly): continue
            
            # Intersection Area
            inter_area = current_poly.intersection(candidate_poly).area
            candidate_area = gdf['area_calc'].iloc[j]
            
            if candidate_area > 0:
                # How much of the SMALL tree is covered by the BIG tree?
                fraction = inter_area / candidate_area
                
                # If > 60% covered, we assume it's a duplicate/artifact -> DELETE
                if fraction > CONTAINMENT_THRESHOLD:
                    drop_indices.add(j)

    print(f"  > Removing {len(drop_indices)} overlapping trees.")
    gdf_clean = gdf.drop(index=list(drop_indices)).reset_index(drop=True)
    return gdf_clean.drop(columns=['area_calc'])

def main():
    if not os.path.exists(INPUT_FILE):
        print("File not found.")
        return

    print("Loading data...")
    gdf = gpd.read_parquet(INPUT_FILE)
    
    if gdf.crs is None:
        gdf.set_crs("EPSG:25833", inplace=True)
        
    print("Fixing geometry...")
    gdf['geometry'] = gdf.geometry.buffer(0)
    
    initial_count = len(gdf)

    gdf = clean_overlaps_natural(gdf)
    
    # --- Save ---
    print(f"Saving to {OUTPUT_FILE}...")
    gdf.to_parquet(OUTPUT_FILE)
    
    # Optional GPKG for QGIS
    # gdf.to_file(OUTPUT_FILE.replace('.parquet', '.gpkg'), driver='GPKG')

    print("--- Summary ---")
    print(f"Original: {initial_count}")
    print(f"Final:    {len(gdf)}")
    print(f"Deleted:  {initial_count - len(gdf)}")

if __name__ == "__main__":
    main()