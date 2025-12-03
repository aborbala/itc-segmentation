import os
import glob
from tqdm import tqdm

# Use "r" before the string to handle Windows backslashes
input_folder = r"C:/tree-canopy/data/preds_no_struct_veg_mask_all_crowns_dice_loss_experiment_lr_0.0025_L0.3_D0.7"
output_file  = r"C:/tree-canopy/data/merged_predictions.geojson"
# ==========================================

def fast_merge():
    # 1. Get list of files
    print(f"Scanning files in {input_folder}...")
    files = glob.glob(os.path.join(input_folder, "*.geojson"))
    
    total_files = len(files)
    print(f"Found {total_files} files.")
    
    if total_files == 0:
        print("❌ No GeoJSON files found! Check your path.")
        return

    # 2. Create the Output File
    with open(output_file, 'w', encoding='utf-8') as outfile:
        
        # Write GeoJSON Header
        outfile.write('{"type": "FeatureCollection", "features": [\n')
        
        first_file = True
        
        # Iterate and Merge
        for filename in tqdm(files, desc="Merging", unit="file"):
            try:
                with open(filename, 'r', encoding='utf-8') as infile:
                    content = infile.read()
                    
                    start_index = content.find('[') + 1
                    end_index = content.rfind(']')
                    
                    # specific check to ensure file isn't empty or malformed
                    if start_index > 0 and end_index > 0:
                        features_body = content[start_index:end_index].strip()
                        
                        if features_body:
                            # If this isn't the first file, add a comma
                            if not first_file:
                                outfile.write(',\n')
                            
                            outfile.write(features_body)
                            first_file = False
                            
            except Exception as e:
                print(f"\nError reading {filename}: {e}")

        # Write GeoJSON Footer
        outfile.write('\n]}')

    print(f"DONE! Merged file saved to: {output_file}")

if __name__ == "__main__":
    fast_merge()