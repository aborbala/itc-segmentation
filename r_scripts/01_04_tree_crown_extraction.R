# Load necessary libraries
library("rstudioapi")
library("Rcpp")
library("sf")
library("lidR")
library("stars")
library("terra")
library("raster")
library("dplyr")

# Set working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# ---- CONFIGURATION ----
#aoi_code <- "386_5818"  # training data
aoi_code <- "384_5816"   # training data

# Handle both ("Meine Ablage") and  ("My Drive") Drive names
drive_options <- c("G:/Meine Ablage", "G:/My Drive")
existing_drive <- drive_options[dir.exists(drive_options)][1]
if (is.na(existing_drive)) stop("No Google Drive mount found at expected paths.")

# Define base paths
base_data_path <- file.path(existing_drive, "masterthesis", "data")
aoi_base_path <- file.path(base_data_path, aoi_code)

# Required file paths
# Extract crowns for small tif tiles
tif_directory <- file.path(aoi_base_path, "sliced_imgs_2020S")
las_nobuild_path <- file.path(aoi_base_path, "LAS_no_structures_veg_mask")
las_files <- list.files(path = las_nobuild_path, pattern = "\\.las$", full.names = TRUE, recursive = FALSE)

#las_nobuild_path <- "C:/tree-canopy/data/386_5818/LAS_no_buildings"

# Outputs
crowns_path <- file.path(aoi_base_path, "crowns_no_structures_veg_mask_Silva")

if (!dir.exists(crowns_path)) {
  dir.create(crowns_path, recursive = TRUE)
}

### Calculate width to height ratio to eliminate elongated polygons
### Using bbox instead of oo-bbox
calculate_ratio <- function(polygon) {
  # Calculate the minimum bounding box
  mbr <- st_bbox(polygon)
  
  # The width and height of the MBR can be calculated from its coordinates
  width = mbr["xmax"] - mbr["xmin"]
  height = mbr["ymax"] - mbr["ymin"]
  
  # Calculate the width-to-height ratio
  ratio = width / height
  
  return(ratio)
}
custom_crown_metrics <- function(z, i) { 
  metrics <- list(
    z_max = max(z),   # max height
    z_sd = sd(z),     # vertical variability of points
    i_mean = mean(i), # mean intensity
    i_max  = max(i)   # max intensity
  )
  return(metrics) # output
}

calculate_ratio_df <- function(df) {
  df %>%
    rowwise() %>%
    mutate(width_to_height_ratio = calculate_ratio(geometry))
}

## Local Maximum Filter with variable windows size
# Function for Deciduous Trees
ws_deciduous <- function(H) {
  return(3.09632 + 0.00895* (H^2))
}


ws_2nd_polynomial <- function(H) {
  result <- (6.2299 + 1.1495 * H - 0.0105 * H^2) / 2
  
  # Replace NA values with 1 (ensuring minimum window size is 1)
  result[is.na(result)] <- 1
  
  # Ensure all values are at least 1
  result[result < 1] <- 1
  
  return(result)
}

### Function to extract crowns from Lidar data
extract_crowns <- function(las_clip, bbox) {
  print("extracting crowns...")
  las_aoi <- clip_roi(las_clip, bbox)

  # If no LAS point is present, skip
  if (dim(las_aoi@data)[1] < 10){
    print("extracting crowns... NO LAS present, return...")
    return()
  }
  print("extracting crowns... LAS present!")

  # point by a 20 cm radius circlfe of 8 points
  #chm_p2r_05 <- rasterize_canopy(las_aoi, 0.5, p2r(subcircle = 0.2), pkg = "terra")
  
  chm_pitfree_subcirlce <- tryCatch({
    rasterize_canopy(las_aoi, res = 0.5, pitfree( thresholds = c(0, 2, 5, 10, 15), subcircle = 0.15))
  }, error = function(e) {
    print(paste("Error in rasterization:", e$message))
    return(NULL)
  })
  
  # Check if rasterization failed and handle accordingly
  if (is.null(chm_pitfree_subcirlce)) {
    print("Skipping due to rasterization failure.")
    return(NULL) 
  }
  
  #chm_pitfree_subcirlce <- rasterize_canopy(las_aoi, res = 0.5, pitfree( thresholds = c(0, 2, 5, 10, 15), subcircle = 0.15))
  # Calculate focal ("moving window") values for each cell: smoothing steps with a median filter
  #kernel <- matrix(1,3,3)
  #chm_p2r_05_smoothed <- terra::focal(chm_p2r_05, w = kernel, fun = median, na.rm = TRUE)
  
  ## Postprocessing CHM: try to remove traffic signs and lamps
  chm_pitfree_subcirlce[chm_pitfree_subcirlce < 5] <- NA
  # Create a binary raster (e.g., threshold > 0 for tree areas)
  binary_chm <- chm_pitfree_subcirlce > 5
  
  # Identify connected components (clusters)
  patches_chm <- patches(binary_chm, directions = 8, zeroAsNA = FALSE)
  
  # Calculate patch sizes
  patch_sizes <- freq(patches_chm) # Frequency table of patch IDs
  small_patches <- patch_sizes$value[patch_sizes$count <= 5] # IDs of small patches
  
  if (length(small_patches) > 0) {
    small_patches_mask <- patches_chm %in% small_patches
    chm_pitfree_subcirlce_cleaned <- mask(chm_pitfree_subcirlce, small_patches_mask, maskvalue = TRUE)
  } else {
    chm_pitfree_subcirlce_cleaned <- chm_pitfree_subcirlce
  }
  
  ccm = ~custom_crown_metrics(z = Z, i = Intensity)

 
  ttops_pitfree_subcirlce_cleaned <- locate_trees(chm_pitfree_subcirlce_cleaned, lmf(ws = ws_2nd_polynomial, hmin = 5,  ws_args = list("Z")))
  
  # Use dalponte algorithm
  
  # algo_dalponte <- dalponte2016(chm_pitfree_subcirlce_cleaned, ttops_pitfree_subcirlce_cleaned,   th_tree = 2,    # Minimum tree height
  #                               th_seed = 0.45, # Seed threshold for initial growth
  #                               th_cr = 0.65,   # Crown merging threshold
  #                               max_cr = 30)    # Max crown diameter (20 pixels = 10m for 0.5m CHM)
  # las_dalponte <- segment_trees(las_aoi, algo_dalponte)
  # crowns_dalponte <- tryCatch({
  #   crown_metrics(las_dalponte, func = ccm, geom = "concave")
  # }, error = function(e) {
  #   print("No crowns detected. Adjust parameters.")
  #   return(data.frame())
  # })
  
  # Use Silva algorithm
  algo_silva <- silva2016(chm_pitfree_subcirlce_cleaned, ttops_pitfree_subcirlce_cleaned, max_cr = 0.6, exclusion = 0.3)
  las_silva <- segment_trees(las_aoi, algo_silva)
  crowns_silva <- crown_metrics(las_silva, func = ccm, geom = "concave")
  crowns_silva <- tryCatch({
    crown_metrics(las_silva, func = ccm, geom = "concave")
  }, error = function(e) {
    print("No crowns detected. Adjust parameters.")
    return(data.frame())
  })
  #ttops_chm_p2r_05_smoothed <- locate_trees(chm_p2r_05_smoothed, lmf(8))
  #algo <- silva2016(chm_p2r_05_smoothed, ttops_chm_p2r_05_smoothed)
  #las_tree <- segment_trees(las_aoi, algo)
  #crowns <- crown_metrics(las_tree, func = .stdtreemetrics, geom = "convex")
  #crowns <- crowns[crowns$convhull_area > 10,]
  #crowns <- calculate_ratio_df(crowns)
  #crowns <- crowns[crowns$width_to_height_ratio > 0.5 & crowns$width_to_height_ratio < 1.5 ,]
  #ccm = ~custom_crown_metrics(z = Z, i = Intensity)
  #crowns <- crown_metrics(las_tree, func = ccm, geom = "concave")
  #crowns <- crowns[crowns$z_sd > 0.5,]
  #crowns <- calculate_ratio_df(crowns)
  #crowns <- crowns[crowns$width_to_height_ratio > 0.5 & crowns$width_to_height_ratio < 1.5 ,]
  if (is.null(crowns_silva) || nrow(crowns_silva) == 0) {
    print("No crowns detected. Adjust parameters.")
    return(NULL)
  }
  
  if (!inherits(crowns_silva, "sf")) {
    print("Extracted crowns are not an sf object. Returning NULL.")
    return(NULL)
  }
  return(crowns_silva)
}

### Function to process all .tif files in a given directory
# process_tif_files <- function(dir_path, crowns_path) {
#   files <- list.files(path = dir_path, pattern = "\\.tif$", full.names = TRUE, recursive = FALSE)
#   
#   lapply(files, function(x) {
#     print(x)
#     ras <- st_as_sf(read_stars(x))
#     st_crs(ras) <- 25833
#     ext <- st_bbox(ras)
#     crowns <- extract_crowns(las_clip, ext)
#     
#     if (!is.null(crowns)) {
#       filename <- strsplit(sub(".*/", "", x), "\\.")[[1]][1]
#       st_write(crowns, paste(crowns_path, filename, ".geojson", sep = ""), delete_dsn = T)
#     }
#   })
# }
### Function to process all .tif files in a given directory
process_tif_files <- function(dir_path, crowns_path, las_files) {
  tif_files <- list.files(path = dir_path, pattern = "\\.tif$", full.names = TRUE, recursive = FALSE)
  lapply(tif_files, function(tif_file) {
    print(tif_file)
    
    #if (tif_file != "G:/Meine Ablage/data/386_5818/sliced_imgs_2020S/3dm_33_386_5819_1_be_nobuild_134.tif"){
    #  return(NULL)
    #}
    ras <- st_as_sf(read_stars(tif_file))
    st_crs(ras) <- 25833
    ext <- st_bbox(ras)
    
    # Extract base name of the tif file
    tif_base_name <- strsplit(basename(tif_file), "_be_")[[1]][1]
    
    # Find the matching las file
    las_file <- las_files[grep(tif_base_name, las_files)]
    las_clip <- readLAS(las_file)
    
    # Only consider crowns taller than 5 m
    las_clip <- filter_poi(las_clip, Z >= 0)
    #chm <- rasterize_canopy(las_clip, 1, p2r())
    crowns <- extract_crowns(las_clip, ext)
    
    if (!is.null(crowns)) {
      filename <- strsplit(sub(".*/", "", tif_file), "\\.")[[1]][1]
      st_write(crowns, file.path(crowns_path, paste0(filename, ".geojson")), delete_dsn = TRUE)
    }
  })
}
#debug: "G:/Meine Ablage/data/386_5818/sliced_imgs_2020S/3dm_33_386_5819_1_be_nobuild_134.tif"
### Processing
#process_tif_files(tif_directory, crowns_path)
process_tif_files(tif_directory, crowns_path, las_files)

