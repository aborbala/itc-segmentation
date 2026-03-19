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
aoi_code <- "384_5816"

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

# Outputs
crowns_path <- file.path(aoi_base_path, "crowns_no_structures_veg_mask_Silva")

if (!dir.exists(crowns_path)) {
  dir.create(crowns_path, recursive = TRUE)
}

#' Calculate width-to-height ratio of a polygon's bounding box
#'
#' Used to filter out elongated polygons that are unlikely to be tree crowns.
#'
#' @param polygon An sf geometry object.
#' @return Numeric width-to-height ratio.
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
#' Compute custom per-crown metrics from LiDAR point cloud
#'
#' @param z Numeric vector of point heights (Z values).
#' @param i Numeric vector of point intensities.
#' @return Named list with z_max, z_sd, i_mean, i_max.
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

#' Variable window size function for deciduous tree top detection
#'
#' @param H Numeric canopy height value(s).
#' @return Window size in map units.
ws_deciduous <- function(H) {
  return(3.09632 + 0.00895* (H^2))
}


#' Variable window size function using a 2nd-degree polynomial
#'
#' @param H Numeric canopy height value(s).
#' @return Window size in map units (minimum 1).
ws_2nd_polynomial <- function(H) {
  result <- (6.2299 + 1.1495 * H - 0.0105 * H^2) / 2
  
  # Replace NA values with 1 (ensuring minimum window size is 1)
  result[is.na(result)] <- 1
  
  # Ensure all values are at least 1
  result[result < 1] <- 1
  
  return(result)
}

#' Extract tree crowns from LiDAR data within a bounding box
#'
#' Clips the point cloud to the bbox, generates a pit-free CHM, detects tree tops
#' using a local maximum filter, and segments crowns with the Silva algorithm.
#'
#' @param las_clip A LAS object (normalized heights, filtered).
#' @param bbox An sf bounding box defining the area of interest.
#' @return An sf object with crown polygons, or NULL if no crowns detected.
extract_crowns <- function(las_clip, bbox) {
  print("extracting crowns...")
  las_aoi <- clip_roi(las_clip, bbox)

  # If no LAS point is present, skip
  if (dim(las_aoi@data)[1] < 10){
    print("extracting crowns... NO LAS present, return...")
    return()
  }
  print("extracting crowns... LAS present!")

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
  
  ## Postprocessing CHM: filter out low vegetation, traffic signs, and lamps
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

  # Segment tree crowns using Silva algorithm
  algo_silva <- silva2016(chm_pitfree_subcirlce_cleaned, ttops_pitfree_subcirlce_cleaned, max_cr = 0.6, exclusion = 0.3)
  las_silva <- segment_trees(las_aoi, algo_silva)
  crowns_silva <- tryCatch({
    crown_metrics(las_silva, func = ccm, geom = "concave")
  }, error = function(e) {
    print("No crowns detected. Adjust parameters.")
    return(data.frame())
  })
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

#' Process all .tif tiles and extract tree crowns from matching LAS files
#'
#' For each TIF tile, finds the corresponding LAS file by name, clips to
#' the tile extent, extracts crowns, and saves results as GeoJSON.
#'
#' @param dir_path Directory containing .tif tile images.
#' @param crowns_path Output directory for crown GeoJSON files.
#' @param las_files Character vector of full paths to LAS files.
process_tif_files <- function(dir_path, crowns_path, las_files) {
  tif_files <- list.files(path = dir_path, pattern = "\\.tif$", full.names = TRUE, recursive = FALSE)
  lapply(tif_files, function(tif_file) {
    print(tif_file)
    
    ras <- st_as_sf(read_stars(tif_file))
    st_crs(ras) <- 25833
    ext <- st_bbox(ras)
    
    # Extract base name of the tif file
    tif_base_name <- strsplit(basename(tif_file), "_be_")[[1]][1]
    
    # Find the matching las file
    las_file <- las_files[grep(tif_base_name, las_files)]
    las_clip <- readLAS(las_file)
    
    las_clip <- filter_poi(las_clip, Z >= 0)
    crowns <- extract_crowns(las_clip, ext)
    
    if (!is.null(crowns)) {
      filename <- strsplit(sub(".*/", "", tif_file), "\\.")[[1]][1]
      st_write(crowns, file.path(crowns_path, paste0(filename, ".geojson")), delete_dsn = TRUE)
    }
  })
}

process_tif_files(tif_directory, crowns_path, las_files)

