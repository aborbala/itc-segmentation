library(sf)
library(lidR)
library(tools)
library(osmdata)
library(ggplot2)
library(terra)

# ---- CONFIGURATION ----
#aoi_code <- "386_5818" # training data
aoi_code <- "384_5816" # test data

FETCH_STRUCTURES = TRUE
ELIMINATE_STRUCTURES = TRUE
MASKING = FALSE

crs <-  25833
# Handle both German ("Meine Ablage") and English ("My Drive") Drive names
drive_options <- c("H:/Meine Ablage", "H:/My Drive")
existing_drive <- drive_options[dir.exists(drive_options)][1]
if (is.na(existing_drive)) stop("No Google Drive mount found at expected paths.")

# Define base paths
base_data_path <- file.path(existing_drive, "masterthesis", "data")
aoi_base_path <- file.path(base_data_path, aoi_code)

# Dynamic path definitions
las_folder_path <- file.path(aoi_base_path, "LAS")
buildings_path <- file.path(aoi_base_path, "buildings.gpkg")
bridges_output_path <- file.path(aoi_base_path, "bridges.gpkg")
merged_structures_path <- file.path(aoi_base_path, "structures.gpkg")
output_folder_path <- file.path(aoi_base_path, "LAS_no_buildings")
output_veg_mask_folder_path <- file.path(aoi_base_path, "LAS_no_buildings_veg_mask")
output_no_struct_veg_mask_folder_path <- file.path(aoi_base_path, "LAS_no_structures_veg_mask")
men_made_output_path <- file.path(aoi_base_path, "men_made.gpkg")
veg_mask_path <- file.path(aoi_base_path, "vegetation_mask", "veg_mask_buffered.gpkg")
aoi_tif <- file.path(aoi_base_path, "DOP", sprintf("truedop20cir_%s_2_be_2020.tif", aoi_code))

# Create output directories if needed
if (!dir.exists(output_folder_path)) dir.create(output_folder_path, recursive = TRUE)
if (!dir.exists(output_veg_mask_folder_path)) dir.create(output_veg_mask_folder_path, recursive = TRUE)
if (!dir.exists(output_no_struct_veg_mask_folder_path)) dir.create(output_no_struct_veg_mask_folder_path, recursive = TRUE)

# ---- FUNCTIONS ----
# Function to remove buildings from Lidar data
process_las_with_masking   <- function(merged_structures_path,
                                       las_path,
                                       veg_mask_path,
                                       eliminate_structures = ELIMINATE_STRUCTURES,
                                       masking = MASKING
                                       ) {
  las <- readLAS(las_path, filter = "-drop_z_below 0")
  las_check(las)
  st_crs(las) <- crs
  
  # Normalization (subtract the DTM)
  gnd <- filter_ground(las)
  dtm <- rasterize_terrain(las, 1, knnidw())
  nlas <- normalize_height(las, knnidw())
  # Define bounding box geometry
  las_bbox <- st_as_sfc(st_bbox(las))  
  
  final_mask <- las_bbox
  
  #structures <- st_read(merged_structures_path, quiet = TRUE)
  #st_crs(structures) <- 25833
  
  if (eliminate_structures) {
    structures <- st_read(merged_structures_path, quiet = TRUE)
    st_crs(structures) <- crs
    structures_aoi <- st_geometry(st_crop(structures, las_bbox))
    final_mask <- st_difference(final_mask, st_union(structures_aoi))
  }

  # If masking is enabled, intersect with vegetation mask
  if (masking  && !is.null(veg_mask_path)) {
    veg_mask <- st_read(veg_mask_path, quiet = TRUE)
    st_crs(veg_mask) <- 25833
    veg_mask_aoi <- st_geometry(st_crop(veg_mask, las_bbox))
    final_mask <- st_intersection(final_mask, veg_mask_aoi)
  }
  
  # Clip using the final constructed mask
  las_clip <- clip_roi(nlas, final_mask)
  return(las_clip)
}

get_osm_structures <- function() { 
  local_crs <- crs
  aoi_rast <- terra::rast(aoi_tif)
  aoi_poly <- st_as_sfc(st_bbox(aoi_rast), crs = crs)
  aoi_wgs84 <- st_transform(aoi_poly, 4326)
  bbox_wgs84 <- st_bbox(aoi_wgs84)
  
  # Convert bounding box to WGS 84 (EPSG:4326) for OSM query
  #bbox_local <- st_bbox(buildings)
  #bbox_wgs84 <- st_transform(st_as_sfc(bbox_local, crs = local_crs), crs = 4326)
  #bbox_wgs84 <- st_bbox(bbox_wgs84)  # Convert to bbox format
  
  # Query OSM for all man_made features within the bounding box
  query_man_made <- osmdata::opq(bbox = c(bbox_wgs84["xmin"], bbox_wgs84["ymin"], bbox_wgs84["xmax"], bbox_wgs84["ymax"])) %>%
    osmdata::add_osm_feature(key = "man_made")
  
  # Query OSM for specific bridge types within the bounding box
  query_bridges <- osmdata::opq(bbox = c(bbox_wgs84["xmin"], bbox_wgs84["ymin"], bbox_wgs84["xmax"], bbox_wgs84["ymax"])) %>%
    osmdata::add_osm_feature(key = "bridge", value = c("viaduct", "yes", "aqueduct", "beam", "suspension", "cantilever"))
  
  # Retrieve OSM data
  osm_man_made <- osmdata::osmdata_sf(query_man_made)
  osm_bridges <- osmdata::osmdata_sf(query_bridges)
  
  # Extract geometries only (lines and polygons)
  geometries_list <- list()
  
  if (!is.null(osm_man_made$osm_lines) && nrow(osm_man_made$osm_lines) > 0) {
    geometries_list <- append(geometries_list, list(st_geometry(osm_man_made$osm_lines)))
  }
  if (!is.null(osm_man_made$osm_polygons) && nrow(osm_man_made$osm_polygons) > 0) {
    geometries_list <- append(geometries_list, list(st_geometry(osm_man_made$osm_polygons)))
  }
  
  if (!is.null(osm_bridges$osm_lines) && nrow(osm_bridges$osm_lines) > 0) {
    geometries_list <- append(geometries_list, list(st_geometry(osm_bridges$osm_lines)))
  }
  if (!is.null(osm_bridges$osm_polygons) && nrow(osm_bridges$osm_polygons) > 0) {
    geometries_list <- append(geometries_list, list(st_geometry(osm_bridges$osm_polygons)))
  }
  
  # If no geometries were found, return NULL
  if (length(geometries_list) == 0) {
    warning("No man_made features or bridges found in the bounding box.")
    return(NULL)
  }
  
  # Combine geometries into a single sf object (dropping attributes)
  structures_geom <- do.call(c, geometries_list)
  
  # Transform to match the local CRS
  structures_geom <- st_transform(structures_geom, crs = local_crs)
  
  # Dissolve (union) all geometries into one
  structures_union <- st_union(structures_geom)
  
  # Convert back to an sf object (geometry only)
  structures_union_sf <- st_sf(geometry = structures_union, crs = local_crs)
  
  return(structures_union_sf)
}

get_osm_buildings <- function() { 
  # Convert bounding box to WGS 84 (EPSG:4326) for OSM query
  #bbox_local <- st_bbox(buildings)
  #bbox_wgs84 <- st_transform(st_as_sfc(bbox_local, crs = local_crs), crs = 4326)
  #bbox_wgs84 <- st_bbox(bbox_wgs84)  # Convert to bbox format
  local_crs <- crs
  aoi_rast <- terra::rast(aoi_tif)
  aoi_poly <- st_as_sfc(st_bbox(aoi_rast), crs = crs)
  aoi_wgs84 <- st_transform(aoi_poly, 4326)
  bbox_wgs84 <- st_bbox(aoi_wgs84)
  
  # Query OSM for buildings within the bounding box
  query_buildings <- osmdata::opq(bbox = c(bbox_wgs84["xmin"], bbox_wgs84["ymin"], bbox_wgs84["xmax"], bbox_wgs84["ymax"])) %>%
    osmdata::add_osm_feature(key = "building")
  
  # Retrieve OSM data
  osm_buildings <- osmdata::osmdata_sf(query_buildings)
  
  # Extract geometries only (lines and polygons)
  geometries_list <- list()
  
  if (!is.null(osm_buildings$osm_lines) && nrow(osm_buildings$osm_lines) > 0) {
    geometries_list <- append(geometries_list, list(st_geometry(osm_buildings$osm_lines)))
  }
  if (!is.null(osm_buildings$osm_polygons) && nrow(osm_buildings$osm_polygons) > 0) {
    geometries_list <- append(geometries_list, list(st_geometry(osm_buildings$osm_polygons)))
  }
  
  # If no geometries were found, return NULL
  if (length(geometries_list) == 0) {
    warning("No buildings found in the bounding box.")
    return(NULL)
  }
  
  # Combine geometries into a single sf object (dropping attributes)
  buildings_geom <- do.call(c, geometries_list)
  
  # Transform to match the local CRS
  buildings_geom <- st_transform(buildings_geom, crs = local_crs)
  
  # Dissolve (union) all geometries into one
  buildings_union <- st_union(buildings_geom)
  
  # Convert back to an sf object (geometry only)
  buildings_union_sf <- st_sf(geometry = buildings_union, crs = local_crs)
  
  return(buildings_union_sf)
}


# Folder paths
#las_folder_path <- "H:/My Drive/masterthesis/data/386_5818/LAS"
#buildings_path <- "H:/My Drive/masterthesis/data/386_5818/buildings.gpkg"
#bridges_output_path <- "H:/My Drive/masterthesis/data/386_5818/bridges.gpkg"
#merged_structures_path <- "H:/My Drive/masterthesis/data/386_5818/structures.gpkg"
#output_folder_path <- "H:/My Drive/masterthesis/data/386_5818/LAS_no_buildings"
#output_veg_mask_folder_path <- "H:/My Drive/masterthesis/data/386_5818/LAS_no_buildings_veg_mask"
#men_made_output_path <- "H:/My Drive/masterthesis/data/386_5818/men_made.gpkg"
#veg_mask_path <- "H:/My Drive/masterthesis/data/386_5818/vegetation_mask/veg_mask_buffered.gpkg"
#if (!dir.exists(output_folder_path)) dir.create(output_folder_path)


if (FETCH_STRUCTURES == TRUE) {
  #bridges_sf <- get_bridges_osm(buildings_path)
  structures_sf <- get_osm_structures()
  
  # Load buildings data
  #buildings_sf <- st_read(buildings_path, quiet = TRUE)
  buildings_sf <- get_osm_buildings()
  # Keep only geometries (drop attributes)
  buildings_geom <- st_geometry(buildings_sf)
  bridges_geom <- st_geometry(structures_sf)
  
  # Union buildings and bridges into one geometry set
  merged_structures_geom <- st_union(buildings_geom, bridges_geom)
  
  # Convert back to sf object (without attributes)
  merged_structures <- st_sf(geometry = merged_structures_geom, crs = st_crs(buildings_sf))
  merged_structures <- st_buffer(merged_structures, 1)
  merged_structures <- st_cast(merged_structures, "MULTIPOLYGON")
  
  # Overwrite existing merged structures file
  st_write(merged_structures, merged_structures_path, quiet = TRUE, delete_layer = TRUE)
  cat("Merged structures saved to:", merged_structures_path, "\n")
}else {
  merged_structures <- st_read(merged_structures_path,
                               quiet = TRUE)  
}

# Loop through all LAS files in the folder and remove buildings
las_files <- list.files(las_folder_path, pattern = "\\.las$", full.names = TRUE)
for (las_path in las_files) {
  cat("Processing:", las_path, "\n")
  las_nobuild <- process_las_with_masking(merged_structures_path, las_path, veg_mask_path)
  # Save the resulting LAS file without buildings
  output_file_name <- paste0(file_path_sans_ext(basename(las_path)), "_nobuild.las")
  if (MASKING == TRUE){
    output_file_path <- file.path(output_no_struct_veg_mask_folder_path, output_file_name)
  }else {
    output_file_path <- file.path(output_folder_path, output_file_name)
  }
  writeLAS(las_nobuild, output_file_path)

}

### PLOTTING
# Compare osm buldings and ALKIS buildings
buildings_alkis <- st_read(buildings_path, quiet = TRUE)
buildings_osm <-get_osm_buildings(buildings_path)
buildings_alkis <- st_union(buildings_alkis)
buildings_alkis <- st_sf(geometry = buildings_alkis)

# Compute area of ALKIS and OSM datasets
area_alkis <- st_area(buildings_alkis)
area_osm <- st_area(buildings_osm)

# Compute the intersection between ALKIS and OSM
intersection <- st_intersection(buildings_alkis, buildings_osm)

# Compute intersection area
intersection_area <- st_area(intersection)

# Compute area differences
area_difference <- abs(area_alkis - area_osm)

# Compute the symmetric difference (areas that do not intersect)
symmetric_difference <- st_sym_difference(buildings_alkis, buildings_osm)
symmetric_difference_area <- st_area(symmetric_difference)

# Print results
cat("ALKIS Area:", as.numeric(area_alkis), "m²\n")
cat("OSM Area:", as.numeric(area_osm), "m²\n")
cat("Intersection Area:", as.numeric(intersection_area), "m²\n")
cat("Area Difference:", as.numeric(area_difference), "m²\n")
cat("Symmetric Difference Area:", as.numeric(symmetric_difference_area), "m²\n")

# Plot comparison
ggplot() +
  geom_sf(data = buildings_alkis, fill = "blue", alpha = 0.3, color = "blue") +
  geom_sf(data = buildings_osm, fill = "red", alpha = 0.3, color = "red") +
  geom_sf(data = intersection, fill = "green", alpha = 0.5, color = "green") +
  labs(title = "ALKIS vs OSM Buildings Comparison",
       subtitle = "Blue = ALKIS, Red = OSM, Green = Intersection") +
  theme_minimal()

