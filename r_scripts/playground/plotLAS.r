library(lidR)
library(rgl)
library(viridis)

# Set working directory
setwd(dirname(rstudioapi::getSourceEditorContext()$path))

# ---- CONFIGURATION ----
#aoi_code <- "386_5818"  # training data
#aoi_code <- "384_5816"   # training data

# Handle both German ("Meine Ablage") and English ("My Drive") Drive names
drive_options <- c("G:/Meine Ablage", "H:/My Drive")
existing_drive <- drive_options[dir.exists(drive_options)][1]
if (is.na(existing_drive)) stop("No Google Drive mount found at expected paths.")

#las_nobuild_path <- "C:/tree-canopy/data/400_5816/LAS_no_buildings"

las_nobuild_path <- "G:/Meine Ablage/masterthesis/data/386_5818/LAS"
las_files <- list.files(path = las_nobuild_path, pattern = "\\.las$", full.names = TRUE, recursive = FALSE)

las <- readLAS(las_files[1])

#plot(las, color = "Z", size = 1, bg = "white")
#plot(las, color = "Z", size = 1, bg = "white", pal = viridis(10))


# Filter out classes
las_ground <- filter_poi(las, Classification == 2)
plot(las_ground, color = "Z", size = 1, bg = "white", pal = mako(10))



las_low_veg <- filter_poi(las, Classification == 3)
plot(las_low_veg, color = "Z", size = 1, bg = "white", pal = mako(10))

las_mid_veg <- filter_poi(las, Classification == 4)
plot(las_mid_veg, color = "Z", size = 1, bg = "white", pal = mako(10))

las_high_veg <- filter_poi(las, Classification == 5)
plot(las_high_veg, color = "Z", size = 1, bg = "white", pal = mako(10))
