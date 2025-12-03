# RandCrown for preprocessing
library(fs)

# Assuming all .tif and .las files are named consistently
tif_paths <- dir_ls("H:/My Drive/masterthesis/data/test", recurse = TRUE, regexp = "\\.tif$")
las_paths <- dir_ls("H:/My Drive/masterthesis/data/test", recurse = TRUE, regexp = "\\.las$")

# Optional: ensure they are sorted/matching
result_df <- tibble::tibble()

evaluate_randcrowns <- function(predictions, groundtruth, iou_threshold = 0.3) {
  scores <- c()
  
  predictions <- st_make_valid(predictions)
  groundtruth <- st_make_valid(groundtruth)
  
  for (i in seq_len(nrow(groundtruth))) {
    target <- groundtruth[i, ]
    best_score <- 0
    
    for (j in seq_len(nrow(predictions))) {
      pred <- predictions[j, ]
      
      if (st_is_empty(pred) || st_is_empty(target)) next
      
      inter <- st_intersection(pred, target)
      union <- st_union(pred, target)
      
      area_union <- as.numeric(st_area(union))
      area_inter <- as.numeric(st_area(inter))
      
      if (length(area_union) == 1 && area_union > 0 &&
          length(area_inter) == 1) {
        
        iou_val <- area_inter / area_union
        
        if (!is.na(iou_val) && iou_val > iou_threshold) {
          rc_score <- compute_randcrowns(pred, target)
          if (rc_score > best_score) {
            best_score <- rc_score
          }
        }
      }
      
    }
    
    scores <- c(scores, best_score)
  }
  
  return(scores)
}


for (i in seq_along(tif_paths)) {
  test <- tif_paths[i]
  las_nobuild_path <- las_paths[i]
  
  # Load data
  ras <- st_as_sf(read_stars(test))
  st_crs(ras) <- 25833
  ext <- st_bbox(ras)
  
  las_nobuild <- readLAS(las_nobuild_path)
  las_unfiltered <- clip_roi(las_nobuild, ext)
  las_unfiltered <- filter_poi(las_unfiltered, Z < 40)
  las_above5 <- filter_poi(las_unfiltered, Z >= 5)
  
  if (nrow(las_above5@data) < 5) next
  
  # CHM
  chm <- chm_pitfree_subcirlce_cleaned  # replace this with your real CHM for that tile
  
  # Top trees: use cadaster here
  ttops <- tree_cadaster_buffer
  colnames(ttops) <- c("Z", "geom")
  ttops$treeID <- 1:nrow(ttops)
  ttops <- ttops[, c("treeID", "Z", "geom")]
  
  # Dalponte
  algo_dalponte <- dalponte2016(chm, ttops, th_tree = 2, th_seed = 0.45, th_cr = 0.65, max_cr = 35)
  las_dalponte <- segment_trees(las_unfiltered, algo_dalponte)
  crowns_dalponte <- crown_metrics(las_dalponte, func = ccm, geom = "concave")
  
  # Dalponte Convex
  crowns_dalponte_convex <- crown_metrics(las_dalponte, func = ccm, geom = "convex")
  
  # Silva
  algo_silva <- silva2016(chm, ttops, max_cr = 0.6, exclusion = 0.3)
  las_silva <- segment_trees(las_unfiltered, algo_silva)
  crowns_silva <- crown_metrics(las_silva, func = ccm, geom = "concave")
  
  # Evaluate
  s1 <- evaluate_randcrowns(crowns_dalponte, tree_cadaster_buffer)
  s2 <- evaluate_randcrowns(crowns_dalponte_convex, tree_cadaster_buffer)
  s3 <- evaluate_randcrowns(crowns_silva, tree_cadaster_buffer)
  
  # Store results
  result_df <- bind_rows(
    result_df,
    tibble(method = "Dalponte", tile = basename(test), score = s1),
    tibble(method = "Dalponte_convex", tile = basename(test), score = s2),
    tibble(method = "Silva", tile = basename(test), score = s3)
  )
}

library(ggplot2)

ggplot(result_df %>% filter(score > 0), aes(x = method, y = score, fill = method)) +
  geom_boxplot(alpha = 0.7) +
  geom_jitter(width = 0.2, alpha = 0.4) +
  labs(title = "RandCrowns Score by Method", x = "Method", y = "Score") +
  theme_minimal() +
  theme(legend.position = "none")

