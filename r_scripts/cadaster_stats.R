
library(sf)
library(ggplot2)

# Set the path to your GeoPackage file.
file_path <- "G:/Meine Ablage/masterthesis/data/386_5818/trees.gpkg"

trees_data <- NULL
tryCatch({
  trees_data <- st_read(file_path)
  cat("File successfully loaded.\n")
}, error = function(e) {
  cat("Error reading file: ", e$message, "\n")
})


if (!is.null(trees_data)) {
  
  # Check if the required column 'kronendurch' exists
  if ("kronedurch" %in% names(trees_data)) {
    
    # 0%, 25%, 50% (median), 75%, and 100% (max) quantiles
    # na.rm = TRUE removes any NA values before calculation, which is important
    kronendurch_quantiles <- quantile(trees_data$kronedurch, na.rm = TRUE)
    
    print(kronendurch_quantiles)

    boxplot_kronendurch <- ggplot(trees_data, aes(y = kronedurch)) +
      geom_boxplot(fill = "skyblue", color = "black", alpha = 0.7, outlier.colour = "red") +
      labs(
        title = "Boxplot of Crown Diameter (Kronendurchmesser)",
        y = "Crown Diameter (meters)",
        x = "" # Hide x-axis label as it's not needed
      ) +
      theme_minimal() +
      theme(
        plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
        axis.title.y = element_text(size = 12),
        axis.text.x = element_blank(), # Hide x-axis text
        axis.ticks.x = element_blank()  # Hide x-axis ticks
      )
    
    print(boxplot_kronendurch)
  }
}
