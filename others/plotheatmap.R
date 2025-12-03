# --- 1. Load Required Libraries ---
library(ggplot2)
library(dplyr)
library(tidyr)
library(showtext)
library(viridis)

#TODO: see how th plots from python look like and try to adjust R


# --- 2. Load the Montserrat Font ---
font_add_google("Montserrat", "Montserrat")
showtext_auto()

red_color <- "#ec3c3a"

# --- 3. Create the Data Frame with Your Results ---
# results_df <- data.frame(
#   Model = c("1. Baseline", 
#             "2. High-Certainty", 
#             "3. Ignore-Region", 
#             "4. High-Cert + Custom Loss", 
#             "5. Baseline + Custom Loss"),
#   P_25 = c(0.556, 0.629, 0.620, 0.722, 0.501),
#   R_25 = c(0.797, 0.678, 0.746, 0.651, 0.837),
#   F1_25 = c(0.655, 0.653, 0.677, 0.684, 0.627),
#   P_50 = c(0.428, 0.506, 0.527, 0.590, 0.383),
#   R_50 = c(0.614, 0.546, 0.634, 0.532, 0.641),
#   F1_50 = c(0.504, 0.525, 0.575, 0.560, 0.480),
#   RandCrowns = c(0.625, 0.510, 0.579, 0.501, 0.651)
# )

# --- 3. Create the Data Frame with Your Results ---
results_df <- data.frame(
  Model = c("1. Baseline", 
            "2. High-Certainty", 
            "3. Ignore-Region", 
            "4. High-Cert + Custom Loss", 
            "5. Baseline + Custom Loss"),
  P_50 = c(0.720, 0.729, 0.704, 0.763, 0.667),
  R_50 = c(0.604, 0.531, 0.604, 0.551, 0.581),
  F1_50 = c(0.657, 0.615, 0.650, 0.640, 0.621),
  RandCrowns = c(0.577, 0.496, 0.564, 0.520, 0.614)
)

# --- 4. Reshape the Data and Reverse Y-Axis Order ---
long_results <- results_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = gsub("_", "@0.", Metric)) %>%
  mutate(Model = factor(Model, levels = rev(unique(Model))))

# --- 5. Create a separate data frame for the highest values ---
max_scores <- long_results %>%
  group_by(Metric) %>%
  filter(Score == max(Score))

# --- 6. Create the Heatmap Plot ---
heatmap_plot <- ggplot(long_results, aes(x = Metric, y = Model)) +
  # Main heatmap tiles with color fill and white borders
  geom_tile(aes(fill = Score), color = "white", lwd = 1) +
  
  # --- NEW: Add a second geom_tile layer just for the highlight border ---
  geom_tile(data = max_scores, fill = NA, color = red_color, lwd = 1) +
  
  # Use the 'mako' color palette
  scale_fill_viridis_c(option = "mako", 
                       direction = -1, 
                       name = "Score",
                       limits = c(min(long_results$Score), 1.0)) +
  
  # Add the text labels
  geom_text(aes(label = sprintf("%.3f", Score)), color = "white", size = 3.5, family = "Montserrat") +
  geom_text(data = max_scores, aes(label = sprintf("%.3f", Score)), 
            color = "white", size = 3.5, family = "Montserrat", fontface = "bold") +
  
  # Apply a clean theme and set the font
  theme_minimal() +
  theme(
    text = element_text(family = "Montserrat"),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"), 
    axis.text.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, hjust = 0.5, margin = margin(b = 10)),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 20)),
    legend.position = "right",
    panel.grid = element_blank()
  ) +
  
  # Add titles
   labs(
     title = "Comparison of Deep Learning Model Performance",
     subtitle = "Confidence Threshold = 0.1"
   )

# --- 7. Display the Plot ---
print(heatmap_plot)

# --- 8. Save the Plot to a File ---
ggsave("model_performance_heatmap_highlighted.png", plot = heatmap_plot, width = 10, height = 6, dpi = 120, bg = "white")



# --- 1. Prepare the Data ---
# The 'check.names = FALSE' argument allows for spaces and symbols in the column names.
# lidar_results_df <- data.frame(
#   Method = c("Dalponte", "Silva"),
#   `RandCrowns` = c(0.525, 0.553),
#   `F1-Score @ 0.5` = c(0.498, 0.527),
#   `Precision @ 0.5` = c(0.495, 0.525),
#   `Recall @ 0.5` = c(0.502, 0.529),
#   check.names = FALSE
# )

lidar_results_df <- data.frame(
  Method = c("Dalponte", "Silva"),
  `RandCrowns` = c(0.511, 0.569),
  `F1-Score @ 0.5` = c(0.549, 0.544),
  `Precision @ 0.5` = c(0.631, 0.543),
  `Recall @ 0.5` = c(0.485, 0.545),
  check.names = FALSE
)

# --- 4. Reshape the Data and Set Y-Axis Order ---
# Convert data from wide to long format for ggplot2 and reverse the y-axis order.
long_lidar_results <- lidar_results_df %>%
  pivot_longer(cols = -Method, names_to = "Metric", values_to = "Score") %>%
  mutate(Method = factor(Method, levels = rev(unique(Method))))

# --- 5. Find the Highest Score for Each Metric for Highlighting ---
max_lidar_scores <- long_lidar_results %>%
  group_by(Metric) %>%
  filter(Score == max(Score))

# --- 6. Create the Heatmap Plot ---
lidar_heatmap_plot <- ggplot(long_lidar_results, aes(x = Metric, y = Method)) +
  # Main heatmap tiles with color fill and white borders
  geom_tile(aes(fill = Score), color = "white", lwd = 1) +
  
  # Add a second layer for the highlight border around the best scores
  geom_tile(data = max_lidar_scores, fill = NA, color = red_color, lwd = 1) +
  
  # Use the 'mako' color palette from viridis
  scale_fill_viridis_c(option = "mako", 
                       direction = -1, 
                       name = "Score",
                       # --- FIX APPLIED HERE ---
                       # Set the color scale limits from the minimum score up to 1.0
                       limits = c(0.0, 1.0)) +
  
  # Add the text labels for all scores
  geom_text(aes(label = sprintf("%.3f", Score)), color = "white", size = 3.5, family = "Montserrat") +
  # Add bold text labels for the highest scores
  geom_text(data = max_lidar_scores, aes(label = sprintf("%.3f", Score)), 
            color = "white", size = 3.5, family = "Montserrat", fontface = "bold") +
  
  # Apply the clean, minimal theme and set fonts
  theme_minimal() +
  theme(
    text = element_text(family = "Montserrat"),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
    axis.text.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, hjust = 0.5, margin = margin(b = 10)),
    legend.position = "right",
    panel.grid = element_blank()
  ) +
  
  # Add title
  labs(
    title = "Performance Comparison of LiDAR Segmentation Methods"
  )

# --- 7. Display the Plot ---
print(lidar_heatmap_plot)

# --- 8. Save the Plot to a File ---
ggsave("lidar_performance_heatmap_final.png", plot = lidar_heatmap_plot, width = 8, height = 4, dpi = 120, bg = "white")


################################################################
################################################################
################################################################

# --- 3. Create the Full Data Frame with All Experiments ---
results_df <- data.frame(
  `Experiment Label` = c("A. Balanced Weights", 
                         "B. Prioritize LACE (Low LR)",
                         "C. Prioritize LACE (High LR)", 
                         "D. Prioritize Dice",
                         "E. Dice Only",
                         "F. LACE Only"),
  
  `lr0` = c(0.001, 0.001, 0.0025, 0.0025, 0.0025, 0.0025),
  
  `LACE Weight` = c(1.0, 0.7, 0.7, 0.3, 0.0, 1.0),
  `Dice Weight` = c(1.0, 0.3, 0.3, 0.7, 1.0, 0.0),
  
  `F1-Score @ 0.50` = c(0.621,  0.576, 0.611, 0.660, 0.670, 0.667),
  `Precision @ 0.50` = c(0.667,  0.587, 0.615,0.670, 0.703, 0.699),
  `Recall @ 0.50` = c(0.581,  0.566, 0.607, 0.650, 0.640, 0.637),
  
  
  `RandCrowns` = c(0.614, 0.512, 0.610, 0.632, 0.605, 0.585),
  
  check.names = FALSE
)

#   `F1-Score @ 0.25` = c(0.825, 0.822, 0.813, 0.831, 0.794, 0.798),


# --- 4. Reshape Data and Define Column Order for Plotting ---
long_results <- results_df %>%
  pivot_longer(cols = -`Experiment Label`, names_to = "Metric", values_to = "Score") %>%
  mutate(
    DataType = ifelse(Metric %in% c("lr0", "LACE Weight", "Dice Weight"), "Hyperparameter", "Performance Metric")
  )

# Define the exact order of columns and rows
metric_order <- c("lr0", "LACE Weight", "Dice Weight", 
                  "F1-Score @ 0.25", "F1-Score @ 0.50", 
                  "Precision @ 0.50", "Recall @ 0.50", "RandCrowns")
experiment_order <- rev(c("A. Balanced Weights", 
                          "B. Prioritize LACE (Low LR)",
                          "C. Prioritize LACE (High LR)", 
                          "D. Prioritize Dice",
                          "E. Dice Only",
                          "F. LACE Only"))
long_results <- long_results %>%
  mutate(
    Metric = factor(Metric, levels = metric_order),
    `Experiment Label` = factor(`Experiment Label`, levels = experiment_order)
  )

# --- 5. Find the Highest Score for Each *Performance Metric* for Highlighting ---
max_scores <- long_results %>%
  filter(DataType == "Performance Metric") %>%
  group_by(Metric) %>%
  filter(Score == max(Score))

# --- 6. Create the Heatmap Plot ---
# MODIFICATION: The 'label' aesthetic is now mapped directly to the 'Score' column.
hyperparameter_heatmap <- ggplot(long_results, aes(x = Metric, y = `Experiment Label`, label = Score)) +
  
  # Neutral background tiles - ONLY for hyperparameters
  geom_tile(data = . %>% filter(DataType == "Hyperparameter"), 
            fill = "#EFF1F2", color = "white", lwd = 1) +
  
  # Heatmap tiles - ONLY for performance metrics
  geom_tile(data = . %>% filter(DataType == "Performance Metric"), 
            aes(fill = Score), color = "white", lwd = 1) +
  
  # Add a vertical line to visually separate the two sections
  geom_vline(xintercept = 3.5, color = "white", lwd = 1) +
  
  # Highlight border for the best performance scores
  geom_tile(data = max_scores, fill = NA, color = red_color, lwd = 1) +
  
  # Text for hyperparameters (black)
  geom_text(data = . %>% filter(DataType == "Hyperparameter"), 
            color = "black", size = 3.5, family = "Montserrat") +
  
  # Text for performance metrics (white)
  geom_text(data = . %>% filter(DataType == "Performance Metric"), 
            color = "white", size = 3.5, family = "Montserrat") +
  
  # Bold text for the highest scores
  geom_text(data = max_scores, fontface = "bold",
            color = "white", size = 3.5, family = "Montserrat") +
  
  # Use the 'mako' color palette, scaled to 1.0
  scale_fill_viridis_c(option = "mako", direction = -1, name = "Score", limits = c(0.4, 1.0)) +
  
  # Apply the clean theme
  theme_minimal(base_family = "Montserrat") +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
    axis.text.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, hjust = 0.5, margin = margin(b = 15)),
    legend.position = "right",
    panel.grid = element_blank()
  ) +
  # Add title
  labs(title = "Hyperparameter Tuning Results for Custom Loss Model")

# --- 7. Display the Plot ---
print(hyperparameter_heatmap)

# --- 8. Save the Plot to a File ---
ggsave("hyperparameter_heatmap_final.png", plot = hyperparameter_heatmap, width = 10, height = 5, dpi = 120, bg = "white")

###################################################
###################################################
###################################################
# --- 3. Create the Data Frame with Your Results ---
# results_df <- data.frame(
#   Model = c("1. Final Model",
#             "2. DeepForest",
#             "3. Dalponte"),
# 
#   # Metrics at IoU >= 0.25
#   P_25 = c(0.834, 0.831, 0.850),
#   R_25 = c(0.812, 0.647, 0.637),
#   F1_25 = c(0.823, 0.727, 0.728),
# 
#   # Metrics at IoU >= 0.50
#   P_50 = c(0.641, 0.667, 0.631),
#   R_50 = c(0.644, 0.521, 0.485),
#   F1_50 = c(0.643, 0.585, 0.549),
# 
#   # Metrics at IoU >= 0.75
#   P_75 = c(0.178, 0.198, 0.167),
#   R_75 = c(0.178, 0.155, 0.129),
#   F1_75 = c(0.177, 0.174, 0.146),
# 
#   # Custom Metric
#   RandCrowns = c(0.637, 0.492, 0.511)
# )
results_df <- data.frame(
  Model = c("1. Final Model",
            "2. DeepForest",
            "3. Dalponte"),

  # Metrics at IoU >= 0.50
  P_50 = c(0.641, 0.667, 0.631),
  R_50 = c(0.644, 0.521, 0.485),
  F1_50 = c(0.643, 0.585, 0.549),
  
  # Custom Metric
  RandCrowns = c(0.637, 0.492, 0.511)
)
# --- 4. Reshape the Data and Reverse Y-Axis Order ---
long_results <- results_df %>%
  pivot_longer(cols = -Model, names_to = "Metric", values_to = "Score") %>%
  mutate(Metric = gsub("_", "@0.", Metric)) %>%
  mutate(Model = factor(Model, levels = rev(unique(Model))))

# --- 5. Create a separate data frame for the highest values ---
max_scores <- long_results %>%
  group_by(Metric) %>%
  filter(Score == max(Score))

# --- 6. Create the Heatmap Plot ---
heatmap_plot <- ggplot(long_results, aes(x = Metric, y = Model)) +
  # Main heatmap tiles with color fill and white borders
  geom_tile(aes(fill = Score), color = "white", lwd = 1) +
  
  # --- NEW: Add a second geom_tile layer just for the highlight border ---
  geom_tile(data = max_scores, fill = NA, color = red_color, lwd = 1) +
  
  # Use the 'mako' color palette
  scale_fill_viridis_c(option = "mako", 
                       direction = -1, 
                       name = "Score",
                       limits = c(min(long_results$Score), 1.0)) +
  
  # Add the text labels
  geom_text(aes(label = sprintf("%.3f", Score)), color = "white", size = 3.5, family = "Montserrat") +
  geom_text(data = max_scores, aes(label = sprintf("%.3f", Score)), 
            color = "white", size = 3.5, family = "Montserrat", fontface = "bold") +
  
  # Apply a clean theme and set the font
  theme_minimal() +
  theme(
    text = element_text(family = "Montserrat"),
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"), 
    axis.text.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, hjust = 0.5, margin = margin(b = 10)),
    plot.subtitle = element_text(size = 12, hjust = 0.5, margin = margin(b = 20)),
    legend.position = "right",
    panel.grid = element_blank()
  ) +
  
  # Add titles
  labs(
    title = "Comparison of Silva and Dalponte Algorithm",
  )

# --- 7. Display the Plot ---
print(heatmap_plot)

# --- 8. Save the Plot to a File ---
ggsave("model_performance_heatmap_highlighted.png", plot = heatmap_plot, width = 10, height = 2, dpi = 120, bg = "white")

########################################
########################################
########################################

# --- 3. Create the Data Frame with Your Per-Category Results ---
category_df <- data.frame(
  Category = c("Size: Small", "Size: Medium", "Size: Large", "Context: Isolated", "Context: Clumped"),
  `F1-Score` = c(0.222, 0.588, 0.710, 0.767, 0.595),
  `Recall` = c(0.176, 0.506, 0.763, 0.800, 0.519),
  `Precision` = c(0.300, 0.702, 0.664, 0.737, 0.695),
  `RandCrowns` = c(0.180, 0.444, 0.705, 0.698, 0.476),
  check.names = FALSE
)

# --- 4. Reshape Data for Plotting ---
# Convert data from wide to long format for ggplot2
long_results <- category_df %>%
  pivot_longer(cols = -Category, names_to = "Metric", values_to = "Score") %>%
  # Set the order of categories for a more logical plot layout
  mutate(
    Category = factor(Category, levels = rev(c("Size: Small", "Size: Medium", "Size: Large", "Context: Isolated", "Context: Clumped"))),
    Metric = factor(Metric, levels = c("F1-Score", "Recall", "Precision", "RandCrowns"))
  )

# --- 5. Find the Highest Score for Each Metric for Highlighting ---
max_scores <- long_results %>%
  group_by(Metric) %>%
  filter(Score == max(Score))

# --- 6. Create the Heatmap Plot ---
category_heatmap <- ggplot(long_results, aes(x = Metric, y = Category, fill = Score)) +
  
  # Main heatmap tiles with color fill and white borders
  geom_tile(color = "white", lwd = 1) +
  
  # Add a second layer for the highlight border around the best scores
  geom_tile(data = max_scores, fill = NA, color = red_color, lwd = 1) +
  
  # Add the numeric labels inside each cell
  geom_text(aes(label = sprintf("%.3f", Score)), 
            color = "white", 
            size = 3.5, 
            family = "Montserrat") +
  
  # Add bold labels for the highest scores
  geom_text(data = max_scores, aes(label = sprintf("%.3f", Score)), 
            color = "white", size = 3.5, family = "Montserrat", fontface = "bold") +
  
  # Use the 'mako' color palette, with the scale going up to 1.0
  scale_fill_viridis_c(option = "mako", direction = -1, name = "Score", limits = c(0, 1.0)) +
  
  # Apply a clean theme and set fonts
  theme_minimal(base_family = "Montserrat") +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
    axis.text.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, hjust = 0.5, margin = margin(b = 15)),
    panel.grid = element_blank(),
    legend.position = "right"
  ) +
  
  # Add title
  labs(title = "Model Performance by Tree Category")

# --- 7. Display the Plot ---
print(category_heatmap)

# --- 8. Save the Plot to a File ---
ggsave("per_category_heatmap.png", plot = category_heatmap, width = 8, height = 5, dpi = 120, bg = "white")


############################################################
############################################################
## Category, and size evaluation


# --- 3. Create the Data Frame with Your Per-Category Results ---
# This data matches your "Table Y"
category_df <- data.frame(
  Category = c("Size: Small", "Size: Medium", "Size: Large", "Context: Isolated", "Context: Clumped"),
  `F1-Score` = c(0.222, 0.588, 0.710, 0.767, 0.595),
  `Recall` = c(0.176, 0.506, 0.763, 0.800, 0.519),
  `Precision` = c(0.300, 0.702, 0.664, 0.737, 0.695),
  `RandCrowns Score` = c(0.180, 0.444, 0.705, 0.698, 0.476),
  check.names = FALSE # Prevents R from changing column names (e.g., "F1-Score" to "F1.Score")
)

# --- 4. Reshape Data for Plotting ---
# Convert data from wide to long format for ggplot2
long_results <- category_df %>%
  pivot_longer(cols = -Category, names_to = "Metric", values_to = "Score") %>%
  # Set the order of categories for a more logical plot layout
  mutate(
    Category = factor(Category, levels = rev(c("Size: Small", "Size: Medium", "Size: Large", "Context: Isolated", "Context: Clumped"))),
    Metric = factor(Metric, levels = c("F1-Score", "Recall", "Precision", "RandCrowns Score"))
  )

# --- 5. Find the Highest Score for Each Metric for Highlighting ---
max_scores <- long_results %>%
  group_by(Metric) %>%
  filter(Score == max(Score))

# --- 6. Create the Heatmap Plot ---
category_heatmap <- ggplot(long_results, aes(x = Metric, y = Category, fill = Score)) +
  
  # Main heatmap tiles with color fill and white borders
  geom_tile(color = "white", lwd = 1) +
  
  # Add a second layer for the highlight border around the best scores
  # This finds the 'max_scores' data frame and draws invisible tiles with a red border
  geom_tile(data = max_scores, fill = NA, color = red_color, lwd = 1) +
  
  # Add the numeric labels inside each cell
  geom_text(aes(label = sprintf("%.3f", Score)),  
            color = "white",  
            size = 3.5,  
            family = "Montserrat") +
  
  # Add bold labels *only* for the highest scores
  # This plots them on top of the regular labels
  geom_text(data = max_scores, aes(label = sprintf("%.3f", Score)),  
            color = "white", size = 3.5, family = "Montserrat", fontface = "bold") +
  
  # Use the 'mako' color palette, with the scale going up to 1.0
  scale_fill_viridis_c(option = "mako", direction = -1, name = "Score", limits = c(0, 1.0)) +
  
  # Apply a clean theme and set fonts
  theme_minimal(base_family = "Montserrat") +
  theme(
    axis.title = element_blank(), # Remove axis titles
    axis.text.x = element_text(angle = 45, hjust = 1, size = 10, face = "bold"),
    axis.text.y = element_text(size = 10, face = "bold"),
    plot.title = element_text(size = 12, hjust = 0.5, margin = margin(b = 15)),
    panel.grid = element_blank(),
    legend.position = "right"
  )

# --- 7. Display the Plot ---
print(category_heatmap)

# --- 8. Save the Plot to a File ---
ggsave("per_category_heatmap.png", plot = category_heatmap, width = 8, height = 5, dpi = 120, bg = "white")


############################################################
############################################################
# BAR CHART
# --- 3. Create the Data Frame ---
category_df <- data.frame(
  Category = c("Size: Small", "Size: Medium", "Size: Large", "Context: Isolated", "Context: Clumped"),
  `F1-Score` = c(0.222, 0.588, 0.710, 0.767, 0.595),
  `Recall` = c(0.176, 0.506, 0.763, 0.800, 0.519),
  `Precision` = c(0.300, 0.702, 0.664, 0.737, 0.695),
  `RandCrowns Score` = c(0.180, 0.444, 0.705, 0.698, 0.476),
  check.names = FALSE
)

# --- 4. Reshape Data for Plotting ---
long_results <- category_df %>%
  pivot_longer(cols = -Category, names_to = "Metric", values_to = "Score") %>%
  mutate(
    # Reverse order for Y-axis so "Small" is at the top or bottom as preferred
    Category = factor(Category, levels = rev(c("Size: Small", "Size: Medium", "Size: Large", "Context: Isolated", "Context: Clumped"))),
    Metric = factor(Metric, levels = c("F1-Score", "Recall", "Precision", "RandCrowns Score"))
  )

# --- 5. Create the Balloon Plot (Bubble Heatmap) ---
balloon_plot <- ggplot(long_results, aes(x = Metric, y = Category)) +
  
  # Add light grid lines to guide the eye
  geom_tile(fill = "transparent", color = "gray95", lwd = 0.5) +
  
  # Create the "Balloons"
  # Size matches magnitude, Color matches magnitude
  geom_point(aes(size = Score, fill = Score), shape = 21, color = "white", stroke = 1) +
  
  # Add numeric labels centered on the bubbles
  geom_text(aes(label = sprintf("%.2f", Score)), 
            color = "white", 
            size = 3.5, 
            family = "Montserrat", 
            fontface = "bold") +
  
  # Size Scale: Control the min and max size of bubbles
  scale_size_continuous(range = c(8, 18), guide = "none") +
  
  # Color Scale: Use 'mako' palette
  scale_fill_viridis_c(option = "mako", direction = -1, begin = 0.2, end = 0.9, name = "Score", limits = c(0, 1)) +
  
  # Theme Customization
  theme_minimal(base_family = "Montserrat") +
  theme(
    axis.title = element_blank(),
    axis.text.x = element_text(size = 11, face = "bold", margin = margin(t = 10)), # Move X labels down slightly
    axis.text.y = element_text(size = 11, face = "bold"),
    plot.title = element_text(size = 16, face = "bold", hjust = 0.5, margin = margin(b = 20)),
    panel.grid = element_blank(), # We drew our own grid lines above
    legend.position = "right",
    legend.title = element_text(face = "bold")
  ) +
  
  # Add Title
  labs(title = "Model Performance by Tree Category")

# --- 6. Display the Plot ---
print(balloon_plot)

# --- 7. Save the Plot ---
ggsave("per_category_balloonplot.png", plot = balloon_plot, width = 10, height = 6, dpi = 300, bg = "white")

