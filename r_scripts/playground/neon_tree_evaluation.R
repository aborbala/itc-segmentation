library(raster)
library(dplyr)
library(NeonTreeEvaluation)



download()
#files <- NeonTreeEvaluation::download()
#print(files)

field_polygons <- list_field_crowns()

# Get the path to the RGB data for a specific plot
rgb_path<-get_data(plot_name = "SJER_059_2018",type="rgb")
rgb<-stack(rgb_path)

# Get the path to the annotations file
xml <- get_data("SJER_052_2018", "annotations")

# Parse the XML annotations
annotations <- xml_parse(xml)

# Convert the bounding box annotations to spatial polygons
ground_truth_polygons <- boxes_to_spatial_polygons(annotations, img)