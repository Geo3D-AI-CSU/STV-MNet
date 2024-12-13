# This is the code to create an inventory automatically with pre-processed images from YOLOv3 and semantic segmentation with a cityscape pretrained model.
AutomaticCode <-
  function(RawImage, YoloTable, processDir, VH = 2.5) {
    print("Entering the function")
    pacman::p_load("imager", "tidyverse", "foreach", "segmented", "rmarkdown", "pander", "strucchange")
    FileName <- gsub(".json.jpg", "", tail(unlist(strsplit(RawImage, "/")), 1))
    print(FileName)
    # Split the string by "_"
    parts <- strsplit(FileName, "_")[[1]]

    # Extract information
    Latitude_Longitude <- strsplit(parts[2], ",")[[1]]
    Latitude <- Latitude_Longitude[2]
    Longitude <- Latitude_Longitude[1]
    Heading_end <- strsplit(parts[4], "_")[[1]][1]
    Heading <- substr(Heading_end, 1, nchar(Heading_end) - 4)
    print("#######################################################")
    print(Heading)
    # Build a DataFrame
    MetaData <- data.frame(
      Latitude = Latitude,
      Longitude = Longitude,
      Heading = Heading,
      pitch = 0
    )
    colnames(MetaData) <- c("Latitude", "Longitude", "Heading", "pitch")
    print(MetaData)

    Yolo_bbox <- read.csv(YoloTable) # YOLO image: each detected object's ID, left, top, right, bottom, class

    # Information of original pictures
    W <- 2048 # Width of the picture
    H <- 1024 # Height of the picture
    VP <- (H + 1) / 2 # View point of the picture (!H+1 is required to point out the middle line of the image)


    TreeInventory  <-
      function(VH = 2.5, LocInfo = TreeLoc) { # VH of GSV is known as 2.5m 8.2 feet
        print("Entering TreeInventory")
        W <- 2048 # Width of the picture
        H <- 1024 # Height of the picture
        VP <- (H + 1) / 2 # View point of the picture
        print(nrow(LocInfo))
        BM <- foreach(i = 1:nrow(LocInfo), .combine = rbind) %do% {
          print(i)
          ID <- i
          print(as.numeric(LocInfo[i, c(2:5)]))
          Base <- as.numeric(LocInfo[i, c(3)])
          BaseToVP        <- ceiling(abs((Base - VP))) # As they have less distortion we put them just like that. (Basic Pixel width)
          CorrectedPixSum <- sum(1 / cos(asin(c(1:BaseToVP) / (H / 2))), na.rm = T)
          print(CorrectedPixSum)
          print(BaseToVP)
          PixelWidth      <- VH / CorrectedPixSum # Wr, calculate DBH  #### wr calculation has a problem, consider multiplying by depth/focal length, the final reference pixel width is about 0.6-0.8
          print(PixelWidth)
          result          <- data.frame(ID, PixelWidth) #
          result$PixelWidth  <- PixelWidth
          result$ID       <- i
          # Ensure that the result does not contain NA values
          print(result)
          return(result)
        }
      }

    # structure of the tree
    Tree.df <- TreeInventory(VH = 2.5, LocInfo = Yolo_bbox)
    print("TreeInventory execution complete!")
    print(FileName)
    out_file <- paste0(processDir, substr(FileName, 1, nchar(FileName) - 4), "_result.csv")
    print(out_file)
    write.csv(Tree.df, out_file, row.names = F)
    return(Tree.df)
  }

root <- "E:/Suyingcai/STV_MNet"
inputData <- paste0(root, "/data/input data/Structure/")
csv_folder <- paste0(inputData, "csv")
print(file.exists(csv_folder))
# Get the list of CSV files
csv_files <- list.files(csv_folder, pattern = "\\.csv$", full.names = TRUE)

processDir0 <- paste0(root, "/results/Structure calculation/results_Wr/")
VH <- 2.5
# Iterate through each CSV file
for (csv_file in csv_files) {
  # Build file name related paths
  csv_file_name <- basename(csv_file)
  csv_file_name_no_ext <- tools::file_path_sans_ext(csv_file_name)
  YoloTable0 <- csv_file
  print(YoloTable0)
  # Original street view image
  RawImage0 <- paste0("E:/Suyingcai/changsha/changsha.zip/changsha/", csv_file_name_no_ext, ".jpg")
  print(RawImage0)
  # Network predicted street view image
  YoloImage0 <- paste0(root, "/results/STV_MNet/predict_changsha/", csv_file_name_no_ext, ".jpg")
  print(YoloImage0)
  # Network predicted mask image
  SemsegImage0 <- paste0(root, "/data/input data/Structure/mask/", csv_file_name_no_ext, ".png")
  print(SemsegImage0)


  result_file <- paste0(processDir0, csv_file_name_no_ext, "_result.csv")
  print(result_file)
  print(file.exists(result_file))
  if (file.exists(result_file)) {
    print("File exists")
  } else {

    if (file.exists(YoloTable0)) {
      # Call the function
      result <- AutomaticCode(RawImage0, YoloTable0, processDir0, VH)

      # Print the result if needed
      print(result)
    }
  }
}
