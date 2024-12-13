# This is the code to create inventory automatically with pre-processed images from YOLOv3 and semantic segmentation with cityscape pretrained model.
AutomaticCode <-
    function(RawImage, YoloImage, YoloTable, SemsegImage, processDir, VH=2.5){
        print("into function")
      # Install and load essential packages for analysis
        pacman::p_load("imager", "tidyverse", "foreach", "segmented", "rmarkdown", "pander", "strucchange")
        print("Line8")
        # Extract Metadata from the file name of the Raw Image
        FileName <- gsub(".json.jpg", "", tail(unlist(strsplit(RawImage, "/")), 1))
        print(FileName)
        # Cutting strings by “_”
        parts <- strsplit(FileName, "_")[[1]]
        
        # get information from name of the file
        Latitude_Longitude <- strsplit(parts[2], ",")[[1]]
        Latitude <- Latitude_Longitude[2]
        Longitude <- Latitude_Longitude[1]
        Heading_end <- strsplit(parts[4], "_")[[1]][1]
        Heading <- substr(Heading_end, 1, nchar(Heading_end)-4)
        print(Heading)
        # build DataFrame
        MetaData <- data.frame(
          Latitude = Latitude,
          Longitude = Longitude,
          Heading = Heading,
          pitch = 0
        )
        colnames(MetaData) <- c("Latitude", "Longitude", "Heading","pitch")
        print(MetaData)
        print("Line13")
        # Load images
        
        Orig_img <- load.image(RawImage)     # original image for image size
        Yolo_img <- load.image(YoloImage)     # original image for image size   Load the resultant image of YOLO target detection
        SmSg_img <- load.image(SemsegImage)   # Image from semantic segmentation

        # Load location of trees
        # (Bounding box from YOLO (left top right bottom class))
        Yolo_bbox <- read.csv(YoloTable)   #Yolo image ID，left，top，right，bottom，class
        
        # Information of original pictures
        W <- nrow(Orig_img) # Width of the picture
        H <- ncol(Orig_img) # Height of the picture
        VP <- (H+1)/2       # View point of the picture (!H+1 is required to point out the middel line of the image)
        print("Line26")
        # Required functions
        # Split image with imsub
        HorSpliter <- function(Img, BBox){ Img %>% imsub(x >= BBox[1] & x <= BBox[3])} #Crop img vertically according to the bounding box
        VerSpliter <- function(Img, BBox){ Img %>% imsub(y >= BBox[4] & y <= BBox[2])} #Crop img horizontally according to the bounding box

        
        TreeExtractor <-
          function(Img){
            IO <- Img
            R(IO)[which(!R(IO)[] * 255 == 128)] <- NA#The dark red part of the image is trees, extracted by color
            G(IO)[which(!G(IO)[] * 255 == 0)] <- NA
            B(IO)[which(!B(IO)[] * 255 == 0)] <- NA
            return(IO)
          }
        


        BaseDetector <-
          function(Img){
            
            TreeOnly_col <- as.numeric(colSums(R(Img), na.rm=T)) %>% ifelse(. < 1, NA, .)
            # Check if there are any non-NA values
            if(all(is.na(TreeOnly_col))) {
              return(c(NA, NA, NA, NA))  # If all values are NA, return an NA vector directly
            }
            # Find the minimum index of non-zero pixels, which is the tree's bottom position
            TreeOnly_min <- ((TreeOnly_col * 0 + 1) * c(1:H)) %>% min(.,na.rm=T)
            # Find the maximum index of non-zero pixels, which is the tree's top position
            TreeOnly_max <- ((TreeOnly_col * 0 + 1) * c(1:H)) %>% max(.,na.rm=T)
            # Set the maximum index of the tree as the base point
            Base         <- TreeOnly_max
            # Set the distance between the base point and the tree's top position to zero
            Diff         <- 0       # Adding one to include the pixel cell at the base point
            # Return the base point, distance, minimum index of the tree, and maximum index of the tree
            return(c(Base, Diff, TreeOnly_min, TreeOnly_max))

          }
        
        
        # Image analysis tools
        StructureAnalyzer <- 
            function(out, H){
                print("进入StructureAnalyzer")
                # print(out)
                # Remove the row with zero vegetation pixels
                out.nonzero <- out[out[]>0]
                # print(out.nonzero)
                out.df      <- data.frame(PixW = out.nonzero)
                # print(out.df)
                out.df$PixH <- c(1:nrow(out.df)) 
                # print(out.df)
                cellNumber  <- as.numeric(gsub("X", "", row.names(out.df)))#get rid of X,cellNumber is the number of the row
                # Check the first change point of the plot
                
                #Detection of Incomplete Trees in Rejected Images
                print(length(out.df$PixW)>10)
                if (length(out.df$PixW)>10){
                    # --- Mutation Point Detection (Change Point Detection) ---

                    # Perform F-statistic test for change point detection in 'PixW' data.
                    # This assumes a model where 'PixW' is modeled by a constant (intercept only).
                    out.cp <- strucchange::Fstats(PixW ~ 1, data=out.df) 

                    # Calculate the vertical position of the center of the crown (VP). 
                    # It's assumed that 'H' represents the total height of the analyzed structure.
                    VP       <- (H + 1) / 2

                    # Get the maximum width (Tree_W) of 'PixW' values, representing the crown width.
                    Tree_W   <- max(out.df$PixW, na.rm=TRUE) # Crown width
                    # print(Tree_W)  # (This is a commented-out print statement)

                    # Calculate the distance in cells from the cell number to the crown center (VP).
                    CellToVP <- ceiling(abs(cellNumber - VP))
                    # print(CellToVP)  # (This is a commented-out print statement)

                    # Calculate the total tree height (Tree_H) using a trigonometric calculation.
                    # This assumes a certain geometric relationship and utilizes 'CellToVP' and 'H'.
                    Tree_H   <- sum(1/cos(asin(CellToVP/(H/2))), na.rm=TRUE)  # Tree Height
                    # print(Tree_H)  # (This is a commented-out print statement)

                    # Extract the breakpoint (mutation point) identified by the F-statistic test.
                    BreakPt  <- out.cp$breakpoint

                    # Create a plot visualizing the structure analysis
                    #  Plots 'PixW' against 'PixH' with lines, color-coded in blue.
                    #  Adds labels for axes and a title for the plot.
                    plot(out.df$PixH, out.df$PixW, type="b", col="blue", xlab="CelltoBase", ylab="PixW", main="Structure Analyzer Plot")

                    # Display the breakpoint location
                    # Adds a vertical red dashed line at the detected breakpoint.
                    abline(v = out.cp$breakpoint, col = "red", lty = 2)

                    # Calculate tree height below the crown (Tree_BH), using the breakpoint
                    # It assumes a geometric relationship and utilizes 'CellToVP' and 'H' 
                    # but only considering cells up to the breakpoint.
                    Tree_BH  <- sum(1/cos(asin(CellToVP[1:BreakPt]/(H/2))), na.rm=TRUE) # Height below crown.


                    # --- Calculate DBH (Diameter at Breast Height) ---
                    # DBH is assumed to be a representative width below the crown (here taken as the 10th percentile).
                    # The next code block conditionally calculates DBH based on a comparison of Tree_BH and data lengths.
                    if (Tree_BH < length(out.df$PixW)) {
                        # DBH is calculated by finding the 10th percentile of 'PixW' values from index 1 to Tree_BH
                        Tree_DBH <- quantile(out.df$PixW[1:Tree_BH], 0.1, na.rm = TRUE) 
                        print("Tree_DBH*******************************************************") # Print a debugging message
                        print(Tree_DBH) # Print the calculated DBH

                        # Create a data frame to store the calculated tree metrics.
                        M <- data.frame(Tree_H, Tree_W, Tree_BH, Tree_DBH, TreeTop = min(cellNumber))
                        print(M) # Print the metrics data frame
                        return(M) # Return the metric data frame
                    }

                    
                  }
                }
                

        # Combined function
        # VH: height of the camera
        TreeInventory  <- 
            function(Img, VH = 2.5, LocInfo = TreeLoc){ # VH of GSV is known as 2.5m 8.2 feet
              print("进入TreeInventory")
                W <- nrow(Img) # Width of the picture
                H <- ncol(Img) # Height of the picture
                f <- 0.54
                
                VP <- (H+1)/2      # View point of the picture
                print(nrow(LocInfo))
                BM <- foreach(i = 1:nrow(LocInfo), .combine=rbind) %do% {
                      print(i)
                      print(as.numeric(LocInfo[i,c(2:5)]))
                      plot(Img)

                      HorSplitImg     <- HorSpliter(Img, as.numeric(LocInfo[i,c(2:5)]))
                      # print(dim(HorSplitImg))
                      plot(HorSplitImg)
                      # Check if the cropped image is empty
                      
                      IndTreeOnly <- TreeExtractor(HorSplitImg)
                      # print(dim(IndTreeOnly))
                          
                      BaseDetector    <- BaseDetector(IndTreeOnly)
                      print("BaseDetector")
                      print(BaseDetector)
                      # Check that BaseDetector is all NA before processing it
                      panduan <- all(is.na(BaseDetector))
  
                      print(!panduan)
                      if (!panduan  ) {
                        Base            <- BaseDetector[1]
                        Diff            <- BaseDetector[2]              
                        TreeTop         <- BaseDetector[3]             
                        TreeBottom      <- BaseDetector[4]
                        # Calculate the pixel width of the tree
                        BaseToVP        <- ceiling(abs((Base - VP))) # Calculate the distance from each tree pixel to the center line 
                        # # print("BaseTOVP")
                        
                        #calculateWr
                        # #c(1:BaseToVP)对应公式中Di，1/cos(asin(c(1:BaseToVP)/(H/2)))对应Cv,i
                        CorrectedPixSum <- sum(1/cos(asin(c(1:BaseToVP)/(H/2))), na.rm=T)-BaseToVP
                        print(CorrectedPixSum)

                        PixelWidth      <- VH/CorrectedPixSum#Wr,DBH
                        print(PixelWidth)
                        # plot(IndTreeOnly)
                        
                        #Horizontal cutting of images
                        TreeFit         <- VerSpliter(IndTreeOnly, as.numeric(LocInfo[i,c(2:5)]))
                        # plot(TreeFit)
                        # 1. Select the data of the first dimension of TreeFit
                        subset_data <- TreeFit[,,1]
                        print("subset_data***********************************************************")
                        print(all(is.na(subset_data)))
                        if (!all(is.na(subset_data))){
                          # 2. Convert the rounded data into a data frame
                          data_frame_data <- data.frame(ceiling(subset_data))
                          print("data_frame_data")
                          # 3. Apply the sum function to each column of the data frame, ignoring missing values
                          column_sums <- sapply(data_frame_data, function(x) sum(x, na.rm = TRUE))
                          # 4. Reverse the results
                          PixelCnt <- rev(column_sums)
                          # PixelCnt       <- rev(sapply(data.frame(ceiling(TreeFit[,,1])), FUN = function(x){sum(x, na.rm=T)}))
                          # Get row numbers from the bottom to the top of the tree (modified)
                          Begin <- as.integer(LocInfo$bottom[i])+1
                          End <- as.integer(LocInfo$top[i])
                          a <- c(LocInfo$bottom[i]:LocInfo$top[i])
                          if (length(c(LocInfo$bottom[i]:LocInfo$top[i])) > length(PixelCnt)){
                            count = length(c(LocInfo$bottom[i]:LocInfo$top[i])) - length(PixelCnt)
                            a <- c(LocInfo$bottom[i]:(LocInfo$top[i] - count))
                          }
                          # print(length(a))
                          names(PixelCnt) <- paste0("X", a) # Record the column count for each row, with row numbers named according to their position in the original image
                          print("PixelCnt*******************************************************************************")

                          print(all(is.na(PixelCnt))) # http://127.0.0.1:8597/graphics/plot_zoom_png?width=1200&height=900
                          # print(i)
                          PixelStr        <- StructureAnalyzer(PixelCnt, H)
                          print("StructureAnalyzer completed!")
                          print(PixelStr)
                          # Ensure the result does not contain NA values
                          if (!is.null(PixelStr)){
                            result          <- (PixelStr %>% mutate(Tree_H = Tree_H, Tree_BH = Tree_BH)) * PixelWidth #tree_H*Wr
                            result$TreeTop  <- PixelStr$TreeTop
                            result$ID       <- i
                            print(result)
                            return(result)
                          }
                        }

          
                        }
                        
                      }
                      
                }
                
            }

        # structure of the tree
        Tree.df <- TreeInventory(Img = SmSg_img, VH = 2.5, LocInfo = Yolo_bbox)
        print("TreeInventory执行完毕！")
        
          
        # Location of the tree
        Tree.Inv <- data.frame(Tree.df, Yolo_bbox[Tree.df$ID,])
        print(all(is.na(Tree.Inv)))
        
        if (!all(is.na(Tree.Inv))){
          Tree.Inv <- Tree.Inv %>%
            # The field of view (FOV) of Google street view is known as 127 degree and 63.5 is the half of the FOV.
            mutate(Dist  = abs(Tree_H - VH)/tan(asin(ceiling(abs(TreeTop-(H+1)/2))/(H/2)))) %>%
            # Horizontally, 1 pixel denotes the W/360 degree
            mutate(Angle = (as.numeric(MetaData$Heading) + (0.5 * (left + right) - W/2) * 360/W) %%360) %>%
            # Polar coordinate to cartesian coordinate
            mutate(rel_x = Dist * cos(Angle), rel_y = Dist * sin(Angle)) %>%
            # mutate(c_lat = as.numeric(MetaData$Latitude), c_lon = as.numeric(MetaData$Longitude))
            mutate(c_lat = MetaData$Latitude, c_lon = MetaData$Longitude)
          
          print(substr(FileName, 1, nchar(FileName)-4))
          write.csv(Tree.Inv, paste0(processDir, substr(FileName, 1, nchar(FileName)-4), "_result.csv") , row.names=F)
          return(Tree.Inv)
        }
        
    }
#mydataset
# # Specify the paths to your images and files

# Specify the path to the CSV folder
# The CSV files are generated from the bounding boxes predicted by the network, in a specific format.
root <- "E:/Suyingcai/STV_MNet"
inputData<-  paste0(root,"/data/input data/Structure/")
csv_folder <-  paste0(inputData,"csv")
print(file.exists(csv_folder))

# Get the list of CSV files
csv_files <- list.files(csv_folder, pattern = "\\.csv$", full.names = TRUE)

processDir0 <- paste0(root,"/results/Structure calculation/results0.1/")
VH <- 2.5

# Iterate over each CSV file
for (csv_file in csv_files) {
  # Construct file paths related to the filename
  csv_file_name <- basename(csv_file)
  csv_file_name_no_ext <- tools::file_path_sans_ext(csv_file_name)
  YoloTable0 <- csv_file
  print(YoloTable0)
  
  # Original street view image
  RawImage0 <- paste0("E:/Suyingcai/changsha/changsha.zip/changsha/", csv_file_name_no_ext, ".jpg")
  print(RawImage0)
  
  # Network predicted street view image
  YoloImage0 <- paste0(root,"/results/STV_MNet/predict_changsha/", csv_file_name_no_ext, ".jpg")
  print(YoloImage0)
  
  # Network predicted mask image
  SemsegImage0 <- paste0(root,"/data/input data/Structure/mask/", csv_file_name_no_ext, ".png")
  print(SemsegImage0)
  
  
  file.exists(RawImage0)
  file.exists(YoloImage0)
  file.exists(YoloTable0)
  file.exists(SemsegImage0)
  result_file <- paste0(processDir0, csv_file_name_no_ext, "_resultTest.csv")
  print(result_file)
  
  if (file.exists(result_file)) {
    print("File exists")
  } else {
  
    # Call the function
    result <- AutomaticCode(RawImage0, YoloImage0, YoloTable0, SemsegImage0, processDir0, VH)
    
    # Print the result if needed
    print(result)
  }
}
