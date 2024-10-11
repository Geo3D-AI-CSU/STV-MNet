# This is the code to create inventory automatically with pre-processed images from YOLOv3 and semantic segmentation with cityscape pretrained model.
AutomaticCode <-
  function(RawImage,YoloTable, processDir, VH=2.5){
    print("进入函数")
    pacman::p_load("imager", "tidyverse", "foreach", "segmented", "rmarkdown", "pander", "strucchange")
    FileName <- gsub(".json.jpg", "", tail(unlist(strsplit(RawImage, "/")), 1))
    print(FileName)
    # 按照"_"切割字符串
    parts <- strsplit(FileName, "_")[[1]]
    
    # 提取信息
    Latitude_Longitude <- strsplit(parts[2], ",")[[1]]
    Latitude <- Latitude_Longitude[2]
    Longitude <- Latitude_Longitude[1]
    Heading_end <- strsplit(parts[4], "_")[[1]][1]
    Heading <- substr(Heading_end, 1, nchar(Heading_end)-4)
    print("#######################################################")
    print(Heading)
    # 构建DataFrame
    MetaData <- data.frame(
      Latitude = Latitude,
      Longitude = Longitude,
      Heading = Heading,
      pitch = 0
    )
    colnames(MetaData) <- c("Latitude", "Longitude", "Heading","pitch")
    print(MetaData)

    Yolo_bbox <- read.csv(YoloTable)   #Yolo image中每个检测对象的ID，left，top，right，bottom，class
    
    # Information of original pictures
    W <- 2048 # Width of the picture
    H <- 1024 # Height of the picture
    VP <- (H+1)/2       # View point of the picture (!H+1 is required to point out the middel line of the image)
    
    
    TreeInventory  <- 
      function( VH = 2.5, LocInfo = TreeLoc){ # VH of GSV is known as 2.5m 8.2 feet
        print("进入TreeInventory")
        W <- 2048 # Width of the picture
        H <- 1024 # Height of the picture
        VP <- (H+1)/2      # View point of the picture
        print(nrow(LocInfo))
        BM <- foreach(i = 1:nrow(LocInfo), .combine=rbind) %do% {
          print(i)
          ID <- i
          print(as.numeric(LocInfo[i,c(2:5)]))
          Base <- as.numeric(LocInfo[i,c(3)])
          BaseToVP        <- ceiling(abs((Base - VP)))    # As they have less distortion we put them just like that. (Basic Pixel width)
          CorrectedPixSum <- sum(1/cos(asin(c(1:BaseToVP)/(H/2))), na.rm=T)
          print(CorrectedPixSum)
          print(BaseToVP)
          PixelWidth      <- VH/CorrectedPixSum #Wr,算DBH    ####wr计算有问题，可考虑乘以深度/焦距，，最终基准像素宽度大概在0.6-0.8之间
          print(PixelWidth)
          result          <- data.frame(ID,PixelWidth)#
          result$PixelWidth  <- PixelWidth
          result$ID       <- i
          # 确保结果不含有NA值
          print(result)
          return(result)
            
          
        }
        
      }
    
    # structure of the tree
    Tree.df <- TreeInventory( VH = 2.5, LocInfo = Yolo_bbox)
    print("TreeInventory执行完毕！")
    print(FileName)
    out_file <- paste0(processDir, substr(FileName, 1, nchar(FileName)-4), "_result.csv")
    print(out_file)
    write.csv(Tree.df, out_file, row.names=F)
    return(Tree.df)
    
  }

root <- "E:/Suyingcai/STV_MNet"
inputData<-  paste0(root,"/data/input data/Structure/")
csv_folder <-  paste0(inputData,"csv")
print(file.exists(csv_folder))
# 获取CSV文件列表
csv_files <- list.files(csv_folder, pattern = "\\.csv$", full.names = TRUE)

processDir0 <- paste0(root,"/results/Structure calculation/results_Wr/")
VH <- 2.5
# 遍历每个CSV文件
for (csv_file in csv_files) {
  # 构建文件名相关路径
  csv_file_name <- basename(csv_file)
  csv_file_name_no_ext <- tools::file_path_sans_ext(csv_file_name)
  YoloTable0 <- csv_file
  print(YoloTable0)
  #原街景影像
  RawImage0 <- paste0("E:/Suyingcai/changsha/changsha.zip/changsha/", csv_file_name_no_ext, ".jpg")
  print(RawImage0)
  #网络预测街景图
  YoloImage0 <- paste0(root,"/results/STV_MNet/predict_changsha/", csv_file_name_no_ext, ".jpg")
  print(YoloImage0)
  #网络预测mask图
  SemsegImage0 <- paste0(root,"/data/input data/Structure/mask/", csv_file_name_no_ext, ".png")
  print(SemsegImage0)


  result_file <- paste0(processDir0, csv_file_name_no_ext, "_result.csv")
  print(result_file)
  print(file.exists(result_file))
  if (file.exists(result_file)) {
    print("文件存在")
  } else {
    
    if (file.exists(YoloTable0)){
      # Call the functiona
      result <- AutomaticCode(RawImage0,YoloTable0,  processDir0, VH)
      
      # Print the result if needed
      print(result)
      
    }
    
  }
}