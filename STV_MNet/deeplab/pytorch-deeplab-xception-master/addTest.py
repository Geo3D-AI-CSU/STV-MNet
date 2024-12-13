import os
import shutil

# Define the source and target folder paths
source_root = r'E:\Suyingcai\changsha\survey\select'  # Actual six road data
target_root = r'E:\Suyingcai\STV_MNet\code\STV_MNet\deeplab\deeplearning'

# Iterate through each road folder
for road_folder in ["fuyuan", "donger", "lunan", "xiangzhang", "youyi", "guqu"]:
    road_folder_path = os.path.join(source_root, road_folder)
    jpeg_images_path = os.path.join(road_folder_path, "JPEGImages")
    segmentation_class_path = os.path.join(road_folder_path, "SegmentationClass")

    # Append image names to the test.txt file
    test_txt_path = os.path.join(target_root, "ImageSets", "Segmentation", "test.txt")
    with open(test_txt_path, "a") as test_txt_file:
        for filename in os.listdir(jpeg_images_path):
            if filename.endswith(".jpg"):
                test_txt_file.write(os.path.splitext(filename)[0] + "\n")

    # Copy files from JPEGImages and SegmentationClass to the target folder
    target_jpeg_images_path = os.path.join(target_root, "JPEGImages")
    target_segmentation_class_path = os.path.join(target_root, "SegmentationClass")

    os.makedirs(target_jpeg_images_path, exist_ok=True) # Create the directory if it doesn't exist
    os.makedirs(target_segmentation_class_path, exist_ok=True) # Create the directory if it doesn't exist


    for filename in os.listdir(jpeg_images_path):
        shutil.copy(os.path.join(jpeg_images_path, filename), os.path.join(target_jpeg_images_path, filename))

    for filename in os.listdir(segmentation_class_path):
        shutil.copy(os.path.join(segmentation_class_path, filename), os.path.join(target_segmentation_class_path, filename))
