import os
import shutil

# 定义源文件夹和目标文件夹路径
source_root = r'E:\Suyingcai\changsha\survey\select'#实测六条道路
target_root = r'E:\Suyingcai\STV_MNet\code\STV_MNet\deeplab\deeplearning'

# 遍历每个道路文件夹
for road_folder in ["fuyuan", "donger", "lunan", "xiangzhang", "youyi", "guqu"]:
    road_folder_path = os.path.join(source_root, road_folder)
    jpeg_images_path = os.path.join(road_folder_path, "JPEGImages")
    segmentation_class_path = os.path.join(road_folder_path, "SegmentationClass")

    # 将图片名追加到 test.txt 文件中
    test_txt_path = os.path.join(target_root, "ImageSets", "Segmentation", "test.txt")
    with open(test_txt_path, "a") as test_txt_file:
        for filename in os.listdir(jpeg_images_path):
            if filename.endswith(".jpg"):
                test_txt_file.write(os.path.splitext(filename)[0] + "\n")

    # 复制 JPEGImages 和 SegmentationClass 中的文件到目标文件夹中
    target_jpeg_images_path = os.path.join(target_root, "JPEGImages")
    target_segmentation_class_path = os.path.join(target_root, "SegmentationClass")

    os.makedirs(target_jpeg_images_path, exist_ok=True)
    os.makedirs(target_segmentation_class_path, exist_ok=True)

    for filename in os.listdir(jpeg_images_path):
        shutil.copy(os.path.join(jpeg_images_path, filename), os.path.join(target_jpeg_images_path, filename))

    for filename in os.listdir(segmentation_class_path):
        shutil.copy(os.path.join(segmentation_class_path, filename), os.path.join(target_segmentation_class_path, filename))
