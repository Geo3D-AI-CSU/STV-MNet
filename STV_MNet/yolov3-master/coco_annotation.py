# *_*coding: utf-8 *_*
# Author --LiMing--

import os
import random
import shutil
import time

def copyFile(fileDir, origion_path1, class_name):
    name = class_name# Get the name of the first image in order
    path = origion_path1# Label path
    image_list = os.listdir(fileDir)  # Get all images in the original image path
    image_number = len(image_list)
    train_number = int(image_number * train_rate)# Calculate the number of training images needed
    test_number = int(image_number * test_rate)# Calculate the number of validation images needed
    train_sample = random.sample(image_list, train_number)  # Randomly select 0.8 fraction of images from image_list.
    test_sample = random.sample(list(set(image_list) - set(train_sample)), test_number)
    val_sample = list(set(image_list ) - set(train_sample) - set(test_sample))
    # test_sample = list(set(image_list) - set(train_sample))# Keep the remaining images in the test set

    sample = [train_sample, test_sample, val_sample]# Generate three lists
    # Copy images to the target directory
    for k in range(len(save_dir)):# Length of addresses, currently k is two values, 0 and 1
        if os.path.isdir(save_dir[k]) and os.path.isdir(save_dir1[k]):# Check if the path exists
            for name in sample[k]:# Loop through train_sample data in order
                name1 = name[0:-4] + '.txt'# Split the string, e.g., 1927.jpg to '1927.txt'
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))# Join to concatenate strings
                shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))# Copy to copy
        else:
            os.makedirs(save_dir[k])# Create image path
            os.makedirs(save_dir1[k])# Create label path
            for name in sample[k]:
                name1 = name[0:-4] + '.txt'
                shutil.copy(os.path.join(fileDir, name), os.path.join(save_dir[k], name))
                shutil.copy(os.path.join(path, name1), os.path.join(save_dir1[k], name1))
        print('ok')

if __name__ == '__main__':
    time_start = time.time()

    # Original dataset path
    origion_path = r'E:\syc\StreetView\yolov3-master\labels\png/'# Image path
    origion_path1 = r'E:\syc\StreetView\yolov3-master\labels\txt/'# Label path

    # Save paths
    save_train_dir = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\train\images/'# Training set image path
    save_test_dir = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\test\images/'# Test set image path
    save_val_dir = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\val\images/'  # Validation set image path
    save_train_dir1 = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\train\labels/'# Training set label path
    save_test_dir1 = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\test\labels/'# Test set label path
    save_val_dir1 = r'E:\syc\StreetView\yolov3-master\labels\MyCOCO\val\labels/'  # Validation set label path
    save_dir = [save_train_dir, save_test_dir, save_val_dir]
    save_dir1 = [save_train_dir1, save_test_dir1, save_val_dir1]

    # Training set ratio
    train_rate = 0.8
    test_rate = 0.2

    # Dataset classes and numbers
    file_list = os.listdir(origion_path)
    num_classes = len(file_list)
    for i in range(num_classes):
        class_name = file_list[i]
        copyFile(origion_path, origion_path1, class_name)
    print('Split completed!')
    time_end = time.time()
    print('---------------')
    print('The time taken to split the training and test sets is %s!' % (time_end - time_start))
