import os
import shutil

# Define a function to read a list of file names from a file
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # Remove newline characters at the end of each line and return the list of file names
    return [line.strip() for line in lines]

# Create output files

def DeeplabTXT2YOLO(yolo_txt_dir, deeplab_txt_dir, output_files):
    # Iterate over each output file
    for output_file in output_files:
        # Read the corresponding list of file names
        yolo_txt_file = os.path.join(deeplab_txt_dir, output_file)
        Lines = read_file(yolo_txt_file)
        # Write the path of each file
        with open(os.path.join(yolo_txt_dir, output_file), 'a') as f:
            for Line in Lines:
                txt_path = os.path.join(yolo_txt_dir, 'labels', output_file[:-4], Line + '.txt')
                f.write(txt_path + '\n')

    print("Completed")

# Read list of files
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# Copy files
def copy_files(source_files, dest_dir):
    for file in source_files:
        shutil.copy(file, dest_dir)

def copy_txt_png(coco_dir, source_dir, output_files):
    # Read each file list
    for file_name in output_files:
        # Get the list of file names
        file_list = read_file(os.path.join(coco_dir, file_name))
        # Copy txt files to the corresponding folder
        txt_names = [os.path.basename(file) for file in file_list]
        txt_arr = [os.path.join(source_dir, 'txt', file) for file in txt_names]
        png_arr = [os.path.join(source_dir, 'png', file[:-4] + '.jpg') for file in txt_names]
        copy_files(txt_arr, os.path.join(coco_dir, 'labels', file_name.split('.')[0]))
        # Copy images to the corresponding folder
        copy_files(png_arr, os.path.join(coco_dir, 'images', file_name.split('.')[0]))

    print("Completed")

# Set paths
ROOT = r"E:\Suyingcai\STV_MNet\code"
yoloBASEDIR = ROOT + r"\STV_MNet\yolov3-master\data\labelsSegment"
yolo_txt_dir = yoloBASEDIR + r'\MyCOCO'
deeplab_txt_dir = ROOT + r'\code\STV_MNet\deeplab\deeplearning\ImageSets\Segmentation'
output_files = ['test.txt', 'train.txt', 'trainval.txt', 'val.txt']

# Convert Deeplab TXT files to YOLO format
DeeplabTXT2YOLO(yolo_txt_dir, deeplab_txt_dir, output_files)

# Copy txt and png files
copy_txt_png(yolo_txt_dir, yoloBASEDIR, output_files)
