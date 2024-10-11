
import os
import shutil

# 定义函数来读取文件中的文件名列表
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    # 移除每行末尾的换行符并返回文件名列表
    return [line.strip() for line in lines]

# 创建输出文件

def DeeplabTXT2YOLO(yolo_txt_dir,deeplab_txt_dir,output_files):
    # 遍历每个输出文件
    for output_file in output_files:
        # 读取对应的文件名列表
        yolo_txt_file = os.path.join(deeplab_txt_dir, output_file)
        Lines = read_file(yolo_txt_file)
        # 写入每个文件的路径
        with open(os.path.join(yolo_txt_dir, output_file), 'a') as f:
            for Line in Lines:
                txt_path = os.path.join(yolo_txt_dir, 'labels', output_file[:-4],Line + '.txt')
                f.write(txt_path + '\n')

    print("完成")

#读取每个文件列表
def read_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# 复制文件
def copy_files(source_files, dest_dir):
    for file in source_files:
        shutil.copy(file, dest_dir)
def copy_txt_png(coco_dir,source_dir,output_files):
    # 读取每个文件列表
    for file_name in output_files:
        # 获取文件名列表
        file_list = read_file(os.path.join(coco_dir, file_name))
        # 复制txt文件到对应文件夹
        txt_names=[os.path.basename(file) for file in file_list]
        txt_arr=[os.path.join(source_dir, 'txt', file)for file in txt_names]
        png_arr=[os.path.join(source_dir, 'png', file[:-4]+'.jpg')for file in txt_names]
        copy_files(txt_arr, os.path.join(coco_dir, 'labels', file_name.split('.')[0]))
        # 复制图片到对应文件夹
        copy_files(png_arr, os.path.join(coco_dir, 'images', file_name.split('.')[0]))

    print("完成")

# 设置路径
ROOT=r"E:\Suyingcai\STV_MNet\code"
yoloBASEDIR = ROOT+r"\STV_MNet\yolov3-master\data\labelsSegment"
yolo_txt_dir = yoloBASEDIR+r'\MyCOCO'
deeplab_txt_dir = ROOT+r'\code\STV_MNet\deeplab\deeplearning\ImageSets\Segmentation'
output_files = ['test.txt', 'train.txt', 'trainval.txt', 'val.txt']
DeeplabTXT2YOLO(yolo_txt_dir,deeplab_txt_dir,output_files)

copy_txt_png(yolo_txt_dir,yoloBASEDIR,output_files)