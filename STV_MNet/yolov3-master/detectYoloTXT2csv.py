import os
import csv

# 定义函数将中心点坐标和宽度高度转换为左上角和右下角坐标
def xywh_to_left_top_right_bottom(xywh, image_width=2048, image_height=1024):
    center_x, center_y, width, height = xywh
    left = (center_x - width / 2) * image_width
    top = (center_y - height / 2) * image_height
    right = (center_x + width / 2) * image_width
    bottom = (center_y + height / 2) * image_height
    return left, top, right, bottom

# 输入文件夹路径
input_folder = r'E:\Suyingcai\StreetView\ultralytics_miou\ultralytics_miou\runs\segment\predict_lunan\box_labels'
# 输出文件夹路径
output_folder = r'E:\Suyingcai\StreetView\ultralytics_miou\ultralytics_miou\runs\segment\predict_lunan\csv'

# 遍历每个txt文件，生成对应的csv文件
for txt_file_name in os.listdir(input_folder):
    # 检查文件后缀名是否为txt
    if txt_file_name.endswith('.txt'):
        txt_file_path = os.path.join(input_folder, txt_file_name)
        csv_file_path = os.path.join(output_folder, os.path.splitext(txt_file_name)[0] + '.csv')

        # 创建并写入CSV文件
        with open(txt_file_path, 'r') as txt_file, open(csv_file_path, 'w', newline='') as csv_file:
            txt_reader = csv.reader(txt_file, delimiter=' ')
            csv_writer = csv.writer(csv_file)

            # 写入CSV头部
            csv_writer.writerow(["ID", "left", "top", "right", "bottom", "class"])

            # 解析每一行，并将数据写入CSV文件
            for idx, line in enumerate(txt_reader):
                # cls, center_x, center_y, width, height,score = map(float, line)
                cls, center_x, center_y, width, height = map(float, line)
                left, bottom, right,top  = xywh_to_left_top_right_bottom((center_x, center_y, width, height))
                csv_writer.writerow([idx, left, top, right, bottom, "Tree"])
